import tensorflow.keras.backend as K
from tensorflow.keras.layers import concatenate, UpSampling2D, Input, AveragePooling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .evaluate import keras_psnr, keras_ssim
from .keras_utils.conv import conv_2d, wavelet_pooling

# those stds were computed with a noise of std 30/255
WAV_STDS = [0.10474847, 0.01995609, 0.008383126, 0.004030478, 0.0020313154]
WAV_STDS = [wav_std / (30 / 255) for wav_std in WAV_STDS]


def wav_analysis_model(input_size, n_scales=4, coarse=False, normalize=True):
    image = Input(input_size)
    low_freqs = image
    wav_coeffs = list()
    if normalize:
        wav_filters_norm = get_wavelet_filters_normalisation(n_scales)
    for i_scale in range(n_scales):
        low_freqs, high_freqs = wavelet_pooling(low_freqs)
        if normalize:
            wav_norm = wav_filters_norm[i_scale]
            prefix = 'wav_normalisation'
            name = f'{prefix}_{str(K.get_uid(prefix))}'
            high_freqs = Lambda(lambda x: x / wav_norm, name=name)(high_freqs)
        wav_coeffs.append(high_freqs)
        if i_scale < n_scales - 1:
            low_freqs = AveragePooling2D()(low_freqs)
    if coarse:
        wav_coeffs.append(low_freqs)
    model = Model(image, wav_coeffs)
    # model.compile(optimizer='adam', loss='mse')
    return model

def learnlet_analysis(
        input_size,
        n_tiling=3,
        tiling_use_bias=False,
        tiling_unit_norm=True,
        mixing_details=False,
        **wav_analysis_kwargs,
    ):
    image = Input(input_size)
    wav_analysis_net = wav_analysis_model(input_size, coarse=True, **wav_analysis_kwargs)
    wav_coeffs = wav_analysis_net(image)
    wav_details = wav_coeffs[:-1]
    wav_coarse = wav_coeffs[-1]
    outputs_list = []
    for wav_detail in wav_details:
        details_tiled = conv_2d(
            wav_detail,
            n_tiling,
            activation=None,
            bias=tiling_use_bias,
            unit_norm=tiling_unit_norm,
            noise_std_norm=False,
            name='details_tiling',
        )
        if mixing_details:
            details_tiled = conv_2d(
                details_tiled,
                n_tiling,
                activation=None,
                bias=tiling_use_bias,
                unit_norm=tiling_unit_norm,
                noise_std_norm=False,
                name='details_mixing',
            )
        outputs_list.append(details_tiled)
    outputs_list.append(wav_coarse)
    model = Model(image, outputs_list)
    return wav_analysis_net, model

def learnlet_synthesis(analysis_coeffs, normalize=True, synthesis_use_bias=False, groupping_norm=False):
    details = analysis_coeffs[:-1]
    coarse = analysis_coeffs[-1]
    image = coarse
    n_channels = image.shape[-1]
    if normalize:
        wav_filters_norm = get_wavelet_filters_normalisation(len(analysis_coeffs))
    for i_scale, detail in enumerate(details):
        if normalize:
            wav_norm = wav_filters_norm[i_scale]
            prefix = 'wav_denormalisation'
            name = f'{prefix}_{str(K.get_uid(prefix))}'
            detail = Lambda(lambda x: x * wav_norm, name=name)(detail)
        image = conv_2d(
            concatenate([image, detail]),
            n_channels,
            activation=None,
            bias=synthesis_use_bias,
            name='groupping_conv',
            unit_norm=groupping_norm,
        )
        image = UpSampling2D(size=(2, 2))(image)
    model = Model(analysis_coeffs, image)
    return model

def get_wavelet_filters_normalisation(n_scales):
    if n_scales > len(WAV_STDS):
        raise ValueError('The number of scales is higher than the number of pre-computed normalisation factors')
    wav_filters_norm = WAV_STDS[:n_scales]
    return wav_filters_norm


# legacy learned wavelet
def learned_wavelet_rec(
        image,
        n_scales=1,
        n_details=3,
        n_coarse=1,
        mixing_details=False,
        denoising_activation='relu',
        wav_pooling=False,
        wav_use_bias=True,
        wav_filters_norm=None,
        filters_normed=[],
    ):
    n_channel = int(image.shape[-1])
    if filters_normed is None:
        filters_normed = ['details', 'coarse', 'groupping']
    if wav_pooling:
        low_freqs, high_freqs = wavelet_pooling(image)
        coarse = low_freqs
        if wav_filters_norm:
            prefix = 'wav_normalisation'
            name = f'{prefix}_{str(K.get_uid(prefix))}'
            wav_norm = wav_filters_norm.pop(0)
            high_freqs = Lambda(lambda x: x / wav_norm, name=name)(high_freqs)
        else:
            wav_norm = None
        details_activation = denoising_activation
        if mixing_details:
            details_activation = 'linear'
        details_thresholded = conv_2d(
            high_freqs,
            n_details,
            kernel_size=5,
            activation=details_activation,
            bias=wav_use_bias,
            unit_norm='details' in filters_normed,
            noise_std_norm=not mixing_details,
            name='details_tiling',
        )
        if mixing_details:
            details_thresholded = conv_2d(
                details_thresholded,
                n_details,
                activation=denoising_activation,
                bias=wav_use_bias,
                unit_norm='details' in filters_normed,
                noise_std_norm=True,
                name='details_mixing',
            )
        if wav_norm is not None:
            prefix = 'wav_denormalisation'
            name = f'{prefix}_{str(K.get_uid(prefix))}'
            details_thresholded = Lambda(lambda x: x * wav_norm, name=name)(details_thresholded)
    else:
        details_thresholded = conv_2d(
            image,
            n_details,
            activation=denoising_activation,
            unit_norm='details' in filters_normed,
            name='high_pass_filtering',
        )
        coarse = conv_2d(
            image,
            n_coarse,
            activation='linear',
            unit_norm='coarse' in filters_normed,
            name='low_pass_filtering',
        )
    if n_scales > 1:
        coarse_down_sampled = AveragePooling2D()(coarse)
        denoised_coarse = learned_wavelet_rec(
            coarse_down_sampled,
            n_scales=n_scales-1,
            n_details=n_details,
            n_coarse=n_coarse,
            mixing_details=mixing_details,
            denoising_activation=denoising_activation,
            filters_normed=filters_normed,
            wav_pooling=wav_pooling,
            wav_filters_norm=wav_filters_norm,
            wav_use_bias=wav_use_bias,
        )
        denoised_coarse_upsampled = UpSampling2D(size=(2, 2))(denoised_coarse)
    else:
        # NOTE: potentially allow to have thresholding (i.e. non linearity) also on the coarse
        # scale
        denoised_coarse_upsampled = coarse
    bias = not wav_pooling or (wav_pooling and wav_use_bias)
    denoised_image = conv_2d(
        concatenate([denoised_coarse_upsampled, details_thresholded]),
        n_channel,
        activation='linear',
        kernel_size=5,
        unit_norm='groupping' in filters_normed,
        name='groupping_conv',
        bias=bias,
    )
    return denoised_image

def learned_wavelet(
        input_size,
        lr=1e-4,
        n_scales=4,
        n_details=3,
        n_coarse=1,
        mixing_details=False,
        denoising_activation='relu',
        wav_pooling=False,
        wav_use_bias=True,
        wav_normed=False,
        filters_normed=[],
    ):
    image = Input(input_size)
    wav_filters_norm = None
    if wav_normed:
        wav_filters_norm = get_wavelet_filters_normalisation(n_scales)
    denoised_image = learned_wavelet_rec(
        image,
        n_scales=n_scales,
        n_details=n_details,
        n_coarse=n_coarse,
        mixing_details=mixing_details,
        denoising_activation=denoising_activation,
        wav_pooling=wav_pooling,
        wav_use_bias=wav_use_bias,
        wav_filters_norm=wav_filters_norm,
        filters_normed=filters_normed,
    )
    model = Model(inputs=image, outputs=denoised_image)
    model.compile(
        optimizer=Adam(lr=lr),
        loss='mean_squared_error',
        metrics=[keras_psnr, keras_ssim],
    )
    return model
