import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, concatenate, UpSampling2D, Input, AveragePooling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .evaluate import keras_psnr, keras_ssim
from .keras_utils import Normalisation, conv_2d, wavelet_pooling, DynamicSoftThresholding
from .learnlet_layers import LearnletAnalysis, LearnletSynthesis
from .utils.wav_utils import get_wavelet_filters_normalisation


def learnlet(
        input_size,
        lr=1e-3,
        denoising_activation='relu',
        noise_std_norm=True,
        normalize=True,
        n_scales=4,
        exact_reconstruction_weight=0,
        learnlet_analysis_kwargs=None,
        learnlet_synthesis_kwargs=None,
    ):
    image_noisy = Input(input_size)
    if learnlet_analysis_kwargs is None:
        learnlet_analysis_kwargs = {}
    if learnlet_synthesis_kwargs is None:
        learnlet_synthesis_kwargs = {}
    dynamic_denoising = False
    if isinstance(denoising_activation, DynamicSoftThresholding):
        dynamic_denoising = True
        noise_std = Input((1,))
    learnlet_analysis_layer = LearnletAnalysis(
        normalize=normalize,
        n_scales=n_scales,
        **learnlet_analysis_kwargs,
    )
    learnlet_analysis_coeffs = learnlet_analysis_layer(image_noisy)
    details = learnlet_analysis_coeffs[:-1]
    coarse = learnlet_analysis_coeffs[-1]
    thresholding_layer = Activation(denoising_activation, name='thresholding')
    learnlet_analysis_coeffs_thresholded = list()
    for detail in details:
        if noise_std_norm:
            normalisation_layer = Normalisation(1.0)
            detail = normalisation_layer(detail, mode='normal')
        if dynamic_denoising:
            detail_thresholded = thresholding_layer([detail, noise_std])
        else:
            detail_thresholded = thresholding_layer(detail)
        if noise_std_norm:
            detail_thresholded = normalisation_layer(detail_thresholded, mode='inv')
        learnlet_analysis_coeffs_thresholded.append(detail_thresholded)
    learnlet_analysis_coeffs_thresholded.append(coarse)
    learnlet_synthesis_layer = LearnletSynthesis(
        normalize=normalize,
        n_scales=n_scales,
        **learnlet_synthesis_kwargs,
    )
    denoised_image = learnlet_synthesis_layer(learnlet_analysis_coeffs_thresholded)
    if dynamic_denoising:
        learnlet_model = Model([image_noisy, noise_std], denoised_image)
    else:
        learnlet_model = Model(image_noisy, denoised_image)
    if exact_reconstruction_weight:
        # TODO: make exact reconstruction adaptable to dynamic denoising
        image = Input(input_size)
        learnlet_analysis_coeffs_exact = learnlet_analysis_layer(image)
        reconstructed_image = learnlet_synthesis_layer(learnlet_analysis_coeffs_exact)
        learnlet_model = Model([image_noisy, image], [denoised_image, reconstructed_image])
        learnlet_model.compile(
            optimizer=Adam(lr=lr),
            loss=['mse', 'mse'],
            loss_weights=[1, exact_reconstruction_weight],
            metrics=[[keras_psnr, keras_ssim]]*2,
        )
    else:
        learnlet_model.compile(
            optimizer=Adam(lr=lr),
            loss='mse',
            metrics=[keras_psnr, keras_ssim],
        )
    return learnlet_model


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
                kernel_size=5,
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
