from keras.layers import concatenate, UpSampling2D, Input, AveragePooling2D, Lambda
from keras.models import Model
from keras.optimizers import Adam

from .evaluate import keras_psnr, keras_ssim
from .keras_utils.conv import conv_2d, wavelet_pooling, H_normalisation, G_normalisation


def learned_wavelet_rec(
        image,
        n_scales=1,
        n_details=3,
        n_coarse=1,
        n_groupping=3,
        denoising_activation='relu',
        wav_pooling=False,
        wav_normed=False,
        filters_normed=[],
    ):
    n_channel = int(image.shape[-1])
    if filters_normed is None:
        filters_normed = ['details', 'coarse', 'groupping']
    if wav_pooling:
        low_freqs, high_freqs = wavelet_pooling(image, normalized=wav_normed)
        coarse = low_freqs
        norm = False
        if 'details' in filters_normed:
            norm = True
        details_thresholded = conv_2d(
            high_freqs,
            n_details,
            activation=denoising_activation,
            bias=True,
            norm=norm,
            name='details_tiling',
        )
    else:
        norm = False
        if 'details' in filters_normed:
            norm = True
        details_thresholded = conv_2d(
            image,
            n_details,
            activation=denoising_activation,
            norm=norm,
            name='high_pass_filtering',
        )
        norm = False
        if 'coarse' in filters_normed:
            norm = True
        coarse = conv_2d(
            image,
            n_coarse,
            activation='linear',
            norm=norm,
            name='low_pass_filtering',
        )
    if n_scales > 1:
        coarse_down_sampled = AveragePooling2D()(coarse)
        denoised_coarse = learned_wavelet_rec(
            coarse_down_sampled,
            n_scales=n_scales-1,
            n_details=n_details,
            n_coarse=n_coarse,
            n_groupping=n_groupping,
            denoising_activation=denoising_activation,
            filters_normed=filters_normed,
            wav_pooling=wav_pooling,
            wav_normed=wav_normed,
        )
        denoised_coarse_upsampled = UpSampling2D(size=(2, 2))(denoised_coarse)
    else:
        # NOTE: potentially allow to have thresholding (i.e. non linearity) also on the coarse
        # scale
        denoised_coarse_upsampled = coarse
    norm = False
    if 'groupping' in filters_normed:
        norm = True
    if wav_normed:
        denoised_coarse_upsampled = Lambda(lambda x: x * H_normalisation)(denoised_coarse_upsampled)
        details_thresholded = Lambda(lambda x: x * G_normalisation)(details_thresholded)
    denoised_image = conv_2d(
        concatenate([denoised_coarse_upsampled, details_thresholded]),
        n_groupping * n_channel,
        activation='linear',
        norm=norm,
        name='groupping_conv',
    )
    denoised_image = conv_2d(denoised_image, n_channel, kernel_size=1, activation='linear', norm=norm)
    return denoised_image

def learned_wavelet(
        input_size,
        lr=1e-4,
        n_scales=4,
        n_details=3,
        n_coarse=1,
        n_groupping=3,
        denoising_activation='relu',
        wav_pooling=False,
        wav_normed=False,
        filters_normed=[],
    ):
    image = Input(input_size)
    denoised_image = learned_wavelet_rec(
        image,
        n_scales=n_scales,
        n_details=n_details,
        n_coarse=n_coarse,
        n_groupping=n_groupping,
        denoising_activation=denoising_activation,
        wav_pooling=wav_pooling,
        wav_normed=wav_normed,
        filters_normed=filters_normed,
    )
    model = Model(inputs=image, outputs=denoised_image)
    model.compile(
        optimizer=Adam(lr=lr, clipnorm=1.),
        loss='mean_squared_error',
        metrics=[keras_psnr, keras_ssim],
    )
    return model
