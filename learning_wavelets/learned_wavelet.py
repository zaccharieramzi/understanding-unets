from keras.layers import Conv2D, concatenate, UpSampling2D, Input, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

from .evaluate import keras_psnr, keras_ssim

def learned_wavelet_rec(image, n_scales=1, n_details=3, n_coarse=1, n_groupping=3):
    n_channel = int(image.shape[-1])
    details_thresholded = Conv2D(
        n_details,
        3,
        activation='relu',
        padding='same',
        kernel_initializer='glorot_uniform',
    )(image)
    coarse = Conv2D(
        n_coarse,
        3,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
    )(image)
    if n_scales > 1:
        coarse_down_sampled = AveragePooling2D()(coarse)
        denoised_coarse = learned_wavelet_rec(
            coarse_down_sampled,
            n_scales=n_scales-1,
            n_details=n_details,
            n_coarse=n_coarse,
            n_groupping=n_groupping,
        )
        denoised_coarse_upsampled = UpSampling2D(size=(2, 2))(denoised_coarse)
    else:
        # NOTE: potentially allow to have thresholding (i.e. non linearity) also on the coarse
        # scale
        denoised_coarse_upsampled = coarse
    denoised_image = Conv2D(
        n_groupping * n_channel,
        3,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
    )(concatenate([denoised_coarse_upsampled, details_thresholded]))
    denoised_image = Conv2D(
        n_channel,
        1,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
    )(denoised_image)
    return denoised_image


def learned_wavelet(input_size, lr=1e-4, n_scales=4, n_details=3, n_coarse=1, n_groupping=3):
    image = Input(input_size)
    denoised_image = learned_wavelet_rec(
        image,
        n_scales=n_scales,
        n_details=n_details,
        n_coarse=n_coarse,
        n_groupping=n_groupping,
    )
    model = Model(inputs=image, outputs=denoised_image)
    model.compile(
        optimizer=Adam(lr=lr, clipnorm=1.),
        loss='mean_squared_error',
        metrics=[keras_psnr, keras_ssim],
    )
    return model
