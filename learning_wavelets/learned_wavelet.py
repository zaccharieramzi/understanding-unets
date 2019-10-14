from keras.layers import concatenate, UpSampling2D, Input, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD

from .evaluate import keras_psnr, keras_ssim
from .keras_utils.conv import conv_2d

def learned_wavelet_rec(image, n_scales=1, n_details=3, n_coarse=1, n_groupping=3, denoising_activation='relu'):
    n_channel = int(image.shape[-1])
    details_thresholded = conv_2d(image, n_details, activation=denoising_activation)
    coarse = conv_2d(image, n_coarse, activation='linear')
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
    denoised_image = conv_2d(
        concatenate([denoised_coarse_upsampled, details_thresholded]),
        n_groupping * n_channel,
        activation='linear',
    )
    denoised_image = conv_2d(denoised_image, n_channel, kernel_size=1, activation='linear')
    return denoised_image


def learned_wavelet(input_size, lr=1e-4, n_scales=4, n_details=3, n_coarse=1, n_groupping=3, denoising_activation='relu'):
    image = Input(input_size)
    denoised_image = learned_wavelet_rec(
        image,
        n_scales=n_scales,
        n_details=n_details,
        n_coarse=n_coarse,
        n_groupping=n_groupping,
        denoising_activation=denoising_activation,
    )
    model = Model(inputs=image, outputs=denoised_image)
    model.compile(
        optimizer=Adam(lr=lr, clipnorm=1.),
        loss='mean_squared_error',
        metrics=[keras_psnr, keras_ssim],
    )
    return model
