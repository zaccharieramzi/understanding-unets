from modopt.signal.wavelet import get_mr_filters, filter_convolve
import numpy as np
from pysap.extensions.sparse2d import Filter


def wavelet_denoising_pysap(noisy_images, noise_std, wavelet_id='2', n_scales=2, soft_thresh=True, n_sigma=2):
    if soft_thresh:
        type_of_filtering = 2
    else:
        type_of_filtering = 1
    flt = Filter(
        type_of_filtering=type_of_filtering,
        type_of_multiresolution_transform=int(wavelet_id),
        sigma_noise=noise_std,
        number_of_scales=n_scales,
        verbose=0,
        tab_n_sigma=[n_sigma],
    )
    denoised_images = list()
    for noisy_image in noisy_images:
        if (noisy_image < 0).any():
        # if False:
            bias = noisy_image.min()
            noisy_image_biased = np.copy(noisy_image) - bias
        else:
            bias = 0
            noisy_image_biased = np.copy(noisy_image)
        flt.filter(noisy_image_biased)
        denoised_image = flt.data.data
        denoised_image += bias
        denoised_images.append(denoised_image)
    return denoised_images
