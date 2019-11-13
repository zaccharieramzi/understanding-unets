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
        type_of_multiresolution_transform=float(wavelet_id),
        sigma_noise=noise_std,
        number_of_scales=n_scales,
        verbose=0,
        tab_n_sigma=n_sigma,
    )
    denoised_images = list()
    for noisy_image in noisy_images:
        flt.filter(noisy_image)
        denoised_image = flt.data
        denoised_images.append(denoised_image)
    return denoised_images

def threshold_wavelet_coefficients(data, wav_filters, noise_std):
    threshold_per_level = np.empty((len(wav_filters), *data.shape))
    for i_filter, wav_filter in enumerate(wav_filters):
        threshold_per_level[i_filter] = 3 * noise_std * np.linalg.norm(wav_filter)
    threshold_per_level[-1] = 0
    wav_dec = filter_convolve(data, wav_filters)
    wav_dec_threshed = np.maximum((1.0 - threshold_per_level / np.maximum(np.finfo(np.float64).eps, np.abs(wav_dec))), 0.0) * wav_dec
    recon_data = np.sum(wav_dec_threshed, axis=0)
    return recon_data

def wavelet_denoising(noisy_image, noise_std, wavelet_id='2', n_scales=2):
    denoised_image = np.empty_like(noisy_image)
    for i_channel in range(noisy_image.shape[-1]):
        noisy_channel = noisy_image[..., i_channel]
        # from IPython.core.debugger import set_trace; set_trace()
        wav_filters = get_mr_filters(noisy_channel.shape, opt=[f'-t {wavelet_id}', f'-n {n_scales}'], coarse=True)
        denoised_channel = threshold_wavelet_coefficients(noisy_channel, wav_filters, noise_std)
        denoised_image[..., i_channel] = denoised_channel
    return denoised_image
