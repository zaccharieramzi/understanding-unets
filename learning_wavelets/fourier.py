import numpy as np

def fft2(wav_filter):
    fft_coeffs = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(wav_filter), norm='ortho'))
    return fft_coeffs
