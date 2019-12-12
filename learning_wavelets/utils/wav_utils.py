# those stds were computed with a noise of std 1.0
WAV_STDS = [0.94388556, 0.20002668, 0.08651136, 0.04172291, 0.020271296]

def get_wavelet_filters_normalisation(n_scales):
    if n_scales > len(WAV_STDS):
        raise ValueError('The number of scales is higher than the number of pre-computed normalisation factors')
    wav_filters_norm = WAV_STDS[:n_scales]
    return wav_filters_norm
