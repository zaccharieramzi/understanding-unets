# those stds were computed with a noise of std 1.0
WAV_STDS = [0.9466215372085571, 0.23304526507854462, 0.10223511606454849, 0.049666628241539, 0.02393055334687233]

def get_wavelet_filters_normalisation(n_scales):
    if n_scales > len(WAV_STDS):
        raise ValueError('The number of scales is higher than the number of pre-computed normalisation factors')
    wav_filters_norm = WAV_STDS[:n_scales]
    return wav_filters_norm
