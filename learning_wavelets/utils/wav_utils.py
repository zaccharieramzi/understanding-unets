# those stds were computed with a noise of std 1.0
WAV_STDS = [0.9173831343650818, 0.21372182667255402, 0.0925946980714798, 0.04539244994521141, 0.022488027811050415]

def get_wavelet_filters_normalisation(n_scales):
    if n_scales > len(WAV_STDS):
        raise ValueError('The number of scales is higher than the number of pre-computed normalisation factors')
    wav_filters_norm = WAV_STDS[:n_scales]
    return wav_filters_norm
