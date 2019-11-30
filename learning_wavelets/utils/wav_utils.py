# those stds were computed with a noise of std 30/255
WAV_STDS = [0.10474847, 0.01995609, 0.008383126, 0.004030478, 0.0020313154]
WAV_STDS = [wav_std / (30 / 255) for wav_std in WAV_STDS]

def get_wavelet_filters_normalisation(n_scales):
    if n_scales > len(WAV_STDS):
        raise ValueError('The number of scales is higher than the number of pre-computed normalisation factors')
    wav_filters_norm = WAV_STDS[:n_scales]
    return wav_filters_norm
