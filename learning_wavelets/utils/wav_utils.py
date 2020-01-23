# those stds were computed with a noise of std 1.0
WAV_STDS_STARLET = [0.9173831343650818, 0.21372182667255402, 0.0925946980714798, 0.04539244994521141, 0.022488027811050415, 0.012274044, 0.006508918]
WAV_STDS_BIOR = [0.8673086762428284, 0.8324918150901794, 0.8871174454689026, 0.9202567338943481, 0.9301353096961975]

def get_wavelet_filters_normalisation(n_scales, wav_type='starlet'):
    if wav_type == 'starlet':
        wav_stds = WAV_STDS_STARLET
    elif wav_type == 'bior':
        wav_stds = WAV_STDS_BIOR
    else:
        raise ValueError(f'The normalisation is not computed for the wavelet type {wav_type}')
    if n_scales > len(wav_stds):
        raise ValueError('The number of scales is higher than the number of pre-computed normalisation factors')
    wav_filters_norm = wav_stds[:n_scales]
    return wav_filters_norm
