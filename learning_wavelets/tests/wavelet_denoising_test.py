import numpy as np
import pytest

try:
    from learning_wavelets.wavelet_denoising import wavelet_denoising_pysap
except ModuleNotFoundError:
    pysap_unavail = True
else:
    pysap_unavail = False

@pytest.mark.skipif(pysap_unavail, reason='Pysap unavailable')
def test_wavelet_denoising_pysap_call():
    image = np.random.rand(32, 32)
    wavelet_denoising_pysap([image], 1)
