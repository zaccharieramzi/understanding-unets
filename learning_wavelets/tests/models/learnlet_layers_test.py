import pytest
import tensorflow as tf

from learning_wavelets.models.learnlet_layers import WavAnalysis

@pytest.mark.parametrize('undecimated', [True, False])
def test_exact_reconstruction(undecimated):
    wav_analysis_layer = WavAnalysis(normalize=False, coarse=True, undecimated=undecimated)
    upsampling = wav_analysis_layer.wav_pooling.up
    image = tf.random.uniform((1, 32, 32, 1), maxval=1, seed=0)
    wav_coeffs = wav_analysis_layer(image)
    wav_coeffs.reverse()
    res_image = None
    for wav_coeff in wav_coeffs:
        if res_image is None:
            res_image = wav_coeff
        else:
            if not undecimated:
                res_image = upsampling(res_image)
            res_image = res_image + wav_coeff
    res_psnr = tf.image.psnr(image, res_image, 1.).numpy()
    assert res_psnr > 100
