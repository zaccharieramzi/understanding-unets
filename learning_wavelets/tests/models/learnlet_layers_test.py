import tensorflow as tf

from learning_wavelets.models.learnlet_layers import WavAnalysis

def test_exact_reconstruction():
    wav_analysis_layer = WavAnalysis(normalize=False, coarse=True)
    upsampling = wav_analysis_layer.wav_pooling.up
    image = tf.random.uniform((1, 32, 32, 1), maxval=1, seed=0)
    wav_coeffs = wav_analysis_layer(image)
    wav_coeffs.reverse()
    res_image = None
    for wav_coeff in wav_coeffs:
        if res_image is None:
            res_image = wav_coeff
        else:
            res_image = upsampling(res_image) + wav_coeff
    res_psnr = tf.image.psnr(image, res_image, 1.).numpy()
    assert res_psnr > 100
