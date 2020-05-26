import tensorflow as tf

from learning_wavelets.models.multiscale import MultiScale


def test_multiscale_init():
    MultiScale(None, 4)

def test_multiscale_pad_trim():
    model = MultiScale(lambda x: x, 4, dynamic_denoising=False)
    image = tf.random.normal([1, 321, 481, 1])
    image_denoised = model(image)
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllEqual(image, image_denoised)
