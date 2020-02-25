import tensorflow as tf

from learning_wavelets.keras_utils.thresholding import DynamicSoftThresholding


class TestDynamicSoftThresholding(tf.test.TestCase):
    def test_init(self):
        DynamicSoftThresholding(alpha_init=1.0)

    def test_soft_thresh(self):
        alpha = 1.0
        dst_layer = DynamicSoftThresholding(alpha_init=alpha)
        n_samples = 10
        noise_level = 0.5
        noise_std = tf.ones((n_samples, 1)) * noise_level
        # dynamic soft thresholding works with images
        data = tf.linspace(-1., 1., n_samples)[:, None, None, None]
        res_data = dst_layer([data, noise_std])
        expected_res_data = tf.where(
            tf.math.abs(data) > noise_level * alpha,
            (tf.math.abs(data) - noise_level * alpha) * tf.sign(data),
            tf.zeros_like(data),
        )
        self.assertAllClose(res_data, expected_res_data)
