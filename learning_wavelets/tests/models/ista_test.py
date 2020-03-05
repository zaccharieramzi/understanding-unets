import tensorflow as tf

from learning_wavelets.data.toy_datasets import from_image_to_masked_kspace_and_mask
from learning_wavelets.keras_utils.fourier import tf_masked_shifted_normed_fft2d, tf_masked_shifted_normed_ifft2d
from learning_wavelets.models.ista import IstaLearnlet

class TestIsta(tf.test.TestCase):
    def test_no_iterations(self):
        model = IstaLearnlet(
            n_iterations=0,
            forward_operator=tf_masked_shifted_normed_fft2d,
            adjoint_operator=tf_masked_shifted_normed_ifft2d,
        )
        im_size = 64
        image = tf.random.normal([im_size, im_size])
        (kspace_channeled, mask), _ = from_image_to_masked_kspace_and_mask(af=2.0)(image)
        res_model = model([kspace_channeled, mask])
        zero_filled = tf.math.abs(tf_masked_shifted_normed_ifft2d(kspace_channeled[..., 0], mask)[..., None])
        self.assertAllClose(zero_filled, res_model)
