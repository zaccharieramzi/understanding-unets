import tensorflow as tf

from learning_wavelets.keras_utils.normalisation import Normalisation, NormalisationAdjustment
from learning_wavelets.models.learnlet_model import Learnlet

class TestNormalisation(tf.test.TestCase):
    def test_init(self):
        Normalisation(1)

    def test_norm(self):
        norm_factor = 2
        norm_layer = Normalisation(norm_factor)
        data = tf.random.uniform((10,), maxval=4, seed=0)
        res_data = norm_layer(data)
        self.assertAllClose(res_data, data / norm_factor)
        res_data_inv = norm_layer(data, mode='inv')
        self.assertAllClose(res_data_inv, data * norm_factor)

class TestNormalisationAdjustment:
    def test_init(self):
        NormalisationAdjustment()

    def test_on_batch_end_call(self):
        n_scales = 2
        model = Learnlet(normalize=True, n_scales=n_scales)
        model.build([(None, None, None, 1), (None, 1)])
        norm_cback = NormalisationAdjustment(n_pooling=n_scales)
        norm_cback.set_model(model)
        norm_cback.on_batch_end(None)
