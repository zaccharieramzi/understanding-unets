import tensorflow.keras.backend as K

from learning_wavelets.models.unet import unet


def test_init():
    unet(n_layers=3, layers_n_non_lins=2)
    K.clear_session()
