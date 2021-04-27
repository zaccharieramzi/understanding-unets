import pytest
import tensorflow as tf
import tensorflow.keras.backend as K

from learning_wavelets.models.unet import unet


def test_init():
    unet(n_layers=3, layers_n_non_lins=2)
    K.clear_session()

def test_fit():
    model = unet(input_size=(None, None, 1), n_layers=3, layers_n_non_lins=2)
    model.fit(
        x=tf.random.normal((8, 32, 32, 1), seed=0),
        y=tf.random.normal((8, 32, 32, 1), seed=0),
    )
    K.clear_session()
