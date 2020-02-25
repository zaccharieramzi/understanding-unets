import pytest
import tensorflow as tf
import tensorflow.keras.backend as K

from learning_wavelets.models.learned_wavelet import learnlet

def test_init():
    learnlet(input_size=(256, 256, 1))

def test_fit():
    model = learnlet(input_size=(None, None, 1))
    model.fit(
        x=[tf.random.normal((8, 32, 32, 1), seed=0), tf.zeros((8, 1))],
        y=tf.random.normal((8, 32, 32, 1), seed=0),
    )
    K.clear_session()
