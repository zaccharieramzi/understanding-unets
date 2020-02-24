import pytest
import tensorflow as tf
import tensorflow.keras.backend as K

from learning_wavelets.models.learnlet_model import Learnlet

learnlet_test_cases = [
    {},
    {'n_reweights_learn': 3},
    # TODO: maybe change in subclassed model to have a check for these 2 params when doing exact recon
    {'exact_reconstruction': True, 'learnlet_synthesis_kwargs': {'res': True}, 'learnlet_analysis_kwargs': {'skip_connection': True}},
]

@pytest.mark.parametrize('learnlet_kwargs', learnlet_test_cases)
def test_init(learnlet_kwargs):
    Learnlet(**learnlet_kwargs)
    K.clear_session()

@pytest.mark.parametrize('learnlet_kwargs', learnlet_test_cases)
def test_build_compile(learnlet_kwargs):
    model = Learnlet(**learnlet_kwargs)
    model.compile(
        optimizer='adam',
        loss='mse',
    )
    model.build([(None, 32, 32, 1), (None, 1)])
    K.clear_session()

@pytest.mark.parametrize('learnlet_kwargs', learnlet_test_cases)
def test_fit(learnlet_kwargs):
    model = Learnlet(**learnlet_kwargs)
    model.compile(
        optimizer='adam',
        loss='mse',
    )
    model.build([(None, 32, 32, 1), (None, 1)])
    model.fit(
        x=[tf.random.normal((8, 32, 32, 1), seed=0), tf.zeros((8, 1))],
        y=tf.random.normal((8, 32, 32, 1), seed=0),
    )
    K.clear_session()

# TODO: add test for exact reconstruction
