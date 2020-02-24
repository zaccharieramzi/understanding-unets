import pytest

from learning_wavelets.models.learnlet_model import Learnlet

@pytest.mark.parametrize('learnlet_kwargs',[
    {},
    {'n_reweights_learn': 3},
    {'exact_reconstruction': True},
])
def test_init(learnlet_kwargs):
    Learnlet(**learnlet_kwargs)

@pytest.mark.parametrize('learnlet_kwargs',[
    {},
    {'n_reweights_learn': 3},
    {'exact_reconstruction': True},
])
def test_build_compile(learnlet_kwargs):
    model = Learnlet(**learnlet_kwargs)
    model.compile(
        optimizer='adam',
        loss='mse',
    )
    model.build([(None, 32, 32, 1), (None, 1)])

# TODO: add test for exact reconstruction
