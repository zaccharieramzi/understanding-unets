import pytest

from learning_wavelets.models.unet import unet


def test_init():
    unet(n_layers=3, layers_n_non_lins=2)
