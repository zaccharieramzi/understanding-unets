import pytest

from learning_wavelets.models.learned_wavelet import learnlet

def test_init():
    learnlet(input_size=(256, 256, 1))
