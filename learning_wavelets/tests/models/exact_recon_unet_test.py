import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from learning_wavelets.models.exact_recon_unet import ExactReconUnet


def test_init():
    ExactReconUnet()
    K.clear_session()

def test_fit():
    model=ExactReconUnet(n_output_channels=1, kernel_size=3, layers_n_channels=[4, 8])
    model.compile(optimizer=Adam(lr=1e-3), loss='mse')
    model.fit(
        x=(
        	tf.random.normal((8, 32, 32, 1), seed=0),
            tf.random.normal((8, 1), seed=0),
        ),
        y=tf.random.normal((8, 32, 32, 1), seed=0),
    )
    K.clear_session()
