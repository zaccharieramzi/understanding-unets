from keras.constraints import UnitNorm
from keras.layers import Activation, Conv2D, AveragePooling2D, UpSampling2D, Lambda
import numpy as np


def conv_2d(image, n_channels, kernel_size=3, activation='relu', bias=True, norm=False):
    constraint = None
    if norm:
        constraint = UnitNorm(axis=[0, 1, 2])
    image = Conv2D(
        n_channels,
        kernel_size,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
        bias=bias,
        kernel_constraint=constraint,
    )(image)
    image = Activation(activation)(image)
    return image

def wavelet_pooling(image, wav_h_filter=None):
    # TODO: refactor to have it as a shared weight in the network
    # less memory (even if not very large)
    def kernel_initializer(shape, **kwargs):
        # TODO: check that shape is correspdonding
        return wav_h_filter[..., None, None]
    if wav_h_filter is None:
        base_filter = [1/16, 1/4, 3/8, 1/4, 1/16]
        wav_h_filter = np.array([
            [i * j for j in base_filter]
            for i in base_filter
        ])
    conv_h = Conv2D(
        1,
        # TODO: check that wav_h_filter is square
        wav_h_filter.shape[0],
        activation='linear',
        padding='same',
        kernel_initializer=kernel_initializer,
        bias=False,
    )
    conv_h.trainable = False
    low_freqs = conv_h(image)
    low_freqs_downsampled = AveragePooling2D()(low_freqs)
    low_freqs_down_up = UpSampling2D()(low_freqs_downsampled)
    high_freqs = Lambda(lambda x: x[0] - x[1])([image, low_freqs_down_up])
    return [low_freqs, high_freqs]