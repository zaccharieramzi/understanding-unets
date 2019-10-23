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

def wavelet_pooling(image, wav_h_filter=None, wav_g_filter=None, normalized=False):
    # TODO: refactor to have it as a shared weight in the network
    # less memory (even if not very large)
    if wav_h_filter is None:
        base_filter = [1/16, 1/4, 3/8, 1/4, 1/16]
        wav_h_filter = np.array([
            [i * j for j in base_filter]
            for i in base_filter
        ])
    filter_shape = wav_h_filter.shape
    if wav_g_filter is None:
        filter_center_idx = filter_shape[0]//2
        wav_g_filter = -wav_h_filter
        wav_g_filter[filter_center_idx, filter_center_idx] = 0
        wav_g_filter[filter_center_idx, filter_center_idx] = -np.sum(wav_g_filter)
    if normalized:
        wav_h_filter /= np.linalg.norm(wav_h_filter)
        wav_g_filter /= np.linalg.norm(wav_g_filter)
    def h_kernel_initializer(shape, **kwargs):
        # TODO: check that shape is correspdonding
        return wav_h_filter[..., None, None]
    def g_kernel_initializer(shape, **kwargs):
        # TODO: check that shape is correspdonding
        return wav_g_filter[..., None, None]
    conv_h = Conv2D(
        1,
        # TODO: check that wav_h_filter is square
        wav_h_filter.shape[0],
        activation='linear',
        padding='same',
        kernel_initializer=h_kernel_initializer,
        bias=False,
    )
    conv_h.trainable = False
    conv_g = Conv2D(
        1,
        # TODO: check that wav_g_filter is square
        wav_g_filter.shape[0],
        activation='linear',
        padding='same',
        kernel_initializer=g_kernel_initializer,
        bias=False,
    )
    conv_g.trainable = False
    low_freqs = conv_h(image)
    high_freqs = conv_g(image)
    return [low_freqs, high_freqs]
