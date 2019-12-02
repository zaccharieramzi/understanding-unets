import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.layers import Layer, Conv2D, AveragePooling2D, UpSampling2D

from .utils.wav_utils import get_wavelet_filters_normalisation


class WavPooling(Layer):
    def __init__(self):
        super(WavPooling, self).__init__()
        base_filter = [1/16, 1/4, 3/8, 1/4, 1/16]
        wav_h_filter = np.array([
            [i * j for j in base_filter]
            for i in base_filter
        ])
        filter_shape = wav_h_filter.shape
        filter_center_idx = filter_shape[0]//2
        wav_g_filter = -wav_h_filter
        wav_g_filter[filter_center_idx, filter_center_idx] = 0
        wav_g_filter[filter_center_idx, filter_center_idx] = -np.sum(wav_g_filter)
        def h_kernel_initializer(shape, **kwargs):
            return wav_h_filter[..., None, None]
        def g_kernel_initializer(shape, **kwargs):
            return wav_g_filter[..., None, None]
        h_prefix = 'low_pass_filtering'
        self.conv_h = Conv2D(
            1,
            # TODO: check that wav_h_filter is square
            wav_h_filter.shape[0],
            activation='linear',
            padding='same',
            kernel_initializer=h_kernel_initializer,
            use_bias=False,
            name=f'{h_prefix}_{str(K.get_uid(h_prefix))}',
        )
        self.conv_h.trainable = False
        g_prefix = 'high_pass_filtering'
        self.conv_g = Conv2D(
            1,
            # TODO: check that wav_g_filter is square
            wav_g_filter.shape[0],
            activation='linear',
            padding='same',
            kernel_initializer=g_kernel_initializer,
            use_bias=False,
            name=f'{g_prefix}_{str(K.get_uid(g_prefix))}',
        )
        self.conv_g.trainable = False


    def call(self, image):
        low_freqs = self.conv_h(image)
        high_freqs = self.conv_g(image)
        return [low_freqs, high_freqs]

class WavAnalysis(Layer):
    def __init__(self, n_scales=4, coarse=False, normalize=True):
        super(WavAnalysis, self).__init__()
        self.wav_pooling = WavPooling()
        self.pooling = AveragePooling2D()
        self.normalize = normalize
        self.n_scales = n_scales
        self.coarse = coarse
        if self.normalize:
            self.wav_filters_norm = get_wavelet_filters_normalisation(self.n_scales)

    def call(self, image):
        low_freqs = image
        wav_coeffs = list()
        for i_scale in range(self.n_scales):
            low_freqs, high_freqs = self.wav_pooling(low_freqs)
            if self.normalize:
                wav_norm = self.wav_filters_norm[i_scale]
                high_freqs = high_freqs / wav_norm
            wav_coeffs.append(high_freqs)
            if i_scale < self.n_scales - 1:
                low_freqs = self.pooling(low_freqs)
        if self.coarse:
            wav_coeffs.append(low_freqs)
        return wav_coeffs

    def get_config(self):
        config = super(WavAnalysis, self).get_config()
        config.update({
            'n_scales': self.n_scales,
            'coarse': self.coarse,
            'normalize': self.normalize,
        })
        return config

class LearnletAnalysis(Layer):
    def __init__(
            self,
            n_tiling=3,
            tiling_use_bias=False,
            tiling_unit_norm=True,
            mixing_details=False,
            n_scales=4,
            **wav_analysis_kwargs,
        ):
        super(LearnletAnalysis, self).__init__()
        self.n_tiling = n_tiling
        self.tiling_use_bias = tiling_use_bias
        self.tiling_unit_norm = tiling_unit_norm
        self.mixing_details = mixing_details
        self.n_scales = n_scales
        self.wav_analysis = WavAnalysis(coarse=True, n_scales=self.n_scales, **wav_analysis_kwargs)
        constraint = None
        if self.tiling_unit_norm:
            constraint = UnitNorm(axis=[0, 1, 2])
        tiling_prefix = 'details_tiling'
        self.convs_detail_tiling = [
            Conv2D(
                n_tiling,
                5,
                activation='linear',
                padding='same',
                kernel_initializer='glorot_uniform',
                use_bias=tiling_use_bias,
                kernel_constraint=constraint,
                name=f'{tiling_prefix}_{str(K.get_uid(tiling_prefix))}',
            ) for i in range(self.n_scales)
        ]
        if self.mixing_details:
            mixing_prefix = 'details_tiling'
            self.convs_detail_mixing = [
                Conv2D(
                    n_tiling,
                    5,
                    activation='linear',
                    padding='same',
                    kernel_initializer='glorot_uniform',
                    use_bias=tiling_use_bias,
                    kernel_constraint=constraint,
                    name=f'{mixing_prefix}_{str(K.get_uid(mixing_prefix))}',
                ) for i in range(self.n_scales)
            ]

    def call(self, image):
        wav_coeffs = self.wav_analysis(image)
        wav_details = wav_coeffs[:-1]
        wav_coarse = wav_coeffs[-1]
        outputs_list = []
        for i_scale, wav_detail in enumerate(wav_details):
            details_tiled = self.convs_detail_tiling[i_scale](wav_detail)
            if self.mixing_details:
                details_tiled = self.convs_detail_mixing[i_scale](details_tiled)
            outputs_list.append(details_tiled)
        outputs_list.append(wav_coarse)
        return outputs_list

    def get_config(self):
        config = super(LearnletAnalysis, self).get_config()
        config.update({
            'n_tiling': self.n_tiling,
            'tiling_use_bias': self.tiling_use_bias,
            'tiling_unit_norm': self.tiling_unit_norm,
            'mixing_details': self.mixing_details,
            'n_scales': self.n_scales,
        })
        return config

class LearnletSynthesis(Layer):
    def __init__(self, normalize=True, n_scales=4, n_channels=1, synthesis_use_bias=False, synthesis_norm=False, res=False):
        super(LearnletSynthesis, self).__init__()
        self.normalize = normalize
        self.n_scales = n_scales
        self.n_channels = n_channels
        self.synthesis_use_bias = synthesis_use_bias
        self.synthesis_norm = synthesis_norm
        self.res = res
        if self.normalize:
            self.wav_filters_norm = get_wavelet_filters_normalisation(self.n_scales)
            self.wav_filters_norm.reverse()
        self.upsampling = UpSampling2D(size=(2, 2))
        constraint = None
        if self.synthesis_norm:
            constraint = UnitNorm(axis=[0, 1, 2])
        groupping_prefix = 'groupping_conv'
        self.convs_groupping = [
            Conv2D(
                n_channels,
                5,
                activation='linear',
                padding='same',
                kernel_initializer='glorot_uniform',
                use_bias=synthesis_use_bias,
                kernel_constraint=constraint,
                name=f'{groupping_prefix}_{str(K.get_uid(groupping_prefix))}',
            ) for i in range(self.n_scales)
        ]

    def call(self, analysis_coeffs):
        details = analysis_coeffs[:-1]
        details.reverse()
        coarse = analysis_coeffs[-1]
        image = coarse
        for i_scale, detail in enumerate(details):
            if self.normalize:
                wav_norm = self.wav_filters_norm[i_scale]
                detail = detail * wav_norm
            if self.res:
                image = self.convs_groupping[i_scale](detail) + image
            else:
                image = self.convs_groupping[i_scale](tf.concat([image, detail], axis=-1))
            if i_scale < len(details) - 1:
                image = self.upsampling(image)
        return image

    def get_config(self):
        config = super(LearnletSynthesis, self).get_config()
        config.update({
            'normalize': self.normalize,
            'n_scales': self.n_scales,
            'n_channels': self.n_channels,
            'synthesis_use_bias': self.synthesis_use_bias,
            'synthesis_norm': self.synthesis_norm,
        })
        return config
