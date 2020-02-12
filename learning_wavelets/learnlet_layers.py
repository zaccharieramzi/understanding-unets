import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.layers import Layer, Activation, Conv2D, Concatenate

from .keras_utils import Normalisation, DynamicSoftThresholding, DynamicHardThresholding, CheekyDynamicHardThresholding, FixedPointPooling, FixedPointUpSampling, BiorUpSampling
from .utils.wav_utils import get_wavelet_filters_normalisation


h_filter_bior = [
    -6.453888262893856e-02,
    -4.068941760955867e-02,
    4.180922732222124e-01,
    7.884856164056651e-01,
    4.180922732222124e-01,
    -4.068941760955867e-02,
    -6.453888262893856e-02,
]

h_filter_starlet = [1/16, 1/4, 3/8, 1/4, 1/16]

class WavPooling(Layer):
    __name__ = 'wav_pooling'
    def __init__(self, wav_type='starlet'):
        super(WavPooling, self).__init__()
        self.wav_type = wav_type
        if self.wav_type == 'starlet':
            base_filter = h_filter_starlet
        elif self.wav_type == 'bior':
            base_filter = h_filter_bior
        else:
            raise ValueError(f'Wavelet type {self.wav_type} is not implemented')
        wav_h_filter = np.array([
            [i * j for j in base_filter]
            for i in base_filter
        ])
        def h_kernel_initializer(shape, **kwargs):
            return wav_h_filter[..., None, None]
        h_prefix = 'low_pass_filtering'
        self.conv_h = Conv2D(
            1,
            # TODO: check that wav_h_filter is square
            wav_h_filter.shape[0],
            activation='linear',
            padding='valid',
            kernel_initializer=h_kernel_initializer,
            use_bias=False,
            name=f'{h_prefix}_{str(K.get_uid(h_prefix))}',
        )
        self.conv_h.trainable = False
        pad_length = len(base_filter)//2
        self.pad = tf.constant([
            [0, 0],
            [pad_length, pad_length],
            [pad_length, pad_length],
            [0, 0],
        ])
        g_prefix = 'high_pass_filtering'
        self.down = FixedPointPooling()
        if self.wav_type == 'starlet':
            self.up = FixedPointUpSampling()
        elif self.wav_type == 'bior':
            self.up = BiorUpSampling()


    def call(self, images):
        padded_images = tf.pad(images, self.pad, 'SYMMETRIC')
        low_freqs = self.conv_h(padded_images)
        high_freqs = images - self.up(self.down(low_freqs))
        return [low_freqs, high_freqs]

    def get_config(self):
        config = super(WavPooling, self).get_config()
        config.update({
            'wav_type': self.wav_type,
        })
        return config

class WavAnalysis(Layer):
    __name__ = 'wav_analysis'
    def __init__(self, n_scales=4, coarse=False, normalize=True, wav_type='starlet'):
        super(WavAnalysis, self).__init__()
        self.wav_type = wav_type
        self.wav_pooling = WavPooling(wav_type=self.wav_type)
        self.pooling = FixedPointPooling()
        self.normalize = normalize
        self.n_scales = n_scales
        self.coarse = coarse
        if self.normalize:
            self.wav_filters_norm = get_wavelet_filters_normalisation(self.n_scales, wav_type=wav_type)

    def call(self, image):
        low_freqs = image
        wav_coeffs = list()
        for i_scale in range(self.n_scales):
            low_freqs, high_freqs = self.wav_pooling(low_freqs)
            if self.normalize:
                wav_norm = self.wav_filters_norm[i_scale]
                high_freqs = high_freqs / wav_norm
            wav_coeffs.append(high_freqs)
            low_freqs = self.pooling(low_freqs)
        if self.coarse:
            wav_coeffs.append(low_freqs)
        return wav_coeffs

    def get_config(self):
        config = super(WavAnalysis, self).get_config()
        config.update({
            'wav_type': self.wav_type,
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
            kernel_size=5,
            skip_connection=False,
            **wav_analysis_kwargs,
        ):
        super(LearnletAnalysis, self).__init__()
        self.n_tiling = n_tiling
        self.tiling_use_bias = tiling_use_bias
        self.tiling_unit_norm = tiling_unit_norm
        self.mixing_details = mixing_details
        self.n_scales = n_scales
        self.kernel_size = kernel_size
        self.skip_connection = skip_connection
        self.wav_analysis = WavAnalysis(coarse=True, n_scales=self.n_scales, **wav_analysis_kwargs)
        constraint = None
        if self.tiling_unit_norm:
            constraint = UnitNorm(axis=[0, 1, 2])
        tiling_prefix = 'details_tiling'
        self.convs_detail_tiling = [
            Conv2D(
                n_tiling,
                self.kernel_size,
                activation='linear',
                padding='same',
                kernel_initializer='glorot_uniform',
                use_bias=tiling_use_bias,
                kernel_constraint=constraint,
                name=f'{tiling_prefix}_{str(K.get_uid(tiling_prefix))}',
            ) for i in range(self.n_scales)
        ]
        if self.mixing_details:
            mixing_prefix = 'details_mixing'
            self.convs_detail_mixing = [
                Conv2D(
                    n_tiling,
                    self.kernel_size,
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
            if self.skip_connection:
                details_tiled = Concatenate()([details_tiled, wav_detail])
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
            'kernel_size': self.kernel_size,
            'skip_connection': self.skip_connection,
        })
        return config

class LearnletSynthesis(Layer):
    __name__ = 'learnlet_synthesis'
    def __init__(self, normalize=True, n_scales=4, n_channels=1, synthesis_use_bias=False, synthesis_norm=False, res=False, kernel_size=5, wav_type='starlet'):
        super(LearnletSynthesis, self).__init__()
        self.normalize = normalize
        self.n_scales = n_scales
        self.n_channels = n_channels
        self.synthesis_use_bias = synthesis_use_bias
        self.synthesis_norm = synthesis_norm
        self.res = res
        self.kernel_size = kernel_size
        self.wav_type = wav_type
        if self.normalize:
            self.wav_filters_norm = get_wavelet_filters_normalisation(self.n_scales, wav_type=self.wav_type)
            self.wav_filters_norm.reverse()
        if self.wav_type == 'starlet':
            self.upsampling = FixedPointUpSampling()
        elif self.wav_type == 'bior':
            self.upsampling = BiorUpSampling()
        else:
            raise ValueError(f'Wavelet type {self.wav_type} is not implemented for upsampling')
        constraint = None
        if self.synthesis_norm:
            constraint = UnitNorm(axis=[0, 1, 2])
        groupping_prefix = 'groupping_conv'
        self.convs_groupping = [
            Conv2D(
                n_channels,
                self.kernel_size,
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
            image = self.upsampling(image)
            if self.normalize:
                wav_norm = self.wav_filters_norm[i_scale]
                detail = detail * wav_norm
            if self.res:
                image = self.convs_groupping[i_scale](detail) + image
            else:
                image = self.convs_groupping[i_scale](tf.concat([image, detail], axis=-1))
        return image

    def get_config(self):
        config = super(LearnletSynthesis, self).get_config()
        config.update({
            'normalize': self.normalize,
            'n_scales': self.n_scales,
            'n_channels': self.n_channels,
            'synthesis_use_bias': self.synthesis_use_bias,
            'synthesis_norm': self.synthesis_norm,
            'res': self.res,
            'kernel_size': self.kernel_size,
            'wav_type': self.wav_type,
        })
        return config

class ScalesThreshold(Layer):
    def __init__(self, noise_std_norm=True, dynamic_denoising=False, denoising_activation='relu', n_scales=2):
        super(ScalesThreshold, self).__init__()
        self.noise_std_norm = noise_std_norm
        self.dynamic_denoising = dynamic_denoising
        self.denoising_activation = denoising_activation
        self.n_scales = n_scales
        if not self.dynamic_denoising:
            self.thresholding_layer = Activation(self.denoising_activation, name='thresholding')
        if self.noise_std_norm:
            self.normalisation_layers = [Normalisation(1.0) for i in range(self.n_scales)]
        if self.denoising_activation == 'dynamic_soft_thresholding':
            self.thresholding_layers = [DynamicSoftThresholding(2.0, trainable=True) for i in range(self.n_scales)]
        if self.denoising_activation == 'dynamic_hard_thresholding':
            self.thresholding_layers = [DynamicHardThresholding(3.0, trainable=False) for i in range(self.n_scales)]
        elif self.denoising_activation == 'cheeky_dynamic_hard_thresholding':
            self.thresholding_layers = [CheekyDynamicHardThresholding(3.0, trainable=True) for i in range(self.n_scales)]

    def call(self, inputs, weights=None):
        if self.dynamic_denoising:
            details, noise_std = inputs
        else:
            details = inputs
        if weights is None:
            weights = tf.ones_like(details)
        details_thresholded = list()
        for i_scale, (detail, weight) in enumerate(zip(details, weights)):
            if self.noise_std_norm:
                normalisation_layer = self.normalisation_layers[i_scale]
                detail = normalisation_layer(detail, mode='normal')
            weight = noise_std * weight
            if self.dynamic_denoising:
                if isinstance(self.denoising_activation, str):
                    thresholding_layer = self.thresholding_layers[i_scale]
                    detail_thresholded = thresholding_layer([detail, weight])
                else:
                    detail_thresholded = self.denoising_activation([detail, weight])
            else:
                detail_thresholded = self.thresholding_layer(detail)
            if self.noise_std_norm:
                detail_thresholded = normalisation_layer(detail_thresholded, mode='inv')
            details_thresholded.append(detail_thresholded)
        return details_thresholded

    def normalize(self, coefficients):
        coefficients_normalized = [
            normalisation_layer(coefficient, mode='normal')
            for (coefficient, normalisation_layer) in zip(coefficients, self.normalisation_layers)
        ]
        return coefficients_normalized

    def get_config(self):
        config = super(ScalesThreshold, self).get_config()
        config.update({
            'noise_std_norm': self.noise_std_norm,
            'dynamic_denoising': self.dynamic_denoising,
            'denoising_activation': self.denoising_activation,
            'n_scales': self.n_scales,
        })
        return config
