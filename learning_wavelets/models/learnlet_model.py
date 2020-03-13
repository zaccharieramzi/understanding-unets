import tensorflow as tf
from tensorflow.keras.models import Model

from ..keras_utils import DynamicSoftThresholding, DynamicHardThresholding
from .learnlet_layers import LearnletAnalysis, LearnletSynthesis, ScalesThreshold, WavAnalysis

class Learnlet(Model):
    __name__ = 'learnlet'

    def __init__(
            self,
            normalize=True,
            n_scales=2,
            clip=False,
            denoising_activation='dynamic_soft_thresholding',
            learnlet_analysis_kwargs=None,
            learnlet_synthesis_kwargs=None,
            threshold_kwargs=None,
            n_reweights_learn=1,
            exact_reconstruction=False,
            undecimated=False,
            wav_only=False,
            odd_shapes=False,
        ):
        super(Learnlet, self).__init__()
        if learnlet_analysis_kwargs is None:
            learnlet_analysis_kwargs = {}
        if learnlet_synthesis_kwargs is None:
            learnlet_synthesis_kwargs = {}
        if threshold_kwargs is None:
            threshold_kwargs = {}
        self.denoising_activation = denoising_activation
        self.clip = clip
        self.n_scales = n_scales
        self.normalize = normalize
        self.n_reweights_learn = n_reweights_learn
        self.exact_reconstruction = exact_reconstruction
        self.undecimated = undecimated
        self.wav_only = wav_only
        self.odd_shapes = odd_shapes
        if self.wav_only:
            self.analysis = WavAnalysis(
                normalize=self.normalize,
                n_scales=self.n_scales,
                undecimated=self.undecimated,
                **learnlet_analysis_kwargs,
            )
        else:
            self.analysis = LearnletAnalysis(
                normalize=self.normalize,
                n_scales=self.n_scales,
                undecimated=self.undecimated,
                **learnlet_analysis_kwargs,
            )
        self.threshold = ScalesThreshold(
            n_scales=self.n_scales,
            dynamic_denoising=True,
            denoising_activation=self.denoising_activation,
            **threshold_kwargs,
        )
        if self.exact_reconstruction:
            self.threshold_wavelet = ScalesThreshold(
                n_scales=self.n_scales,
                dynamic_denoising=True,
                denoising_activation=self.denoising_activation,
                **threshold_kwargs,
            )
            for thresholding_layer in self.threshold_wavelet.thresholding_layers:
                thresholding_layer.alpha_init += 1
        self.synthesis = LearnletSynthesis(
            normalize=self.normalize,
            wav_filters_norm=self.analysis.get_wav_filters_norm(),
            n_scales=self.n_scales,
            undecimated=self.undecimated,
            wav_only=self.wav_only,
            **learnlet_synthesis_kwargs,
        )

    def call(self, inputs):
        if not self.undecimated:
            inputs, image_shape = self.pad_for_pool(inputs)
        if self.exact_reconstruction:
            image_denoised = self.exact_reconstruction_comp(inputs)
        else:
            image_denoised = self.reweighting(inputs, n_reweights=self.n_reweights_learn)
        if not self.undecimated:
            image_denoised = self.trim_padding(image_denoised, image_shape)
        return image_denoised

    def trim_padding(self, im_shape, image):
        padded_im_shape = image.shape[1:3]
        to_trim = padded_im_shape - im_shape[0]
        trimmed_image = image[:, to_trim[0]//2:-to_trim[0]//2, to_trim[1]//2:-to_trim[1]//2]
        return trimmed_image

    def pad_for_pool(self, inputs):
        image, noise_std = inputs
        im_shape = tf.shape(image)[1:3]
        n_pooling = self.n_scales
        to_pad = (tf.dtypes.cast(im_shape / 2**n_pooling, 'int32') + 1) * 2**n_pooling - im_shape
        # the + 1 is necessary because the images have odd shapes
        if self.odd_shapes:
            pad_seq = [(0, 0), (to_pad[0]//2, to_pad[0]//2 + 1), (to_pad[1]//2, to_pad[1]//2 + 1), (0, 0)]
        else:
            pad_seq = [(0, 0), (to_pad[0]//2, to_pad[0]//2), (to_pad[1]//2, to_pad[1]//2), (0, 0)]
        image_padded = tf.pad(image, pad_seq, 'SYMMETRIC')
        return (image_padded, noise_std), im_shape

    def compute_coefficients(self, images, normalized=True, coarse=False):
        learnlet_analysis_coeffs = self.analysis(images)
        details = learnlet_analysis_coeffs[:-1]
        if normalized:
            learnlet_analysis_coeffs_normalized = self.threshold.normalize(details)
            if coarse:
                coarse = learnlet_analysis_coeffs[-1]
                learnlet_analysis_coeffs_normalized.append(coarse)
            return learnlet_analysis_coeffs_normalized
        else:
            if coarse:
                return learnlet_analysis_coeffs
            else:
                return details

    def update_normalisation(self, i_scale, update_stds):
        if self.exact_reconstruction:
            classical_norm_layer = self.threshold.normalisation_layers[i_scale]
            classical_norm_layer.set_weights([update_stds[:-1]])
            exact_recon_norm_layer = self.threshold_wavelet.normalisation_layers[i_scale]
            exact_recon_norm_layer.set_weights([update_stds[-2:-1]])
        else:
            norm_layer = self.threshold.normalisation_layers[i_scale]
            norm_layer.set_weights([update_stds])

    def reweighting(self, inputs, n_reweights=3):
        image_noisy = inputs[0]
        noise_std = inputs[1]
        learnlet_analysis_coeffs = self.analysis(image_noisy)
        details = learnlet_analysis_coeffs[:-1]
        coarse = learnlet_analysis_coeffs[-1]
        weights = [tf.ones_like(detail) for detail in details]
        for i in range(n_reweights-1):
            learnlet_analysis_coeffs_thresholded = self.threshold(
                [details, noise_std],
                weights=weights,
                no_back_normalisation=True,
            )
            new_weights = []
            for weight, learnlet_analysis_coeff_thresholded, thresholding_layer in zip(
                    weights, learnlet_analysis_coeffs_thresholded, self.threshold.thresholding_layers
                ):
                expanded_noise = tf.expand_dims(tf.expand_dims(noise_std, axis=1), axis=1)
                expanded_alpha =  tf.expand_dims(tf.expand_dims(tf.expand_dims(thresholding_layer.alpha, axis=0), axis=0), axis=0)
                actual_threshold =  expanded_noise * expanded_alpha * weight
                new_weight = weight / (1 + tf.math.abs(learnlet_analysis_coeff_thresholded) / actual_threshold)
                new_weights.append(new_weight)
            weights = new_weights
        learnlet_analysis_coeffs_thresholded = self.threshold([details, noise_std], weights=weights)
        learnlet_analysis_coeffs_thresholded.append(coarse)
        denoised_image = self.synthesis(learnlet_analysis_coeffs_thresholded)
        if self.clip:
            denoised_image = tf.clip_by_value(denoised_image, clip_value_min=-0.5, clip_value_max=0.5)
        return denoised_image

    def exact_reconstruction_comp(self, inputs):
        image_noisy = inputs[0]
        noise_std = inputs[1]
        learnlet_analysis_coeffs = self.analysis(image_noisy)
        details = learnlet_analysis_coeffs[:-1]
        details_tiled = [detail[..., :-1] for detail in details]
        details_identity = [detail[..., -1:] for detail in details]
        coarse = learnlet_analysis_coeffs[-1]
        learnlet_analysis_coeffs_thresholded = self.threshold([details_tiled, noise_std])
        learnlet_analysis_coeffs_thresholded.append(coarse)
        wav_analysis_coeffs_thresholded = self.threshold_wavelet([details_identity, noise_std])
        wav_analysis_coeffs_thresholded_tiled = self.analysis.tiling(wav_analysis_coeffs_thresholded)
        denoised_image = self.synthesis.exact_reconstruction(
            learnlet_analysis_coeffs_thresholded,
            wav_analysis_coeffs_thresholded,
            wav_analysis_coeffs_thresholded_tiled,
        )
        if self.clip:
            denoised_image = tf.clip_by_value(denoised_image, clip_value_min=-0.5, clip_value_max=0.5)
        return denoised_image
