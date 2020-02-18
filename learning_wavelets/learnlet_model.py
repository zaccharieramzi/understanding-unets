import tensorflow as tf
from tensorflow.keras.models import Model

from .keras_utils import DynamicSoftThresholding, DynamicHardThresholding
from .learnlet_layers import LearnletAnalysis, LearnletSynthesis, ScalesThreshold

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
        self.analysis = LearnletAnalysis(
            normalize=self.normalize,
            n_scales=self.n_scales,
            **learnlet_analysis_kwargs,
        )
        self.threshold = ScalesThreshold(
            n_scales=self.n_scales,
            dynamic_denoising=True,
            denoising_activation=self.denoising_activation,
            **threshold_kwargs,
        )
        self.synthesis = LearnletSynthesis(
            normalize=self.normalize,
            n_scales=self.n_scales,
            **learnlet_synthesis_kwargs,
        )

    def call(self, inputs):
        return self.reweighting(inputs, n_reweights=self.n_reweights_learn)

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
