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
        image_noisy = inputs[0]
        noise_std = inputs[1]
        learnlet_analysis_coeffs = self.analysis(image_noisy)
        details = learnlet_analysis_coeffs[:-1]
        coarse = learnlet_analysis_coeffs[-1]
        learnlet_analysis_coeffs_thresholded = self.threshold([details, noise_std])
        learnlet_analysis_coeffs_thresholded.append(coarse)
        denoised_image = self.synthesis(learnlet_analysis_coeffs_thresholded)
        if self.clip:
            denoised_image = tf.clip_by_value(denoised_image, clip_value_min=-0.5, clip_value_max=0.5)
        return denoised_image
