from keras.callbacks import Callback
from keras.layers import Layer, ThresholdedReLU, ReLU
from keras.models import Model
from keras import backend as K
import numpy as np


class SoftThresholding(Layer):
    __name__ = 'soft_thresholding'

    def __init__(self, threshold, **kwargs):
        super(SoftThresholding, self).__init__(**kwargs)
        self.threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = K.cast_to_floatx(threshold)

    def call(self, inputs):
        input_sign = K.sign(inputs)
        soft_thresh_unsigned = ReLU()(input_sign * inputs - self.threshold)
        soft_thresh = soft_thresh_unsigned * input_sign
        return soft_thresh

    def get_config(self):
        config = super(SoftThresholding, self).get_config()
        config.update({'threshold': float(self.threshold)})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

class ThresholdAdjustment(Callback):
    def __init__(self, noise_std, n=2):
        super().__init__()
        self.noise_std = noise_std
        self.n = n

    def set_model(self, model):
        self.model = model
        self.soft_thresholds = list()  # list the soft thresh layers
        soft_threshold_input_model_outputs = list()
        for layer in model.layers:
            if isinstance(layer, SoftThresholding):
                self.soft_thresholds.append(layer)
                # this is from https://stackoverflow.com/a/50858709/4332585
                soft_threshold_input = layer._inbound_nodes[0].inbound_layers[0]
                soft_threshold_input_model_outputs.append(soft_threshold_input)
        self.soft_thresholds_input_model = Model(model.input, soft_threshold_input_model_outputs)

    def on_batch_end(self, batch, logs={}):
        image_shape = list(self.model.input_shape[1:])
        image_shape = [1] + image_shape
        noise = np.random.normal(scale=self.noise_std, size=image_shape)
        soft_threshold_inputs = self.soft_thresholds_input_model.predict_on_batch(noise)
        for soft_threshold_layer, soft_threshold_input in zip(self.soft_thresholds, soft_threshold_inputs):
            # TODO: compute the input average std and set the new threshold to
            # be n times that
            pass
