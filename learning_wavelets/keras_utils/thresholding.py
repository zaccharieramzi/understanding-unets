import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer, ThresholdedReLU, ReLU, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf


class DynamicSoftThresholding(Layer):
    __name__ = 'dynamic_soft_thresholding'

    def __init__(self, alpha_init, trainable=False, **kwargs):
        super(DynamicSoftThresholding, self).__init__(**kwargs)
        self.alpha_init = alpha_init
        self.trainable = trainable

    def build(self, input_shape):
        def _alpha_intializer(shape, **kwargs):
            return tf.ones(shape) * self.alpha_init
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=_alpha_intializer,
            trainable=self.trainable,
        )

    def call(self, inputs):
        image, noise_std = inputs
        threshold = self.alpha * noise_std
        input_sign = K.sign(image)
        soft_thresh_unsigned = ReLU()(input_sign * inputs - threshold)
        soft_thresh = soft_thresh_unsigned * input_sign
        return soft_thresh

    def get_config(self):
        config = super(DynamicSoftThresholding, self).get_config()
        config.update({
            'alpha_init': self.alpha_init,
            'trainable': self.trainable,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

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

class HardThresholding(Layer):
    __name__ = 'hard_thresholding'

    def __init__(self, threshold, **kwargs):
        super(HardThresholding, self).__init__(**kwargs)
        self.threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = K.cast_to_floatx(threshold)

    def call(self, inputs):
        input_sign = K.sign(inputs)
        hard_thresh_unsigned = ThresholdedReLU(self.threshold)(input_sign * inputs)
        hard_thresh = hard_thresh_unsigned * input_sign
        return hard_thresh

    def get_config(self):
        config = super(HardThresholding, self).get_config()
        config.update({'threshold': float(self.threshold)})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class ThresholdAdjustment(Callback):
    def __init__(self, noise_std, n=2, n_pooling=4):
        super().__init__()
        self.noise_std = noise_std
        self.n = n
        # 4 as a minimum just to make sure we have enough samples to compute
        # a reliable std
        self.n_pooling = min(n_pooling, 4) + 3

    def set_model(self, model):
        self.model = model
        self.sts = list()  # list the soft thresh layers
        st_input_model_outputs = list()
        for layer in model.layers:
            if isinstance(layer, Activation) and isinstance(layer.activation, SoftThresholding):
                self.sts.append(layer)
                # this is from https://stackoverflow.com/a/50858709/4332585
                st_input = layer._inbound_nodes[0].inbound_layers[0].output
                st_input_model_outputs.append(st_input)
        self.sts_input_model = Model(model.input, st_input_model_outputs)

    def on_batch_end(self, batch, logs={}):
        n_channels = self.model.input_shape[-1]
        image_shape = [1, 2**self.n_pooling, 2**self.n_pooling]
        image_shape.append(n_channels)
        noise = np.random.normal(scale=self.noise_std, size=image_shape)
        st_inputs = self.sts_input_model.predict_on_batch(noise)
        for st_layer, st_input in zip(self.sts, st_inputs):
            st_input = st_input[0]
            n_channels = st_input.shape[-1]
            average_std = np.mean([np.std(st_input[..., i]) for i in range(n_channels)])
            st_layer.activation.threshold = self.n * average_std
