from keras.layers import Layer, ThresholdedReLU, ReLU
from keras import backend as K

class SoftThresholding(Layer):
    __name__ = 'soft_thresholding'

    def __init__(self, threshold, **kwargs):
        super(SoftThresholding, self).__init__(**kwargs)
        self.threshold = K.cast_to_floatx(threshold)

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
