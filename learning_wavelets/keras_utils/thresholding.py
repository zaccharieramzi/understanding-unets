from tensorflow.keras.callbacks import Callback
from tensorflow.keras.constraints import Constraint, NonNeg, MaxNorm
from tensorflow.keras.layers import Layer, ReLU, Activation, Conv2D, ThresholdedReLU
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf


class AlphaConstraint(Constraint):
    def __init__(self, max_value=5):
        self.max_value = max_value

    def __call__(self, w):
        norms = K.sqrt(tf.math.reduce_sum(tf.math.square(w)))
        desired = K.clip(norms, 0, self.max_value)
        max_norm_constrained_w = w * (desired / (K.epsilon() + norms))
        non_neg_w = max_norm_constrained_w * tf.dtypes.cast(tf.math.greater_equal(max_norm_constrained_w, 0.), K.floatx())
        return non_neg_w

    def get_config(self):
        return {'max_value': self.max_value}

class DynamicHardThresholding(Layer):
    __name__ = 'dynamic_soft_thresholding'

    def __init__(self, alpha_init, trainable=False, **kwargs):
        super(DynamicHardThresholding, self).__init__(**kwargs)
        self.alpha_init = alpha_init
        self.trainable = trainable

    def build(self, input_shape):
        def _alpha_intializer(shape, **kwargs):
            return tf.ones(shape) * self.alpha_init
        # TODO: set constraints on alpha, and potentially have it be varying along the channels
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=_alpha_intializer,
            trainable=self.trainable,
        )

    def call(self, inputs, weights_mode=False):
        image, noise_std = inputs
        threshold = self.alpha * noise_std
        if not weights_mode:
            threshold = tf.expand_dims(threshold, axis=-1)
            threshold = tf.expand_dims(threshold, axis=-1)
        input_sign = K.sign(image)
        soft_thresh_unsigned = ReLU()(input_sign * image - threshold)
        hard_thresh_unsigned = soft_thresh_unsigned + K.sign(soft_thresh_unsigned) * threshold
        hard_thresh = hard_thresh_unsigned * input_sign
        return hard_thresh

    def get_config(self):
        config = super(DynamicHardThresholding, self).get_config()
        config.update({
            'alpha_init': self.alpha_init,
            'trainable': self.trainable,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

class CheekyDynamicHardThresholding(Layer):
    __name__ = 'cheeky_dynamic_soft_thresholding'

    def __init__(self, alpha_init, **kwargs):
        super(CheekyDynamicHardThresholding, self).__init__(**kwargs)
        self.alpha_init = alpha_init

    def build(self, input_shape):
        def _alpha_thresh_intializer(shape, **kwargs):
            return tf.ones(shape) * self.alpha_init
        def _alpha_bias_intializer(shape, **kwargs):
            return tf.ones(shape) * (self.alpha_init -1.0)
        self.alpha_thresh = self.add_weight(
            shape=(1,),
            initializer=_alpha_thresh_intializer,
            trainable=True,
            constraint=AlphaConstraint(5.0),
            name='thresh',
        )
        self.alpha_bias = self.add_weight(
            shape=(1,),
            initializer=_alpha_bias_intializer,
            trainable=True,
            constraint=AlphaConstraint(5.0),
            name='bias',
        )

    def call(self, inputs, weights_mode=False):
        image, noise_std = inputs
        threshold = self.alpha_thresh * noise_std
        if not weights_mode:
            threshold = tf.expand_dims(threshold, axis=-1)
            threshold = tf.expand_dims(threshold, axis=-1)
        bias = self.alpha_bias * noise_std
        if not weights_mode:
            bias = tf.expand_dims(bias, axis=-1)
            bias = tf.expand_dims(bias, axis=-1)
        input_sign = K.sign(image)
        soft_thresh_unsigned = ReLU()(input_sign * image - threshold)
        hard_thresh_unsigned = soft_thresh_unsigned + K.sign(soft_thresh_unsigned) * bias
        hard_thresh = hard_thresh_unsigned * input_sign
        return hard_thresh

    def get_config(self):
        config = super(CheekyDynamicHardThresholding, self).get_config()
        config.update({
            'alpha_init': self.alpha_init,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class DynamicSoftThresholding(Layer):
    __name__ = 'dynamic_soft_thresholding'

    def __init__(self, alpha_init, trainable=False, per_filter=False, **kwargs):
        super(DynamicSoftThresholding, self).__init__(**kwargs)
        self.alpha_init = alpha_init
        self.trainable = trainable
        self.per_filter = per_filter

    def build(self, input_shapes):
        def _alpha_intializer(shape, **kwargs):
            return tf.ones(shape) * self.alpha_init
        if self.per_filter:
            shape = (input_shapes[0][-1],)
        else:
            shape = (1,)
        self.alpha = self.add_weight(
            shape=shape,
            initializer=_alpha_intializer,
            trainable=self.trainable,
            constraint=AlphaConstraint(5.0),
        )

    def call(self, inputs, weights_mode=False):
        image, noise_std = inputs
        threshold = self.alpha * noise_std
        if not weights_mode:
            threshold = tf.expand_dims(threshold, axis=1)
            threshold = tf.expand_dims(threshold, axis=1)
        input_sign = K.sign(image)
        soft_thresh_unsigned = ReLU()(input_sign * image - threshold)
        soft_thresh = soft_thresh_unsigned * input_sign
        return soft_thresh

    def get_config(self):
        config = super(DynamicSoftThresholding, self).get_config()
        config.update({
            'alpha_init': self.alpha_init,
            'trainable': self.trainable,
            'per_filter': self.per_filter,
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

class LocalWienerFiltering(Layer):
    __name__ = 'local_wiener_filtering'

    def __init__(self, kernel_size=3, **kwargs):
        super(LocalWienerFiltering, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        def _mean_initializer(shape, **kwargs):
            return tf.ones(shape) / tf.dtypes.cast(tf.math.reduce_prod(shape), 'float32')
        self.local_mean = Conv2D(
            1,
            self.kernel_size,
            padding='same',
            use_bias=False,
            kernel_initializer=_mean_initializer,
        )
        self.local_mean.trainable = False

    def call(self, inputs):
        image, noise_std = inputs
        threshold = noise_std ** 2
        threshold = tf.expand_dims(threshold, axis=-1)
        threshold = tf.expand_dims(threshold, axis=-1)
        image_variance = image ** 2
        image_variance_local_mean = self.local_mean(image_variance)
        wiener_coefficients = tf.nn.relu(image_variance_local_mean - threshold) / image_variance_local_mean
        filtered_image = image * wiener_coefficients
        return filtered_image

    def get_config(self):
        config = super(LocalWienerFiltering, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

class RelaxedDynamicHardThresholding(Layer):
    __name__ = 'relaxed_dynamic_soft_thresholding'

    def __init__(self, alpha_init, mu=1e-3, trainable=False, **kwargs):
        super(RelaxedDynamicHardThresholding, self).__init__(**kwargs)
        self.alpha_init = alpha_init
        self.trainable = trainable
        self.mu = mu

    def build(self, input_shape):
        def _alpha_intializer(shape, **kwargs):
            return tf.ones(shape) * self.alpha_init
        # TODO: set constraints on alpha, and potentially have it be varying along the channels
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=_alpha_intializer,
            trainable=self.trainable,
            constraint=AlphaConstraint(5.0),
        )

    def call(self, inputs):
        image, noise_std = inputs
        threshold = self.alpha * noise_std
        threshold = tf.expand_dims(threshold, axis=-1)
        threshold = tf.expand_dims(threshold, axis=-1)
        first_exp = tf.exp((-image + threshold)/self.mu)
        second_exp = tf.exp((-image - threshold)/self.mu)
        relaxed_ht = (1 / (1+first_exp) - 1 / (1+second_exp) + 1) * image
        return relaxed_ht

    def get_config(self):
        config = super(RelaxedDynamicHardThresholding, self).get_config()
        config.update({
            'alpha_init': self.alpha_init,
            'mu': self.mu,
            'trainable': self.trainable,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
