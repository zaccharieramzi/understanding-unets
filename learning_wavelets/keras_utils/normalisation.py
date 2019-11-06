from keras import backend as K
from keras.layers import Layer

class NormalisationLayer(Layer):
    def __init__(self, norm_init, **kwargs):
        self.norm_init = norm_init
        super(NormalisationLayer, self).__init__(**kwargs)

    def layer_initializer(self, shape, dtype=None):
        return self.norm_init

    def build(self, input_shape):
        self.norm = self.add_weight(
            name='norm',
            shape=(1,),
            initializer=self.layer_initializer,
            trainable=False,
        )
        super(NormalisationLayer, self).build(input_shape)

    def call(self, x):
        return x / self.norm

    def compute_output_shape(self, input_shape):
        return input_shape
