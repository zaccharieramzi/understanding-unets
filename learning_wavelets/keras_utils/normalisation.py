import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import tensorflow as tf


class Normalisation(Layer):
    def __init__(self, norm_init, **kwargs):
        self.norm_init = norm_init
        super(Normalisation, self).__init__(**kwargs)

    def layer_initializer(self, shape, dtype=None):
        norm_init = np.array([self.norm_init]*shape[0])
        return norm_init

    def build(self, input_shape):
        n_channels = input_shape[-1]
        self.norm = self.add_weight(
            name='norm',
            shape=(n_channels,),
            initializer=self.layer_initializer,
            trainable=False,
        )
        super(Normalisation, self).build(input_shape)

    def call(self, x, mode='normal'):
        if mode == 'normal':
            return x / self.norm
        elif mode == 'inv':
            return x * self.norm
        else:
            raise ValueError('Mode not recognized')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {'norm_init': self.norm_init}


class NormalisationAdjustment(Callback):
    def __init__(self, n_pooling=4, momentum=0.9):
        super().__init__()
        # 4 as a minimum just to make sure we have enough samples to compute
        # a reliable std
        self.n_pooling = min(n_pooling, 4) + 3
        self.current_stds = list()
        self.stds_lists = list()  # this is just for monitoring/ debugging
        self.momentum = momentum

    def set_model(self, model):
        self.model = model
        if self.model.name == 'learnlet':
            self.mode = 'subclassing'
            self.current_stds = [None] * self.model.n_scales
            self.stds_lists = [list()] * self.model.n_scales
            return
        self.mode = 'normal'
        self.normalisation_layers = list()  # list the soft thresh layers
        norms_input_layer = model.get_layer(name='learnlet_analysis')
        self.norms_input_model = Model(model.input[0], norms_input_layer(model.input[0])[:-1])
        for layer in model.layers:
            if isinstance(layer, Normalisation) and layer not in self.normalisation_layers:
                self.normalisation_layers.append(layer)
                self.current_stds.append(None)
                self.stds_lists.append(list())

    def on_batch_end(self, batch, logs={}):
        # NOTE: for now we only support grey images
        image_shape = [1, 2**self.n_pooling, 2**self.n_pooling, 1]
        if self.mode == 'subclassing':
            noise = tf.random.normal(stddev=1.0, shape=image_shape)
            norm_inputs = [norm_input.numpy() for norm_input in self.model.compute_coefficients(noise, normalized=False, coarse=False)]
        else:
            noise = np.random.normal(scale=1.0, size=image_shape)
            norm_inputs = self.norms_input_model.predict_on_batch(noise)
        for i_scale, norm_input in enumerate(norm_inputs):
            norm_input = norm_input[0]
            n_channels = norm_input.shape[-1]
            stds = np.array([np.std(norm_input[..., i]) for i in range(n_channels)])
            current_std = self.current_stds[i_scale]
            if current_std is None:
                update_stds = stds
            else:
                update_stds = self.momentum * current_std + (1 - self.momentum) * stds
            if self.mode == 'subclassing':
                self.model.update_normalisation(i_scale, update_stds)
            else:
                self.normalisation_layers[i_scale].set_weights([update_stds])
            self.stds_lists[i_scale].append(update_stds)
            self.current_stds[i_scale] = update_stds
