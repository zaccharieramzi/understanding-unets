from keras.callbacks import Callback
from keras.layers import Layer
from keras.models import Model
import numpy as np


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

    def call(self, x):
        return x / self.norm

    def compute_output_shape(self, input_shape):
        return input_shape

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
        self.normalisation_layers = list()  # list the soft thresh layers
        norm_input_model_outputs = list()
        for layer in model.layers:
            if isinstance(layer, Normalisation):
                self.normalisation_layers.append(layer)
                # this is from https://stackoverflow.com/a/50858709/4332585
                norm_input = layer._inbound_nodes[0].inbound_layers[0].output
                norm_input_model_outputs.append(norm_input)
                self.current_stds.append(None)
                self.stds_lists.append(list())
        self.norms_input_model = Model(model.input, norm_input_model_outputs)

    def on_batch_end(self, batch, logs={}):
        n_channels = self.model.input_shape[-1]
        image_shape = [1, 2**self.n_pooling, 2**self.n_pooling, n_channels]
        noise = np.random.normal(scale=1.0, size=image_shape)
        norm_inputs = self.norms_input_model.predict_on_batch(noise)
        for i_scale, (norm_layer, norm_input) in enumerate(zip(self.normalisation_layers, norm_inputs)):
            norm_input = norm_input[0]
            n_channels = norm_input.shape[-1]
            stds = np.array([np.std(norm_input[..., i]) for i in range(n_channels)])
            current_std = self.current_stds[i_scale]
            if current_std is None:
                update_stds = stds
            else:
                update_stds = self.momentum * current_std + (1 - self.momentum) * stds
            norm_layer.set_weights([update_stds])
            self.stds_lists[i_scale].append(update_stds)
            self.current_stds[i_scale] = update_stds
