import numpy as np
from tensorflow.keras.callbacks import Callback


class NormalizeWeights(Callback):
    # NOTE: for now only normalizes the individual kernels
    # the normalization is not done on an output feature map basis.
    # That is, when you have an input with more than 1 channel, the filters
    # applied to this input will be normalized by channel
    def __init__(self, layers_names_to_be_normalized=None):
        super().__init__()
        self.layers_names_to_be_normalized = layers_names_to_be_normalized

    def set_model(self, model):
        self.model = model
        if self.layers_names_to_be_normalized is None:
            self.layers_names_to_be_normalized = []
            for layer in model.layers:
                if layer.weights:
                    self.layers_names_to_be_normalized.append(layer.name)

    def on_batch_end(self, batch, logs={}):
        for layer_name in self.layers_names_to_be_normalized:
            layer = self.model.get_layer(name=layer_name)
            weights, biases = layer.get_weights()
            weights_norm = np.linalg.norm(weights, axis=(0, 1), keepdims=True)
            weights_normalized = weights / weights_norm
            layer.set_weights((weights_normalized, biases))
