from tensorflow.keras.layers import Layer

class FixedPointPooling(Layer):
    def call(self, image_batch):
        return image_batch[:, ::2, ::2, :]
