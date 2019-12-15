import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import gen_image_ops


class FixedPointPooling(Layer):
    def call(self, image_batch):
        return image_batch[:, ::2, ::2, :]

class FixedPointUpSampling(Layer):
    def call(self, image_batch):
        initial_shape = tf.shape(image_batch)
        target_shape = 2*initial_shape[1:3]
        resized_image = gen_image_ops.resize_bicubic(
            image_batch,
            target_shape,
            align_corners=False,
            half_pixel_centers=False,
        )
        return resized_image
