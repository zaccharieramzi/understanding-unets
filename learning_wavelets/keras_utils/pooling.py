import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import gen_image_ops


h_tilde_filter = np.array([
    3.782845550699535e-02,
    -2.384946501937986e-02,
    -1.106244044184226e-01,
    3.774028556126536e-01,
    8.526986790094022e-01,
    3.774028556126537e-01,
    -1.106244044184226e-01,
    -2.384946501937986e-02,
    3.782845550699535e-02,
])

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

class BiorUpSampling(Layer):
    def call(self, image_batch):
        core_im_shape = tf.shape(image_batch)[1:3]
        up_sampled_im_shape = tf.Variable(tf.shape(image_batch))
        up_sampled_im_shape[1:3].assign(2*core_im_shape)
        upsampled_images = tf.Variable(tf.zeros(up_sampled_im_shape))
        upsampled_images[:, ::2, ::2, :].assign(image_batch)
        #TODO: maybe deal with padding here
        line_conv_upsampled = tf.nn.conv2d(
            upsampled_images,
            tf.constant(h_tilde_filter[None, ..., None, None], dtype='float'),
            strides=1,
            padding='SAME',
        )
        column_conv_upsampled = tf.nn.conv2d(
            line_conv_upsampled,
            tf.constant(h_tilde_filter[..., None, None, None], dtype='float'),
            strides=1,
            padding='SAME',
        )
        return column_conv_upsampled
