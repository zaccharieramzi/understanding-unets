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
    __name__ = 'bior_upsampling'
    def __init__(self):
        super(BiorUpSampling, self).__init__()
        pad_length = len(h_tilde_filter)//2
        self.pad = tf.constant([
            [0, 0],
            [pad_length, pad_length],
            [pad_length, pad_length],
            [0, 0],
        ])

    def call(self, image_batch):
        im_shape = tf.shape(image_batch)
        output_shape = im_shape * tf.constant([1, 2, 2, 1])
        upsampled_images = tf.nn.conv2d_transpose(
            image_batch,
            tf.ones([1, 1, 1, 1]),
            output_shape,
            strides=2,
            padding='VALID',
        )
        padded_images = tf.pad(upsampled_images, self.pad, 'SYMMETRIC')
        line_conv_upsampled = tf.nn.conv2d(
            padded_images,
            tf.constant(h_tilde_filter[None, ..., None, None], dtype='float'),
            strides=1,
            padding='VALID',
        )
        column_conv_upsampled = tf.nn.conv2d(
            line_conv_upsampled,
            tf.constant(h_tilde_filter[..., None, None, None], dtype='float'),
            strides=1,
            padding='VALID',
        )
        return column_conv_upsampled
