import tensorflow as tf
from tensorflow.keras.models import Model


class MultiScale(Model):
    def __init__(self, multiscale_model, n_scales, dynamic_denoising=True, **kwargs):
        super(MultiScale, self).__init__(**kwargs)
        self.model = multiscale_model
        self.n_scales = n_scales
        self.dynamic_denoising = dynamic_denoising

    def pad_for_pool(self, image):
        im_shape = tf.shape(image)[1:3]
        n_pooling = self.n_scales
        to_pad = (tf.cast(im_shape / 2**n_pooling, 'int32') + 1) * 2**n_pooling - im_shape
        # the + 1 is necessary because the images have odd shapes
        # TODO: revise this in the case of non odd shaped images
        pad_seq = [
            (0, 0),  # batch dimension
            (to_pad[0]//2, to_pad[0]//2 + 1),  # H
            (to_pad[1]//2, to_pad[1]//2 + 1),  # W
            (0, 0),  # channel dimension
        ]
        image_padded = tf.pad(image, pad_seq, 'SYMMETRIC')
        return image_padded, im_shape

    def trim_padding(self, im_shape, image):
        padded_im_shape = tf.shape(image)[1:3]
        to_trim = padded_im_shape - im_shape
        trimmed_image = image[
            :,
            to_trim[0]//2:padded_im_shape[0]-(to_trim[0]//2 + 1),
            to_trim[1]//2:padded_im_shape[1]-(to_trim[1]//2 + 1),
            :,
        ]
        return trimmed_image

    def call(self, inputs):
        if self.dynamic_denoising:
            image, noise = inputs
        else:
            image = inputs
        image_padded, im_shape = self.pad_for_pool(image)
        if self.dynamic_denoising:
            image_denoised = self.model([image_padded, noise])
        else:
            image_denoised = self.model(image_padded)
        image_denoised = self.trim_padding(im_shape, image_denoised)
        return image_denoised
