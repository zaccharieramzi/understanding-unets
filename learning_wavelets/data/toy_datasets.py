from skimage.draw import random_shapes
import tensorflow as tf

from ..keras_utils.fourier import tf_masked_shifted_normed_fft2d, tf_masked_shifted_normed_ifft2d
from .utils import gen_mask_tf


def random_ellipse_gen(im_size):
    while True:
        im, _ = random_shapes(
            (im_size, im_size),
            max_shapes=30,
            min_shapes=20,
            multichannel=False,
            min_size=im_size//8,
            max_size=im_size//3,
            shape='ellipse',
            allow_overlap=True,
        )
        im = im / 255
        im = im - 0.5
        yield im

# TODO: refactor following 4 functions
def from_image_to_masked_kspace_and_mask(af=4):
    def _from_image_to_masked_kspace_and_mask(image):
        image = tf.cast(image, 'complex64')
        mask = gen_mask_tf(image, accel_factor=af)
        kspace = tf_masked_shifted_normed_fft2d(image, mask)
        kspace_channeled = kspace[..., None]
        image_channeled = tf.cast(image[..., None], 'float32')
        return (kspace_channeled, mask), image_channeled
    return _from_image_to_masked_kspace_and_mask


def from_image_to_zero_filled(af=4):
    def _from_image_to_zero_filled(image):
        image = tf.cast(image, 'complex64')
        mask = gen_mask_tf(image, accel_factor=af)
        kspace = tf_masked_shifted_normed_fft2d(image, mask)
        zero_filled = tf_masked_shifted_normed_ifft2d(kspace, mask)
        zero_filled_channeled = zero_filled[..., None]
        image_channeled = image[..., None]
        image_channeled = tf.cast(image[..., None], 'float32')
        return zero_filled_channeled, image_channeled
    return _from_image_to_zero_filled


def masked_kspace_ellipse_dataset(im_size, af=2, batch_size=8):
    im_ds = tf.data.Dataset.from_generator(
        random_ellipse_gen,
        output_types=tf.float32,
        output_shapes=tf.TensorShape([im_size, im_size]),
        args=(im_size,),
    )
    kspace_mask_ds = im_ds.map(
        from_image_to_masked_kspace_and_mask(af=af),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    kspace_mask_ds = kspace_mask_ds.batch(batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return kspace_mask_ds

def zero_filled_ellipse_dataset(im_size, af=2, batch_size=8):
    im_ds = tf.data.Dataset.from_generator(
        random_ellipse_gen,
        output_types=tf.float32,
        output_shapes=tf.TensorShape([im_size, im_size]),
        args=(im_size,),
    )
    zero_filled_ds = im_ds.map(
        from_image_to_zero_filled(af=af),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    zero_filled_ds = zero_filled_ds.batch(batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return zero_filled_ds
