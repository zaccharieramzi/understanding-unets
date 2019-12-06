from collections.abc import Iterable

import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf


# normalisation
def normalise(image):
    return (image / 255) - 0.5

# data augmentation
def random_rotate_image(image):
    # in numpy
    image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
    return image

def tf_random_rotate_image(image):
    # in tf
    im_shape = image.shape
    [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
    image.set_shape(im_shape)
    return image

# patch selection
def select_patch_in_image_function(patch_size, seed=0):
    def select_patch_in_image(image):
        patch = tf.image.random_crop(
            image,
            [patch_size, patch_size, 1],
            seed=seed,
        )
        return patch
    return select_patch_in_image

# noise
def add_noise_function(noise_std_range, return_noise_level=False):
    if not isinstance(noise_std_range, Iterable):
        noise_std_range = (noise_std_range, noise_std_range)
    @tf.py_function
    def add_noise(image):
        noise_std = tf.random.uniform(
            (1,),
            minval=noise_std_range[0],
            maxval=noise_std_range[1],
        )
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_std/255, dtype=tf.float32)
        if return_noise_level:
            return (image + noise), noise_std
        else:
            return image + noise
    return add_noise

def exact_recon_helper(image_noisy, image):
    return (image_noisy, image), (image, image)

def im_dataset_div2k(mode='training', batch_size=1, patch_size=256, noise_std=30, exact_recon=False, return_noise_level=False):
    if mode == 'training':
        path = 'DIV2K_train_HR'
    elif mode == 'validation':
        path = 'DIV2K_valid_HR'
    file_ds = tf.data.Dataset.list_files(f'{path}/*/*.png', seed=0)
    image_ds = file_ds.map(
        tf.io.read_file, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).map(
        tf.image.decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    image_grey_ds = image_ds.map(
        tf.image.rgb_to_grayscale, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).map(
        normalise, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # image_grey_aug_ds = image_grey_ds.map(tf_random_rotate_image)
    select_patch_in_image = select_patch_in_image_function(patch_size)
    image_patch_ds = image_grey_ds.map(
        select_patch_in_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    add_noise = add_noise_function(noise_std, return_noise_level=return_noise_level)
    image_noisy_ds = image_patch_ds.map(
        lambda patch: (add_noise(patch), patch),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if exact_recon:
        # TODO: see how to adapt exact recon for the case of noise level included
        image_noisy_ds = image_noisy_ds.map(
            exact_recon_helper,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    image_noisy_ds = image_noisy_ds.batch(batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return image_noisy_ds
