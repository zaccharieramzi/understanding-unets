from collections.abc import Iterable

import numpy as np
import scipy.ndimage as ndimage
import tensorflow as tf

from ..config import *

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
        if patch_size is not None:
            patch = tf.image.random_crop(
                image,
                [patch_size, patch_size, 1],
                seed=seed,
            )
            return patch
        else:
            return image
    return select_patch_in_image

# padding
def pad_for_pool(n_pooling=1, return_original_shape=False):
    def pad(image):
        im_shape = tf.shape(image)[:2]
        to_pad = (tf.dtypes.cast(im_shape / 2**n_pooling, 'int32') + 1) * 2**n_pooling - im_shape
        # the + 1 is necessary because the images have odd shapes
        pad_seq = [(to_pad[0]//2, to_pad[0]//2 + 1), (to_pad[1]//2, to_pad[1]//2 + 1), (0, 0)]
        image_padded = tf.pad(image, pad_seq, 'SYMMETRIC')
        if return_original_shape:
            return image_padded, im_shape
        else:
            return image_padded
    return pad

# noise
def add_noise_function(noise_std_range, return_noise_level=False, no_noise=False, set_noise_zero=False, decreasing_noise_level=False):
    if not isinstance(noise_std_range, Iterable):
        noise_std_range = (noise_std_range, noise_std_range)
    def add_noise(image):
        noise_std = tf.random.uniform(
            (1,),
            minval=noise_std_range[0],
            maxval=noise_std_range[1],
        )
        if decreasing_noise_level:
            noise_std = tf.minimum(noise_std, tf.random.uniform(
                (1,),
                minval=noise_std_range[0],
                maxval=noise_std_range[1],
            ))
        if no_noise:
            if return_noise_level:
                return image, noise_std/255
            else:
                return image
        else:
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_std/255, dtype=tf.float32)
            if return_noise_level:
                if set_noise_zero:
                    return (image + noise), tf.zeros_like(noise_std)
                else:
                    return (image + noise), noise_std/255
            else:
                return image + noise
    return add_noise

def exact_recon_helper(image_noisy, image):
    return (image_noisy, image), (image, image)

# TODO: refactor the datasets
def im_dataset_div2k(mode='training', batch_size=1, patch_size=256, noise_std=30, exact_recon=False, return_noise_level=False):
    if mode == 'training':
        path = 'DIV2K_train_HR'
    elif mode == 'validation' or mode == 'testing':
        path = 'DIV2K_valid_HR'
    file_ds = tf.data.Dataset.list_files(f'{DIV2K_DATA_DIR}{path}/*/*.png', seed=0)
    file_ds = file_ds.shuffle(800, seed=0)
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
    image_noisy_ds = image_noisy_ds.batch(batch_size)
    if mode != 'testing':
        image_noisy_ds = image_noisy_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return image_noisy_ds

def im_dataset_bsd500(mode='training', **kwargs):
    # the training set for bsd500 is test + train
    # the test set (i.e. containing bsd68 images) is val
    if mode == 'training':
        train_path = 'BSR/BSDS500/data/images/train'
        test_path = 'BSR/BSDS500/data/images/test'
        paths = [train_path, test_path]
    elif mode == 'validation' or mode == 'testing':
        val_path = 'BSR/BSDS500/data/images/val'
        paths = [val_path]
    im_ds = im_dataset_bsd(BSD500_DATA_DIR, paths, 'jpg', from_rgb=True, mode=mode, **kwargs)
    return im_ds

def im_dataset_bsd68(mode='validation', **kwargs):
    path = 'BSD68'
    im_ds = im_dataset_bsd(BSD68_DATA_DIR, [path], 'png', mode=mode, **kwargs)
    return im_ds

def im_dataset_bsd(
        data_dir,
        paths,
        pattern,
        mode='validation',
        batch_size=1,
        patch_size=256,
        noise_std=30,
        no_noise=False,
        return_noise_level=False,
        n_pooling=None,
        n_samples=None,
        set_noise_zero=False,
        from_rgb=False,
        decreasing_noise_level=False,
    ):
    file_ds = None
    for path in paths:
        file_ds_new = tf.data.Dataset.list_files(f'{data_dir}{path}/*.{pattern}', seed=0)
        if file_ds is None:
            file_ds = file_ds_new
        else:
            file_ds.concatenate(file_ds_new)
    if n_samples is not None:
        file_ds = file_ds.take(n_samples)
    file_ds = file_ds.shuffle(800, seed=0)
    if pattern == 'jpg':
        decode_function = tf.image.decode_jpeg
    elif pattern == 'png':
        decode_function = tf.image.decode_png
    image_ds = file_ds.map(
        tf.io.read_file, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).map(
        decode_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if from_rgb:
        image_ds = image_ds.map(
            tf.image.rgb_to_grayscale, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    image_grey_ds = image_ds.map(
        normalise, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if patch_size is not None:
        select_patch_in_image = select_patch_in_image_function(patch_size)
        image_patch_ds = image_grey_ds.map(
            select_patch_in_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    elif n_pooling is not None:
        pad = pad_for_pool(n_pooling, return_original_shape=mode=='testing')
        image_patch_ds = image_grey_ds.map(
            pad, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    else:
        image_patch_ds = image_grey_ds
    add_noise = add_noise_function(
        noise_std,
        return_noise_level=return_noise_level,
        no_noise=no_noise,
        set_noise_zero=set_noise_zero,
        decreasing_noise_level=decreasing_noise_level,
    )
    if mode == 'validation' or mode == 'training':
        image_noisy_ds = image_patch_ds.map(
            lambda patch: (add_noise(patch), patch),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    if mode == 'testing':
        if n_pooling is not None:
            image_noisy_ds = image_patch_ds.map(
                lambda patch, im_shape: (add_noise(patch), patch, im_shape),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        else:
            image_noisy_ds = image_patch_ds.map(
                lambda patch: (add_noise(patch), patch, tf.shape(patch)[:2]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
    image_noisy_ds = image_noisy_ds.batch(batch_size)
    if mode != 'testing':
        image_noisy_ds = image_noisy_ds.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return image_noisy_ds
