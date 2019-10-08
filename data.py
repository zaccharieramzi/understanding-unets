import glob
import random

from keras.datasets import cifar10, mnist
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import  Sequence
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

class MergedGenerators(Sequence):
    def __init__(self, *generators):
        self.generators = generators
        # TODO add a check to verify that all generators have the same length

    def __len__(self):
        return len(self.generators[0])

    def __getitem__(self, index):
        return [generator[index] for generator in self.generators]


def im_generator(validation_split=0.1, noise=False, noise_mean=0.0, noise_std=0.1):
    if noise:
        def add_noise(image):
            noisy_img = image + np.random.normal(loc=noise_mean, scale=noise_std, size=image.shape)
            return noisy_img
        preprocessing_function = add_noise
    else:
        preprocessing_function = None
    return ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=validation_split,
        preprocessing_function=preprocessing_function,
    )

def generator_couple_from_array(x, validation_split=0.1, batch_size=32, seed=0, subset=None, noise_mean=0.0, noise_std=0.1):
    gt_image_datagen = im_generator(validation_split, noise=False)
    noisy_image_datagen = im_generator(validation_split, noise=True, noise_mean=noise_mean, noise_std=noise_std)
    gt_image_generator = gt_image_datagen.flow(
        x,
        batch_size=batch_size,
        seed=seed,
        subset=subset,
    )
    noisy_image_generator = noisy_image_datagen.flow(
        x,
        batch_size=batch_size,
        seed=seed,
        subset=subset,
    )
    return MergedGenerators(noisy_image_generator, gt_image_generator)


def generator_couple_from_dir(
        dir_path,
        validation_split=0.1,
        batch_size=32,
        seed=0,
        subset=None,
        noise_mean=0.0,
        noise_std=0.1,
        target_size=256,
        resizing_function=None,
    ):
    gt_image_datagen = im_generator(validation_split, noise=False)
    noisy_image_datagen = im_generator(validation_split, noise=True, noise_mean=noise_mean, noise_std=noise_std)
    gt_image_generator = gt_image_datagen.flow_from_directory(
        dir_path,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode=None,
        seed=seed,
        subset=subset,
        resizing_function=resizing_function,
    )
    noisy_image_generator = noisy_image_datagen.flow_from_directory(
        dir_path,
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode=None,
        seed=seed,
        subset=subset,
        resizing_function=resizing_function,
    )
    return MergedGenerators(noisy_image_generator, gt_image_generator)


def keras_im_generator(mode='training', batch_size=32, noise_mean=0.0, noise_std=0.1, validation_split=0.1, source='cifar10', seed=0):
    train_modes = ('training', 'validation')
    if source == 'cifar10':
        (x_train, _), (x_test, _) = cifar10.load_data()
    elif source == 'mnist':
        (x_train, _), (x_test, _) = mnist.load_data()
    elif source == 'cifar_grey':
        (x_train, _), (x_test, _) = cifar10.load_data()
        x_train = np.mean(x_train, axis=-1, keepdims=True)
        x_test = np.mean(x_test, axis=-1, keepdims=True)
    else:
        raise ValueError('Source unknown')
    if mode in train_modes:
        x = x_train
        subset = mode
    elif mode == 'testing':
        validation_split = 0.0
        x = x_test
        subset = None
    else:
        raise ValueError('Mode {mode} not recognised'.format(mode=mode))
    return generator_couple_from_array(
        x,
        validation_split=validation_split,
        batch_size=batch_size,
        seed=seed,
        subset=subset,
        noise_mean=noise_mean,
        noise_std=noise_std,
    )


def bds_im_to_array(fname):
    x = np.array(plt.imread(fname))
    if x.shape[1] > x.shape[0]:
        x = np.rot90(x)
    return x


def im_generator_BSD68(path, grey=False, mode='training', batch_size=32, noise_mean=0.0, noise_std=0.1, validation_split=0.1, seed=0, n_pooling=3):
    train_modes = ('training', 'validation')
    if mode in train_modes:
        subset = mode
    elif mode == 'testing':
        validation_split = 0.0
        subset = None
    else:
        raise ValueError('Mode {mode} not recognised'.format(mode=mode))
    filelist = glob.glob(path + '/*.jpg')
    x = np.array([bds_im_to_array(fname) for fname in filelist])
    if grey:
        x = np.mean(x, axis=-1, keepdims=True)
    # padding
    im_shape = np.array(x.shape[1:3])
    to_pad = ((im_shape / 2**n_pooling).astype(int) + 1) * 2**n_pooling - im_shape
    # the + 1 is necessary because the images have odd shapes
    pad_seq = [(0, 0), (to_pad[0]//2, to_pad[0]//2 + 1), (to_pad[1]//2, to_pad[1]//2 + 1), (0, 0)]
    x = np.pad(x, pad_seq, 'constant')
    return generator_couple_from_array(
        x,
        validation_split=validation_split,
        batch_size=batch_size,
        seed=seed,
        subset=subset,
        noise_mean=noise_mean,
        noise_std=noise_std,
    )


def resize_div2k_image_random_patch(div2k_imag, patch_size=256):
    subpatches_slices = list()
    for i in range(int(div2k_imag.shape[0] / patch_size)):
        slice_i = slice(i*patch_size, (i+1)*patch_size)
        for j in range(int(div2k_imag.shape[1] / patch_size)):
            slice_j = slice(j*patch_size, (j+1)*patch_size)
            patch_slices = [slice_i, slice_j]
            subpatches_slices.append(patch_slices)
    random_patch_slices = random.choice(subpatches_slices)
    random_patch = div2k_imag[random_patch_slices[0], random_patch_slices[1]]
    return random_patch


def im_generator_DIV2K(path, patch_size=256, mode='training', batch_size=32, noise_mean=0.0, noise_std=10, validation_split=0.1, seed=0):
    train_modes = ('training', 'validation')
    if mode in train_modes:
        subset = mode
    elif mode == 'testing':
        validation_split = 0.0
        subset = None
    else:
        raise ValueError('Mode {mode} not recognised'.format(mode=mode))
    def resizing_function(image):
        return resize_div2k_image_random_patch(image, patch_size=patch_size)

    return generator_couple_from_dir(
        path,
        validation_split=validation_split,
        batch_size=batch_size,
        seed=seed,
        subset=subset,
        noise_mean=noise_mean,
        noise_std=noise_std,
        target_size=patch_size,
        resizing_function=resizing_function,
    )
