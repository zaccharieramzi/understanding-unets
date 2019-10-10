import glob
import random

from keras.datasets import cifar10, mnist
from keras_preprocessing.image import ImageDataGenerator
from keras.utils import  Sequence
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)


# general keras utilities
class ConcatenateGenerators(Sequence):
    def __init__(self, *generators):
        self.generators = generators
        self.gen_lengths = [len(generator) for generator in self.generators]
        self.cum_gen_lengths = np.roll(np.cumsum(self.gen_lengths), 1)
        self.cum_gen_lengths[0] = 0

    def __len__(self):
        return np.sum(self.gen_lengths)

    def __getitem__(self, index):
        generator_index = np.where(self.cum_gen_lengths - index <= 0)[0][-1]
        index_in_generator = index - self.cum_gen_lengths[generator_index]
        return self.generators[generator_index][index_in_generator]

class MergedGenerators(Sequence):
    def __init__(self, *generators):
        self.generators = generators
        # TODO add a check to verify that all generators have the same length

    def __len__(self):
        return len(self.generators[0])

    def __getitem__(self, index):
        return [generator[index] for generator in self.generators]

def im_generator(validation_split=0.1, noise=False, noise_mean=0.0, noise_std=0.1, no_augment=False):
    if noise:
        def add_noise(image):
            noisy_img = image + np.random.normal(loc=noise_mean, scale=noise_std, size=image.shape)
            return noisy_img
        preprocessing_function = add_noise
    else:
        preprocessing_function = None
    if no_augment:
        augment_kwargs = {}
    else:
        augment_kwargs = {
            'rotation_range': 20,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'horizontal_flip': True,
            'vertical_flip': True,
        }
    return ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        fill_mode='constant',
        validation_split=validation_split,
        preprocessing_function=preprocessing_function,
        **augment_kwargs,
    )

def generator_couple_from_array(x, validation_split=0.1, batch_size=32, seed=0, subset=None, noise_mean=0.0, noise_std=0.1, no_augment=False):
    gt_image_datagen = im_generator(validation_split, noise=False, no_augment=no_augment)
    noisy_image_datagen = im_generator(validation_split, noise=True, noise_mean=noise_mean, noise_std=noise_std, no_augment=no_augment)
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
        no_augment=False,
    ):
    gt_image_datagen = im_generator(validation_split, noise=False, no_augment=no_augment)
    noisy_image_datagen = im_generator(validation_split, noise=True, noise_mean=noise_mean, noise_std=noise_std, no_augment=no_augment)
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

# generator for keras datasets (cifar, mnist)
def keras_im_generator(mode='training', batch_size=32, noise_mean=0.0, noise_std=0.1, validation_split=0.1, source='cifar10', seed=0, no_augment=False):
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
        no_augment=no_augment
    )

# BSD68 utilities
def bsd68_im_to_array(fname):
    x = np.array(plt.imread(fname))
    if x.shape[1] > x.shape[0]:
        x = np.rot90(x)
    return x

def im_generator_BSD68(path, grey=False, mode='training', batch_size=32, noise_mean=0.0, noise_std=0.1, validation_split=0.1, seed=0, n_pooling=3, no_augment=False):
    train_modes = ('training', 'validation')
    if mode in train_modes:
        subset = mode
    elif mode == 'testing':
        validation_split = 0.0
        subset = None
    else:
        raise ValueError('Mode {mode} not recognised'.format(mode=mode))
    filelist = glob.glob(path + '/*.jpg')
    x = np.array([bsd68_im_to_array(fname) for fname in filelist])
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
        no_augment=no_augment,
    )

# div2k utilities
def resize_div2k_image_random_patch(div2k_imag, patch_size=256, seed=None, grey=False):
    # NOTE: not the best solution because it will always select the same patch
    # but it will require some very ad hoc stuff in iterator to increase the
    # seed like in l59 of iterator
    random.seed(seed + int(100 * div2k_imag[0, 0, 0]))
    subpatches_slices = list()
    for i in range(int(div2k_imag.shape[0] / patch_size)):
        slice_i = slice(i*patch_size, (i+1)*patch_size)
        for j in range(int(div2k_imag.shape[1] / patch_size)):
            slice_j = slice(j*patch_size, (j+1)*patch_size)
            patch_slices = [slice_i, slice_j]
            subpatches_slices.append(patch_slices)
    random_patch_slices = random.choice(subpatches_slices)
    random_patch = div2k_imag[random_patch_slices[0], random_patch_slices[1]]
    if grey:
        random_patch = np.mean(random_patch, axis=-1, keepdims=True)
    return random_patch

def im_generator_DIV2K(path, patch_size=256, mode='training', batch_size=32, noise_mean=0.0, noise_std=10, validation_split=0.1, seed=0, no_augment=False, grey=False):
    train_modes = ('training', 'validation')
    if mode in train_modes:
        subset = mode
    elif mode == 'testing':
        validation_split = 0.0
        subset = None
    else:
        raise ValueError('Mode {mode} not recognised'.format(mode=mode))
    def resizing_function(image):
        return resize_div2k_image_random_patch(image, patch_size=patch_size, seed=seed, grey=grey)

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
        no_augment=no_augment,
    )

# common API
def im_generators(source, batch_size=32, validation_split=0.1, no_augment=False, noise_std=30, n_pooling=4, grey=False):
    # TODO: add number of pooling, and possibility to use grey for images
    if 'cifar' in source:
        n_samples_train = 5*1e4
        size = 32
    elif 'mnist' in source:
        n_samples_train = 6*1e4
        size = 28
    elif source == 'bsd68':
        bsd_dir_train = 'BSDS300/images/train'
        bsd_dir_test = 'BSDS300/images/test'
        n_samples_train = 200
        size = None
    elif source == 'div2k':
        div_2k_dir_train = 'DIV2K_train_HR/'
        div_2k_dir_val = 'DIV2K_valid_HR/'
        n_samples_train = 800
        size = 256
    if source in ['cifar', 'cifar_grey', 'mnist']:
        im_gen_train = keras_im_generator(
            mode='training',
            validation_split=validation_split,
            batch_size=batch_size,
            source=source,
            noise_std=noise_std,
            no_augment=no_augment,
        )
        im_gen_val = keras_im_generator(
            mode='validation',
            validation_split=validation_split,
            batch_size=batch_size,
            source=source,
            noise_std=noise_std,
            no_augment=no_augment,
        )
        im_gen_test = keras_im_generator(
            mode='testing',
            validation_split=validation_split,
            batch_size=batch_size,
            source=source,
            noise_std=noise_std,
            no_augment=no_augment,
        )
    elif source == 'bsd68':
        im_gen_train = im_generator_BSD68(
            path=bsd_dir_train,
            mode='training',
            validation_split=validation_split,
            batch_size=batch_size,
            noise_std=noise_std,
            no_augment=no_augment,
            n_pooling=n_pooling,
            grey=grey,
        )
        im_gen_val = im_generator_BSD68(
            path=bsd_dir_train,
            mode='validation',
            validation_split=validation_split,
            batch_size=batch_size,
            noise_std=noise_std,
            no_augment=no_augment,
            n_pooling=n_pooling,
            grey=grey,
        )
        im_gen_test = im_generator_BSD68(
            path=bsd_dir_test,
            mode='testing',
            validation_split=0,
            batch_size=batch_size,
            noise_std=noise_std,
            no_augment=no_augment,
            n_pooling=n_pooling,
            grey=grey,
        )
    elif source == 'div2k':
        im_gen_train = im_generator_DIV2K(
            path=div_2k_dir_train,
            patch_size=size,
            mode='training',
            validation_split=validation_split,
            batch_size=batch_size,
            noise_std=noise_std,
            no_augment=no_augment,
            grey=grey,
        )
        im_gen_val = im_generator_DIV2K(
            path=div_2k_dir_train,
            patch_size=size,
            mode='validation',
            validation_split=validation_split,
            batch_size=batch_size,
            noise_std=noise_std,
            no_augment=no_augment,
            grey=grey,
        )
        im_gen_test = im_generator_DIV2K(
            path=div_2k_dir_val,
            patch_size=size,
            mode='testing',
            validation_split=validation_split,
            batch_size=batch_size,
            noise_std=noise_std,
            no_augment=no_augment,
            grey=grey,
        )
    return im_gen_train, im_gen_val, im_gen_test, size, n_samples_train
