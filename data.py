import glob

from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)


def im_generator(mode='training', batch_size=32, noise_mean=0.0, noise_std=10, validation_split=0.1, source='cifar10'):
    train_modes = ('training', 'validation')
    if source == 'cifar10':
        (x_train, _), (x_test, _) = cifar10.load_data()
    elif source == 'mnist':
        (x_train, _), (x_test, _) = mnist.load_data()
    elif source == 'cifar_grey':
        (x_train, _), (x_test, _) = cifar10.load_data()
        x_train = np.mean(x_train, axis=-1)
        x_test = np.mean(x_test, axis=-1)
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
    image_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=validation_split,
    )
    if len(x.shape) == 3:
        x = x[:, :, :, None]
    image_generator = image_datagen.flow(
        x,
        batch_size=batch_size,
        save_to_dir=None,
        save_prefix="gen",
        seed=1,
        subset=subset
    )
    for img in image_generator:
        noisy_img = img + np.random.normal(loc=noise_mean, scale=noise_std, size=img.shape)
        img[img == 0] = 0
        img /= 255
        noisy_img /= 255
        yield (noisy_img, img)


def bds_im_to_array(fname):
    x = np.array(plt.imread(fname))
    if x.shape[1] > x.shape[0]:
        x = np.rot90(x)
    return x


def im_generator_BSD68(path, grey=False, mode='training', batch_size=32, noise_mean=0.0, noise_std=10, validation_split=0.1):
    train_modes = ('training', 'validation')
    if mode in train_modes:
        subset = mode
    elif mode == 'testing':
        validation_split = 0.0
        subset = None
    else:
        raise ValueError('Mode {mode} not recognised'.format(mode=mode))
    filelist = glob.glob(path + '/*')
    x = np.array([bds_im_to_array(fname) for fname in filelist])
    if grey:
        x = np.mean(x, axis=-1)
        x = x[:, :, :, None]
    # padding
    x = np.pad(x, ((0, 0), (15, 16), (95, 96), (0, 0)), 'edge')
    image_datagen = ImageDataGenerator(
        # rotation_range=20,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # horizontal_flip=True,
        # vertical_flip=True,
        validation_split=validation_split,
    )
    image_generator = image_datagen.flow(
        x,
        batch_size=batch_size,
        save_to_dir=None,
        save_prefix="gen",
        seed=1,
        subset=subset,
    )
    for img in image_generator:
        noisy_img = img + np.random.normal(loc=noise_mean, scale=noise_std, size=img.shape)
        img[img == 0] = 0
        img /= 255
        noisy_img /= 255
        yield (noisy_img, img)


def div2k_im_to_patches(fname, patch_size=256):
    # with patch size of 1 (no padding):
    # train has 27958 patches
    # valid has 3598 patches
    x = np.array(plt.imread(fname))
    subpatches = list()
    for i in range(int(x.shape[0] / patch_size)):
        for j in range(int(x.shape[1] / patch_size)):
            slice_i = slice(i*patch_size, (i+1)*patch_size)
            slice_j = slice(j*patch_size, (j+1)*patch_size)
            patch = x[slice_i, slice_j]
            subpatches.append(patch)
    return subpatches


def im_generator_DIV2K(path, grey=False, mode='training', batch_size=32, noise_mean=0.0, noise_std=10):
    train_modes = ('training', 'validation')
    if mode in train_modes:
        pass
    elif mode == 'testing':
        raise ValueError('Mode {mode} not used in DIV2K'.format(mode=mode))
    else:
        raise ValueError('Mode {mode} not recognised'.format(mode=mode))
    filelist = glob.glob(path + '/*')
    while True:
        x = (patch for fname in filelist for patch in div2k_im_to_patches(fname))
        image_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
        )
        current_batch_noisy = []
        current_batch = []
        for img_idx, img in enumerate(x):
            if grey:
                img = np.mean(img, axis=-1)
                img = img[:, :, None]
            img = image_datagen.random_transform(img)
            noisy_img = img + np.random.normal(loc=noise_mean, scale=noise_std, size=img.shape)
            img[img == 0] = 0
            img /= 255
            noisy_img /= 255
            current_batch_noisy.append(noisy_img)
            current_batch.append(img)
            if (img_idx + 1) % batch_size == 0:
                noisy_img_batch = np.array(current_batch_noisy)
                img_batch = np.array(current_batch)
                current_batch = []
                current_batch_noisy = []
                yield (noisy_img_batch, img_batch)
