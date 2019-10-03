import glob

from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)


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

def generator_couple(x, validation_split=0.1, batch_size=32, seed=0, subset=None, noise_mean=0.0, noise_std=0.1):
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
    return zip(gt_image_generator, noisy_image_generator)


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
    return generator_couple(
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


def im_generator_BSD68(path, grey=False, mode='training', batch_size=32, noise_mean=0.0, noise_std=0.1, validation_split=0.1, seed=0):
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
    x = np.pad(x, ((0, 0), (15, 16), (95, 96), (0, 0)), 'edge')
    return generator_couple(
        x,
        validation_split=validation_split,
        batch_size=batch_size,
        seed=seed,
        subset=subset,
        noise_mean=noise_mean,
        noise_std=noise_std,
    )


def div2k_im_to_patches(fname, patch_size=256):
    # with patch size of 256 (no padding):
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


def im_generator_DIV2K(path, grey=False, patch_size=256, mode='training', batch_size=32, noise_mean=0.0, noise_std=10):
    # TODO: have that in a sequence that will get the file rather than loading everything in memory
    # this will allow patch handling and all the bla bla
    train_modes = ('training', 'validation')
    if mode in train_modes:
        pass
    elif mode == 'testing':
        raise ValueError('Mode {mode} not used in DIV2K'.format(mode=mode))
    else:
        raise ValueError('Mode {mode} not recognised'.format(mode=mode))
    filelist = glob.glob(path + '/*')
    while True:
        x = (patch for fname in filelist for patch in div2k_im_to_patches(fname, patch_size=patch_size))
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
