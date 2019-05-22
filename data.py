from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator
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
