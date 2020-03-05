import tensorflow as tf
from tensorflow.python.ops.signal.fft_ops import fft2d, ifft2d, ifftshift, fftshift


def tf_shifted_normed_fft2d(image):
    im_area = tf.dtypes.cast(tf.math.reduce_prod(tf.shape(image)), 'float32')
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(im_area), image.dtype)
    return ifftshift(fft2d(fftshift(image))) / scaling_norm

def tf_masked_shifted_normed_fft2d(image, mask):
    x_fourier = tf_shifted_normed_fft2d(image)
    x_masked = x_fourier * mask
    return x_masked

def tf_shifted_normed_ifft2d(x):
    im_area = tf.dtypes.cast(tf.math.reduce_prod(tf.shape(x)[1:3]), 'float32')
    scaling_norm = tf.dtypes.cast(tf.math.sqrt(im_area), x.dtype)
    return scaling_norm * fftshift(ifft2d(ifftshift(x)))

def tf_masked_shifted_normed_ifft2d(x, mask):
    x_masked = x * mask
    image = tf_shifted_normed_ifft2d(x_masked)
    return image
