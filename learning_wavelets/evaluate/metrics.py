import tensorflow as tf


def keras_psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, 1)

def keras_ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, 1)
