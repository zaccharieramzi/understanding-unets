import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import tensorflow as tf

def keras_psnr(y_true, y_pred):
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.psnr(y_true, y_pred, max_pixel - min_pixel)

def keras_ssim(y_true, y_pred):
    max_pixel = tf.math.reduce_max(y_true)
    min_pixel = tf.math.reduce_min(y_true)
    return tf.image.ssim(y_true, y_pred, max_pixel - min_pixel)

def psnr_single_image(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max() - gt.min())


def ssim_single_image(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM).

    The images must be in HWC format
    """
    return compare_ssim(
        gt, pred, multichannel=True, data_range=gt.max() - gt.min()
    )

def psnr(gts, preds):
    """Compute the psnr of a batch of images in HWC format.

    Images must be in NHWC format
    """
    if len(gts.shape) == 3:
        return psnr_single_image(gts, preds)
    else:
        mean_psnr = np.mean([psnr_single_image(gt, pred) for gt, pred in zip(gts, preds)])
        return mean_psnr

def ssim(gts, preds):
    """Compute the ssim of a batch of images in HWC format.

    Images must be in NHWC format
    """
    if len(gts.shape) == 3:
        return ssim_single_image(gts, preds)
    else:
        mean_ssim = np.mean([ssim_single_image(gt, pred) for gt, pred in zip(gts, preds)])
        return mean_ssim
