import numpy as np
from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim
import tensorflow as tf

def keras_psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, 1)

def keras_ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, 1)

def psnr_single_image(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=1)


def ssim_single_image(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM).

    The images must be in HWC format
    """
    return compare_ssim(
        gt, pred, multichannel=True, data_range=1
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

METRIC_FUNCS = dict(
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self):
        self.metrics = {
            metric: Statistics() for metric in METRIC_FUNCS
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )
