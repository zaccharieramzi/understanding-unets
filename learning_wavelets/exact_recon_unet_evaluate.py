import os
import numpy as np
from learning_wavelets.config import *
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Subtract, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
from tqdm.notebook import tqdm

from runstats import Statistics
from skimage.measure import compare_psnr, compare_ssim

from learning_wavelets.data.datasets import im_dataset_bsd68

from exact_recon_unet import ExactReconUnet




def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max() - gt.min())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max() - gt.min()
    )


METRIC_FUNCS = dict(
    PSNR=psnr,
    SSIM=ssim,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }
        self.metric_funcs = metric_funcs

    def push(self, target, recons):
        for metric, func in self.metric_funcs.items():
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
            f'{name} = {means[name]:.4g} +/- {stddevs[name]:.4g}' for name in metric_names
        )

def evaluate_unet(
        run_id,
        n_epochs=200,
        n_output_channels = 1,
        kernel_size = 3,
        cuda_visible_devices = '0123',
        layers_n_channels = [64, 128, 256, 512, 1024],
        layers_n_non_lins = 2,
        non_linearity = 'relu',
    ):
    
    val_path = f'{BSD68_DATA_DIR}BSD68'
    n_volumes = 68

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)

    run_params = {
            'n_output_channels': n_output_channels,
            'kernel_size': kernel_size,
            'layers_n_channels': layers_n_channels,
            'layers_n_non_lins': layers_n_non_lins,
            'non_linearity': non_linearity,
    }


   
    dataset = im_dataset_bsd68
    kwargs = {}

        
    val_set = dataset(
        val_path,
        **kwargs,
    )

    val_set = val_set.take(n_volumes)

    model = ExactReconUnet(**run_params)
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5')
    
    eval_res = Metrics(METRIC_FUNCS)
    for x, y_true in tqdm(val_set.as_numpy_iterator(), total=n_volumes):
        y_pred = model.predict(x, batch_size=4)
        eval_res.push(y_true[..., 0], y_pred[..., 0])
    return METRIC_FUNCS, (list(eval_res.means().values()), list(eval_res.stddevs().values()))
