import bm3d
import numpy as np
from tqdm import tqdm

from .multiscale_eval import DEFAULT_NOISE_STDS
from .results_to_csv import results_to_csv
from .runstats import Metrics, METRIC_FUNCS
from learning_wavelets.data.datasets import im_dataset_bsd68
from learning_wavelets.wavelet_denoising import wavelet_denoising_pysap


def metrics_baseline(denoising_function, baseline_name='original', n_pooling=5, **kwargs):
    metrics_list = []
    for noise_std in DEFAULT_NOISE_STDS:
        metrics = Metrics()
        im_ds = im_dataset_bsd68(
            mode='testing',
            batch_size=1,
            patch_size=None,
            noise_std=noise_std,
            return_noise_level=False,
            n_samples=None,
            n_pooling=n_pooling,
        )
        for images_noisy, images_gt, im_shape in tqdm(im_ds, desc='Stats for original noisy images'):
            images_denoised = denoising_function(images_noisy, noise_std, **kwargs)
            metrics.push(images_gt.numpy(), images_denoised, im_shape.numpy())
        metrics_list.append(metrics)
    metrics_names = METRIC_FUNCS.keys()
    formatted_metrics_list = [
        m.means().values()
        for m in metrics_list
    ]
    results_to_csv(
        list(metrics_names),
        [(kwargs, [list(m) for m in formatted_metrics_list])],
        f'{baseline_name}_metrics.csv',
    )
    return metrics_list

def metrics_original():
    def identity(images_noisy, noise_std):
        return images_noisy.numpy()
    return metrics_baseline(identity)


def metrics_wavelets_from_ds(wavelet_id='24'):
    soft_thresh = False
    def wavelet_denoise(
            images_noisy,
            noise_std,
            wavelet_id=wavelet_id,
            soft_thresh=False
        ):
        return np.array(wavelet_denoising_pysap(
            images_noisy[..., 0].numpy(),
            noise_std=noise_std/255,
            wavelet_id=wavelet_id,
            n_scales=5,
            soft_thresh=soft_thresh,
            n_sigma=3,
        ))[..., None]
    return metrics_baseline(
        wavelet_denoise,
        baseline_name='wavelets',
        soft_thresh=soft_thresh,
        wavelet_id=wavelet_id,
    )

def metrics_bm3d():
    def bm3d_denoise(images_noisy, noise_std):
        return bm3d.bm3d(
            images_noisy[0, ..., 0].numpy() + 0.5,
            sigma_psd=noise_std/255,
            stage_arg=bm3d.BM3DStages.ALL_STAGES
        )[None, ..., None] - 0.5
    return metrics_baseline(bm3d_denoise, baseline_name='bm3d')
