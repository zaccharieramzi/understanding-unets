import bm3d
import numpy as np
import tensorflow.keras.backend as K
from tqdm import tqdm_notebook

from ..evaluate import Metrics
from ..keras_utils.load_model import unpack_model
from ..wavelet_denoising import wavelet_denoising_pysap


def metrics_from_ds(ds, with_shape=True, name=None, **net_params):
    model = unpack_model(**net_params)
    metrics = Metrics()
    pred_and_gt_shape = [
        (model.predict_on_batch(images_noisy), images_gt, im_shape)
        for images_noisy, images_gt, im_shape in tqdm_notebook(ds)
    ]
    for im_recos, images, im_shape in tqdm_notebook(pred_and_gt_shape, desc=f'Stats for {name}'):
        metrics.push(images.numpy(), im_recos.numpy(), im_shape.numpy())
    return metrics

def metrics_original_from_ds(ds, with_shape=True):
    metrics = Metrics()
    pred_and_gt_shape = [
        (images_noisy.numpy(), images_gt.numpy(), im_shape.numpy())
        for images_noisy, images_gt, im_shape in tqdm_notebook(ds)
    ]
    for im_recos, images, im_shape in tqdm_notebook(pred_and_gt_shape, desc='Stats for original noisy images'):
        metrics.push(images, im_recos, im_shape)
    return metrics

def metrics_wavelets_from_ds(ds, wavelet_id, noise_std=30, with_shape=True):
    metrics = Metrics()
    pred_and_gt_shape = [
        (
            np.array(wavelet_denoising_pysap(
                images_noisy[..., 0].numpy(),
                noise_std=noise_std/255,
                wavelet_id=wavelet_id,
                n_scales=5,
                soft_thresh=False,
                n_sigma=3,
            ))[..., None],
            images_gt.numpy(),
            im_shape.numpy(),
        )
        for images_noisy, images_gt, im_shape in tqdm_notebook(ds)
    ]
    for im_recos, images, im_shape in tqdm_notebook(pred_and_gt_shape, desc=f'Stats for wavelets {wavelet_id}'):
        metrics.push(images, im_recos, im_shape)
    return metrics

def metrics_bm3d_from_ds(ds, noise_std=30, with_shape=True):
    metrics = Metrics()
    pred_and_gt_shape = [
        (
            (bm3d.bm3d(
                images_noisy[0, ..., 0].numpy() + 0.5,
                sigma_psd=noise_std/255,
                stage_arg=bm3d.BM3DStages.ALL_STAGES
            ) - 0.5)[None, ..., None],
            images_gt.numpy(),
            im_shape.numpy(),
        )
        for images_noisy, images_gt, im_shape in tqdm_notebook(ds)
    ]
    for im_recos, images, im_shape in tqdm_notebook(pred_and_gt_shape, desc=f'Stats for BM3D'):
        metrics.push(images, im_recos, im_shape)
    return metrics


def n_params_from_params(name=None, **net_params):
    model = unpack_model(**net_params)
    return np.sum([K.count_params(w) for w in model.trainable_weights])
