import bm3d
import numpy as np
import tensorflow.keras.backend as K
from tqdm import tqdm_notebook

from ..evaluate import Metrics
from ..keras_utils.load_model import unpack_model
from ..wavelet_denoising import wavelet_denoising_pysap


# TODO: get rid of sequence metrics and adapt notebooks
def enumerate_seq(seq, name):
    return (seq[i] for i in tqdm_notebook(range(len(seq)), desc=f'Val files for {name}'))

def enumerate_seq_noisy(seq, name):
    return (np.squeeze(seq[i][0]) for i in tqdm_notebook(range(len(seq)), desc=f'Val files for {name}'))

def enumerate_seq_gt(seq):
    return (np.squeeze(seq[i][1]) for i in range(len(seq)))

def metrics_for_params(val_seq, name=None, **net_params):
    model = unpack_model(**net_params)
    metrics = Metrics()
    pred_and_gt = [
        (model.predict_on_batch(images_noisy), images_gt)
        for images_noisy, images_gt in enumerate_seq(val_seq, name)
    ]
    for im_recos, images in tqdm_notebook(pred_and_gt, desc=f'Stats for {name}'):
        metrics.push(images, im_recos.numpy())
    return metrics

def metrics_exact_recon_net(val_seq, name=None, **net_params):
    model = unpack_model(**net_params)
    metrics = Metrics()
    pred_and_gt = [
        (model.predict_on_batch((images_noisy, images_noisy))[0], images_gt)
        for images_noisy, images_gt in enumerate_seq(val_seq, name)
    ]
    for im_recos, images in tqdm_notebook(pred_and_gt, desc=f'Stats for {name}'):
        metrics.push(images, im_recos.numpy())
    return metrics

def metrics_dynamic_denoising_net(val_seq, noise_std, name=None, **net_params):
    model = unpack_model(**net_params)
    metrics = Metrics()
    pred_and_gt = [
        (model.predict_on_batch((images_noisy, np.array([[noise_std/255]]))), images_gt)
        for images_noisy, images_gt in enumerate_seq(val_seq, name)
    ]
    for im_recos, images in tqdm_notebook(pred_and_gt, desc=f'Stats for {name}'):
        metrics.push(images, im_recos.numpy())
    return metrics

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

def metrics_original(val_seq):
    metrics = Metrics()
    pred_and_gt = [
        (images_noisy, images_gt)
        for images_noisy, images_gt in enumerate_seq(val_seq, 'Original noisy image')
    ]
    for im_recos, images in tqdm_notebook(pred_and_gt, desc='Original noisy image'):
        metrics.push(images, im_recos)
    return metrics

def metrics_wavelets(val_seq, wavelet_id, noise_std=30):
    metrics = Metrics()
    pred = wavelet_denoising_pysap(
        enumerate_seq_noisy(val_seq, f'Wavelet denoising {wavelet_id}'),
        noise_std=noise_std/255,
        wavelet_id=wavelet_id,
        n_scales=5,
        soft_thresh=False,
        n_sigma=3,
    )
    gt = enumerate_seq_gt(val_seq)
    for im_recos, images in tqdm_notebook(zip(pred, gt), desc='Stats for wavelet denoising'):
        metrics.push(images[..., None], im_recos[..., None])
    return metrics

def metrics_bm3d(val_seq, noise_std=30):
    metrics = Metrics()
    pred = [
        bm3d.bm3d(image_noisy + 0.5, sigma_psd=noise_std/255, stage_arg=bm3d.BM3DStages.ALL_STAGES) - 0.5
        for image_noisy in enumerate_seq_noisy(val_seq, f'BM3D')
    ]
    gt = enumerate_seq_gt(val_seq)
    for im_recos, images in tqdm_notebook(zip(pred, gt), desc='Stats for bm3d'):
        metrics.push(images[..., None], im_recos[..., None])
    return metrics

def n_params_from_params(name=None, **net_params):
    model = unpack_model(**net_params)
    return np.sum([K.count_params(w) for w in model.trainable_weights])
