import tensorflow as tf
from tqdm import tqdm

from learning_wavelets.data.datasets import im_dataset_bsd68
from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.evaluate import Metrics
from learning_wavelets.models.learnlet_model import Learnlet


def evaluate_learnlet(
        run_id,
        noise_stds,
        n_epochs=500,
        n_samples=None,
        denoising_activation='dynamic_soft_thresholding',
        n_filters=256,
        exact_reconstruction=False,
    ):
    run_params = {
        'denoising_activation': denoising_activation,
        'learnlet_analysis_kwargs':{
            'n_tiling': n_filters,
            'mixing_details': False,
            'skip_connection': True,
            'kernel_size': 11,
        },
        'learnlet_synthesis_kwargs': {
            'res': True,
            'kernel_size': 13,
        },
        'wav_type': 'starlet',
        'n_scales': 5,
        'clip': False,
        'random_analysis': True,
        'exact_reconstruction': exact_reconstruction,
    }
    model = Learnlet(**run_params)
    model([
        tf.ones([1, 32, 32, 1]),
        tf.ones([1, 1]),
    ])
    chkpt_path = f'{CHECKPOINTS_DIR}/checkpoints/{run_id}-{n_epochs}.hdf5'
    model.load_weights(chkpt_path)
    noise_std_metrics = {}
    for noise_std in tqdm(noise_stds, 'Noise stds'):
        im_ds = im_dataset_bsd68(
            mode='testing',
            batch_size=1,
            patch_size=None,
            noise_std=noise_std,
            return_noise_level=True,
            n_pooling=5,
            n_samples=n_samples,
        )
        metrics = Metrics()
        pred_and_gt_shape = [
            (model.predict_on_batch(images_noisy), images_gt, im_shape)
            for images_noisy, images_gt, im_shape in tqdm(im_ds)
        ]
        for im_recos, images, im_shape in pred_and_gt_shape:
            metrics.push(images.numpy(), im_recos, im_shape.numpy())
        noise_std_metrics[noise_std] = metrics
    print(noise_std_metrics)
    return ['PSNR' 'SSIM'], noise_std_metrics
