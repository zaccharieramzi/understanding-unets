import os
import os.path as op
import time

import click
from fastmri_recon.config import FASTMRI_DATA_DIR
from fastmri_recon.data.fastmri_tf_datasets import train_masked_kspace_dataset_from_indexable
from fastmri_recon.helpers.nn_mri import _tf_crop
from fastmri_recon.helpers.utils import keras_psnr, keras_ssim
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.keras_utils.fourier import tf_masked_shifted_normed_fft2d, tf_masked_shifted_normed_ifft2d
from learning_wavelets.models.ista import IstaLearnlet

tf.random.set_seed(1)

# paths
train_path = f'{FASTMRI_DATA_DIR}singlecoil_train/singlecoil_train/'
val_path = f'{FASTMRI_DATA_DIR}singlecoil_val/'

@click.command()
@click.option(
    'af',
    '-af',
    default=4,
    type=float,
    help='The acceleration factor for the subsampling. Defaults to 4.',
)
@click.option(
    'cuda_visible_devices',
    '-gpus',
    '--cuda-visible-devices',
    default='0123',
    type=str,
    help='The visible GPU devices. Defaults to 0123',
)
@click.option(
    'denoising_activation',
    '-da',
    '--denoising-activation',
    default='dynamic_soft_thresholding',
    type=click.Choice([
        'dynamic_soft_thresholding',
        'dynamic_hard_thresholding',
        'dynamic_soft_thresholding_per_filter',
        'cheeky_dynamic_hard_thresholding'
    ], case_sensitive=False),
    help='The denoising activation to use. Defaults to dynamic_soft_thresholding',
)
@click.option(
    'n_filters',
    '-nf',
    '--n-filters',
    default=16,
    type=int,
    help='The number of filters in the learnlets. Defaults to 16.',
)
@click.option(
    'n_iters',
    '-ni',
    '--n-iters',
    default=5,
    type=int,
    help='The number of iterations in the unrolled ISTA. Defaults to 5.',
)
@click.option(
    'undecimated',
    '-u',
    is_flag=True,
    help='Set if you want the learnlets to be undecimated.',
)
@click.option(
    'exact_reco',
    '-e',
    is_flag=True,
    help='Set if you want the learnlets to have exact reconstruction.',
)
def train_ista_learnlet(
        af,
        cuda_visible_devices,
        denoising_activation,
        n_filters,
        n_iters,
        undecimated,
        exact_reco,
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    # data preparation
    train_set = train_masked_kspace_dataset_from_indexable(
        train_path,
        AF=af,
        contrast=None,
        inner_slices=8,
        rand=True,
        scale_factor=1e6,
        n_samples=None,
    )
    val_set = train_masked_kspace_dataset_from_indexable(
        val_path,
        AF=af,
        contrast=None,
        scale_factor=1e6,
    )

    learnlet_params = {
        'denoising_activation': denoising_activation,
        'learnlet_analysis_kwargs':{
            'n_tiling': n_filters,
            'mixing_details': False,
            'skip_connection': True,
            'kernel_size': 7,
        },
        'learnlet_synthesis_kwargs': {
            'res': True,
            'kernel_size': 9,
        },
        'threshold_kwargs':{
            'noise_std_norm': True,
        },
        'n_scales': 4,
        'n_reweights_learn': 1,
        'exact_reconstruction': exact_reco,
        'undecimated': undecimated,
        'clip': False,
    }

    n_epochs = 100
    run_id = f'ista_learnlet_fastmri_{n_filters}_{n_iters}_{denoising_activation}_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)

    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=False)
    log_dir = op.join(f'{LOGS_DIR}logs', run_id)
    tboard_cback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        profile_batch=0,
    )

    # run distributed
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = IstaLearnlet(
            n_iterations=n_iters,
            forward_operator=tf_masked_shifted_normed_fft2d,
            adjoint_operator=tf_masked_shifted_normed_ifft2d,
            postprocess=_tf_crop,
            **learnlet_params,
        )
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss='mse',
            metrics=[keras_psnr, keras_ssim],
        )

    model.fit(
        train_set,
        steps_per_epoch=200,
        epochs=n_epochs,
        validation_data=val_set,
        validation_steps=1,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback,],
        shuffle=False,
    )

if __name__ == '__main__':
    train_ista_learnlet()
