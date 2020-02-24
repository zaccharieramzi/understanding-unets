import os
import os.path as op
import time

import click
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.datasets import im_dataset_div2k, im_dataset_bsd500
from learning_wavelets.keras_utils.normalisation import NormalisationAdjustment
from learning_wavelets.learned_wavelet import learnlet


# In[4]:


tf.random.set_seed(1)

@click.command()
@click.option(
    'cuda_visible_devices',
    '-gpus',
    '--cuda-visible-devices',
    default='0123',
    type=str,
    help='The visible GPU devices. Defaults to 0123',
)
def train_learnlet_gmca(cuda_visible_devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    run_params = {
        'denoising_activation': 'dynamic_soft_thresholding',
        'learnlet_analysis_kwargs':{
            'n_tiling': 64,
            'mixing_details': False,
            'kernel_size': 11,
            'skip_connection': True,
        },
        'noise_std_norm': True,
        'learnlet_synthesis_kwargs': {
            'res': True,
            'kernel_size': 13,
        },
        'wav_type': 'starlet',
        'n_scales': 5,
        'exact_reconstruction_weight': 0,
        'clip': False,
    }
    source = 'bsd500'
    run_id = f'learnlet_dynamic_gmca_{source}_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)

    def l_rate_schedule(epoch):
        return max(1e-3 / 2**(epoch//50), 1e-5)
    lrate_cback = LearningRateScheduler(l_rate_schedule)

    if source == 'bsd500':
        data_func = im_dataset_bsd500
    else:
        data_func = im_dataset_div2k
    noise_std_val = 30
    batch_size = 8
    im_ds_val = data_func(
        mode='validation',
        batch_size=batch_size,
        patch_size=256,
        noise_std=noise_std_val,
        return_noise_level=True,
    )


    chkpt_cback = ModelCheckpoint(chkpt_path, period=100, save_weights_only=False)
    log_dir = op.join(f'{LOGS_DIR}logs', run_id)
    tboard_cback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        profile_batch=0,
    )
    norm_cback = NormalisationAdjustment(momentum=0.99, n_pooling=5, dynamic_denoising=True)
    norm_cback.on_train_batch_end = norm_cback.on_batch_end

    n_channels = 1
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = learnlet(input_size=(None, None, n_channels), lr=1e-3, **run_params)
    print(model.summary(line_length=114))


    total_epochs = 500
    noise_stds = np.linspace(75, 0, total_epochs)
    for i_epoch, noise_std in enumerate(noise_stds):
        noise_std_train = (noise_std, noise_std + 5)
        im_ds_train = data_func(
            mode='training',
            batch_size=batch_size,
            patch_size=256,
            noise_std=noise_std_train,
            return_noise_level=True,
            no_noise=True,
        )
        model.fit(
            im_ds_train,
            steps_per_epoch=200,
            epochs=i_epoch + 1,
            initial_epoch=i_epoch,
            validation_data=im_ds_val,
            validation_steps=1,
            verbose=0,
            callbacks=[tboard_cback, chkpt_cback, norm_cback, lrate_cback],
            shuffle=False,
        )

if __name__ == '__main__':
    train_learnlet_gmca()
