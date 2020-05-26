import os
import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_div2k, im_dataset_bsd500
from learning_wavelets.evaluate import keras_psnr, keras_ssim, center_keras_psnr
from learning_wavelets.keras_utils.normalisation import NormalisationAdjustment
from learning_wavelets.models.learnlet_model import Learnlet

tf.random.set_seed(1)


def train_learnlet(
        noise_std_train=(0, 55),
        noise_std_val=30,
        n_samples=None,
        source='bsd500',
        cuda_visible_devices='0123',
        denoising_activation='dynamic_soft_thresholding',
        n_filters=256,
        decreasing_noise_level=False,
        undecimated=True,
        exact_reco=True,
        n_reweights=1,
        n_epochs=500,
        batch_size=8,
        steps_per_epoch=200,
    ):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    # data preparation
    if source == 'bsd500':
        data_func = im_dataset_bsd500
    elif source == 'div2k':
        data_func = im_dataset_div2k
    im_ds_train = data_func(
        mode='training',
        batch_size=batch_size,
        patch_size=256,
        noise_std=noise_std_train,
        return_noise_level=True,
        n_samples=n_samples,
        decreasing_noise_level=decreasing_noise_level,
    )
    im_ds_val = data_func(
        mode='validation',
        batch_size=batch_size,
        patch_size=256,
        noise_std=noise_std_val,
        return_noise_level=True,
    )

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
        'threshold_kwargs':{
            'noise_std_norm': True,
        },
        'n_scales': 5,
        'n_reweights_learn': n_reweights,
        'exact_reconstruction': exact_reco,
        'undecimated': undecimated,
        'clip': False,
    }

    undecimated_str = 'decimated'
    if undecimated:
        undecimated_str = 'un' + undecimated_str
    if exact_reco:
        undecimated_str += '_exact_reco'
    run_id = f'learnlet_subclassed_{undecimated_str}_{n_filters}_{denoising_activation}_{source}_{noise_std_train[0]}_{noise_std_train[1]}_{n_samples}_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)




    def l_rate_schedule(epoch):
        return max(1e-3 / 2**(epoch//25), 1e-5)
    lrate_cback = LearningRateScheduler(l_rate_schedule)




    chkpt_cback = ModelCheckpoint(chkpt_path, period=n_epochs, save_weights_only=False)
    log_dir = op.join(f'{LOGS_DIR}logs', run_id)
    tboard_cback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        profile_batch=0,
    )
    norm_cback = NormalisationAdjustment(momentum=0.99, n_pooling=5)
    norm_cback.on_train_batch_end = norm_cback.on_batch_end


    # run distributed
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = Learnlet(**run_params)
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss='mse',
            metrics=[keras_psnr, keras_ssim, center_keras_psnr],
        )

    model.fit(
        im_ds_train,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        validation_data=im_ds_val,
        validation_steps=1,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback, norm_cback, lrate_cback],
        shuffle=False,
    )
