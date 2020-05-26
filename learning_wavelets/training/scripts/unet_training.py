import os
import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_div2k, im_dataset_bsd500
from learning_wavelets.models.unet import unet


tf.random.set_seed(1)

def train_unet(
        noise_std_train=(0, 55),
        noise_std_val=30,
        n_samples=None,
        source='bsd500',
        cuda_visible_devices='0123',
        base_n_filters=64,
        n_epochs=500,
        batch_size=8,
        steps_per_epoch=200,
        lr=1e-3,
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
        return_noise_level=False,
        n_samples=n_samples,
    )
    im_ds_val = data_func(
        mode='validation',
        batch_size=batch_size,
        patch_size=256,
        noise_std=noise_std_val,
        return_noise_level=False,
    )

    # model definition
    run_params = {
        'n_layers': 5,
        'pool': 'max',
        "layers_n_channels": [base_n_filters * 2**i for i in range(5)],
        'layers_n_non_lins': 2,
        'non_relu_contract': False,
        'bn': True,
    }
    run_id = f'unet_{base_n_filters}_dynamic_st_{source}_{noise_std_train[0]}_{noise_std_train[1]}_{n_samples}_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)

    # callbacks preparation
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

    n_channels = 1
    # run distributed
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = unet(input_size=(None, None, n_channels), lr=lr, **run_params)

    # actual training
    model.fit(
        im_ds_train,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        validation_data=im_ds_val,
        validation_steps=1,
        verbose=0,
        callbacks=[tboard_cback, chkpt_cback, lrate_cback],
        shuffle=False,
    )
    return run_id
