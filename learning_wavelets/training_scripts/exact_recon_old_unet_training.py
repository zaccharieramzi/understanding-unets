import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_div2k, im_dataset_bsd500
from learning_wavelets.models.exact_recon_old_unet import exact_recon_old_unet as unet


tf.random.set_seed(1)

def train_old_unet(
        noise_std_train=(0, 55),
        noise_std_val=30,
        n_samples=None,
        source='bsd500',
        layers_n_non_lins=2,
        base_n_filters=64,
        n_layers=5,
        batch_size=8,
        n_epochs=500,
        exact_recon=False,
        residual=False,
        use_bias=True,
    ):

    # data preparation
    if source == 'bsd500':
        data_func = im_dataset_bsd500
    elif source == 'div2k':
        data_func = im_dataset_div2k
    im_ds_train = data_func(
        mode='training',
        batch_size=batch_size,
        noise_std=noise_std_train,
        return_noise_level=True,
        n_samples=n_samples,
    )
    im_ds_val = data_func(
        mode='validation',
        batch_size=batch_size,
        noise_std=noise_std_val,
        return_noise_level=True,
    )

    run_params = {
        'n_layers': n_layers,
        'pool': 'max',
        "layers_n_channels": [base_n_filters * 2**i for i in range(0, n_layers)],
        'layers_n_non_lins': layers_n_non_lins,
        'non_relu_contract': False,
        'bn': False,
        'use_bias': use_bias,
        'exact_recon': exact_recon,
        'residual': residual,
    }

    run_id = f'unet_{base_n_filters}_{source}_{noise_std_train[0]}_{noise_std_train[1]}_{n_samples}'
    if exact_recon:
        run_id += '_exact_reco'
    elif residual:
        run_id += '_residual'
    if not use_bias:
        run_id += '_biasfree'
    run_id += f'_{int(time.time())}'
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}' + '-{epoch:02d}.hdf5'
    print(run_id)

    # callbacks preparation

    chkpt_cback = ModelCheckpoint(chkpt_path, period=min(500, n_epochs), save_weights_only=True)
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
        model = unet(input_size=(None, None, n_channels), lr=1e-3, **run_params)
        # model.compile(optimizer=tfa.optimizers.RectifiedAdam(), loss='mse'), check this later
    print(model.summary(line_length=114))


    # actual training
    model.fit(
        im_ds_train,
        steps_per_epoch=200,
        epochs=n_epochs,
        validation_data=im_ds_val,
        validation_steps=15,
        verbose=1,
        callbacks=[tboard_cback, chkpt_cback],
        shuffle=False,
    )

    return run_id