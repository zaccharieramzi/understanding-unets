import click
import os.path as op
import tensorflow as tf
import tensorflow_addons as tfa
import time
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_div2k, im_dataset_bsd500
from learning_wavelets.models.unet import unet



tf.random.set_seed(1)

@click.command()
@click.option(
    'noise_std_train',
    '--ns-train',
    nargs=2,
    default=(0, 55),
    type=float,
    help='The noise standard deviation range for the training set. Defaults to [0, 55]',
)
@click.option(
    'noise_std_val',
    '--ns-val',
    default=30,
    type=float,
    help='The noise standard deviation for the validation set. Defaults to 30',
)
@click.option(
    'n_samples',
    '-n',
    default=None,
    type=int,
    help='The number of samples to use for training. Defaults to None, which means that all samples are used.',
)
@click.option(
    'source',
    '-s',
    default='bsd500',
    type=click.Choice(['bsd500', 'div2k'], case_sensitive=False),
    help='The dataset you wish to use for training and validation, between bsd500 and div2k. Defaults to bsd500',
)
@click.option(
    'base_n_filters',
    '-bnf',
    '--base-n-filters',
    default=4,
    type=int,
    help='The number of filters in the first scale of the u-net. Defaults to 4.',
)
@click.option(
    'n_layers',
    '-nla',
    '--n-layers',
    default=4,
    type=int,
    help='The number of layers in the u-net. Defaults to 4.',
)
@click.option(
    'non_linearity',
    '-nli',
    '--non-linearity',
    default='relu',
    type=str,
    help='The type of non-linearities present in the u-net. Defaults to relu.',
)
@click.option(
    'batch_size',
    '-bs',
    '--batch-size',
    default=8,
    type=int,
    help='The size of the considered batches. Defaults to 8.',
)
@click.option(
    'n_epochs',
    '-nep',
    '--n-epochs',
    default=50,
    type=int,
    help='The number of epochs during training. Default value is 50.',
)
@click.option(
    'bn',
    '-bn',
    default=False,
    type=bool,
    help='Indicates whether batch normalization is used. Defaults to False.',
)
@click.option(
    'exact_recon',
    '-er',
    '--exact-recon',
    default=False,
    type=bool,
    help='Indicates whether exact reconstruction is applied. Defaults to False.',
)


def train_unet(
        noise_std_train, 
        noise_std_val, 
        n_samples, 
        source, 
        base_n_filters, 
        n_layers, 
        non_linearity, 
        batch_size, 
        n_epochs,
        bn,
        exact_recon,
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

    run_id = f'ExactReconUnet_{base_n_filters}_{source}_{noise_std_train[0]}_{noise_std_train[1]}_{n_samples}_{int(time.time())}'
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

    
    # run distributed 
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = ExactReconUnet(
            n_output_channels=1, 
            kernel_size=3, 
            layers_n_channels=[base_n_filters*2**j for j in range(0, n_layers)], 
            non_linearity='relu',
            bn=bn,
            exact_recon=exact_recon,
        )
        model.compile(optimizer=tfa.optimizers.RectifiedAdam(), loss='mse')
    

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
