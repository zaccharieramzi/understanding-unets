import os
import os.path as op
import time

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import matplotlib.pyplot as plt

from learning_wavelets.config import LOGS_DIR, CHECKPOINTS_DIR
from tensorflow.keras.optimizers import Adam
from learning_wavelets.data.datasets import im_dataset_div2k, im_dataset_bsd500
from learning_wavelets.models.exact_recon_unet import ExactReconUnet


tf.random.set_seed(1)


def train_unet(noise_std_train, noise_std_val, n_samples, source, cuda_visible_devices, base_n_filters):

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_visible_devices)
    # data preparation
    batch_size = 8
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
    )
    im_ds_val = data_func(
        mode='validation',
        batch_size=batch_size,
        patch_size=256,
        noise_std=noise_std_val,
        return_noise_level=True,
    )

    n_epochs = 200
    run_id = f'ExactReconUnet_{base_n_filters}_dynamic_st_{source}_{noise_std_train[0]}_{noise_std_train[1]}_{n_samples}_{int(time.time())}'
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

    
    # run distributed 
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model=ExactReconUnet(n_output_channels=1, kernel_size=3, layers_n_channels=[4, 8, 16, 32])
        model.compile(optimizer=Adam(lr=1e-3), loss='mse')
    

    # actual training
    history = model.fit(
        im_ds_train,
        steps_per_epoch=200,
        epochs=n_epochs,
        validation_data=im_ds_val,
        validation_steps=1,
        verbose=1,
        callbacks=[tboard_cback, chkpt_cback, lrate_cback],
        shuffle=False,
    )
    
    plt.plot(history.history['loss'], label='Loss (training data)')
    plt.plot(history.history['val_loss'], label='Loss (validation data)')
    plt.title('Loss of the Learnlets on the Vertical Set')
    plt.ylabel('Loss value')
    plt.yscale('log')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()
    
    print(model.summary(line_length=114))
        
if __name__ == '__main__':
    train_unet()
