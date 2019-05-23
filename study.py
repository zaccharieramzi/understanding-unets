import os.path as op

import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard
from keras_tqdm import TQDMCallback
from tensorflow import set_random_seed
from tqdm import tqdm_notebook

from data import im_generator
from unet import unet

np.random.seed(1)
set_random_seed(1)


batch_size = 32
noise_std = 30
source = 'cifar_grey'
validation_split = 0.1
if 'cifar' in source:
    n_samples_train = 5*1e4
    size = 32
elif 'mnist' in source:
    n_samples_train = 6*1e4
    size = 28
n_samples_test = 1e4
im_gen_train = im_generator(mode='training', validation_split=0.1, batch_size=batch_size, source=source, noise_std=noise_std)
im_gen_val = im_generator(mode='validation', validation_split=0.1, batch_size=batch_size, source=source, noise_std=noise_std)
im_gen_test = im_generator(mode='testing', batch_size=batch_size, source=source, noise_std=noise_std)


params = {
    'cifar_classic_1': {'n_layers': 2},
    'cifar_without_relu_contracting_1': {'n_layers': 2, 'non_relu_contract': True},
    'cifar_aver_pool_1': {'n_layers': 2, 'pool': 'average'},
    'cifar_classic_2': {'n_layers': 3},
    'cifar_without_relu_contracting_2': {'n_layers': 3, 'non_relu_contract': True},
    'cifar_aver_pool_2': {'n_layers': 3, 'pool': 'average'},
    'cifar_classic_3': {'n_layers': 4},
    'cifar_without_relu_contracting_3': {'n_layers': 4, 'non_relu_contract': True},
    'cifar_aver_pool_3': {'n_layers': 4, 'pool': 'average'},
}


n_epochs = 1
K.clear_session()
for run_id, run_params in tqdm_notebook(params.items()):
    print(run_id)
    model = unet(input_size=(size, size, 1), with_extra_sigmoid=False, **run_params)
    log_dir = op.join('logs', run_id)
    tboard_cback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_images=True,
    )
    model.fit_generator(
        im_gen_train,
        steps_per_epoch=int((1-validation_split) * n_samples_train / batch_size),
        epochs=n_epochs,
        validation_data=im_gen_val,
        validation_steps=int(validation_split * n_samples_train / batch_size),
        verbose=0,
        callbacks=[TQDMCallback(), tboard_cback],
    )
    K.clear_session()
