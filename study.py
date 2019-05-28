import json
import os.path as op
import time

import click
from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np
from tensorflow import set_random_seed

from data import im_generator
from unet import unet

np.random.seed(1)
set_random_seed(1)


def handle_source(source, batch_size=32, noise_std=30, validation_split=0.1):
    if 'cifar' in source:
        n_samples_train = 5*1e4
        size = 32
    elif 'mnist' in source:
        n_samples_train = 6*1e4
        size = 28
    if source == 'cifar10':
        n_channels = 3
    else:
        n_channels = 1
    im_gen_train = im_generator(mode='training', validation_split=validation_split, batch_size=batch_size, source=source, noise_std=noise_std)
    im_gen_val = im_generator(mode='validation', validation_split=validation_split, batch_size=batch_size, source=source, noise_std=noise_std)
    im_gen_test = im_generator(mode='testing', batch_size=batch_size, source=source, noise_std=noise_std)
    return im_gen_train, im_gen_val, im_gen_test, n_samples_train, size, n_channels


def handle_params_file(params_file_path, source):
    with open(params_file_path) as params_file:
        params = json.load(params_file)
    for param in params:
        param[0] = '{source}_{param_desc}'.format(source=source, param_desc=param[0])
    return params


default_params = [
    ('classic_1', {'n_layers': 2}),
    ('without_relu_contracting_1', {'n_layers': 2, 'non_relu_contract': True}),
    ('aver_pool_1', {'n_layers': 2, 'pool': 'average'}),
    ('classic_2', {'n_layers': 3}),
    ('without_relu_contracting_2', {'n_layers': 3, 'non_relu_contract': True}),
    ('aver_pool_2', {'n_layers': 3, 'pool': 'average'}),
    ('classic_3', {'n_layers': 4}),
    ('without_relu_contracting_3', {'n_layers': 4, 'non_relu_contract': True}),
    ('aver_pool_3', {'n_layers': 4, 'pool': 'average'}),
]

@click.command()
@click.option('--params-id', '-p', help='The id of the params in the list of params.', default=0)
@click.option('--params-file', '-f', help='File containing the list of params in a json format.', default=None, type=click.Path())
@click.option('--source', '-s', help='Dataset on which to run the experiment.', type=click.Choice(['cifar_grey', 'cifar10', 'mnist']), default='mnist')
@click.option('--epochs', '-e', default=50, help='number of epochs to run.')
def run_study(params_id, params_file, source, epochs):
    validation_split = 0.1
    batch_size = 32
    start = time.time()
    K.clear_session()
    if params_file is None:
        params = default_params
    else:
        params = handle_params_file(params_file, source)
    run_id, run_params = params[params_id]
    im_gen_train, im_gen_val, _, n_samples_train, size, n_channels = handle_source(source, batch_size=batch_size, noise_std=30, validation_split=validation_split)
    print('Running {run_id}, for {epochs} epochs'.format(run_id=run_id, epochs=epochs))
    model = unet(input_size=(size, size, n_channels), **run_params)
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
        epochs=epochs,
        validation_data=im_gen_val,
        validation_steps=int(validation_split * n_samples_train / batch_size),
        verbose=0,
        callbacks=[tboard_cback],
    )
    K.clear_session()
    end = time.time()
    print('Training took {dur} mins'.format(dur=(end-start)/60))


if __name__ == '__main__':
    run_study()
