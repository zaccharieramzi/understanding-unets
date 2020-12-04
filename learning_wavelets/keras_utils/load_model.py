from learning_wavelets.config import CHECKPOINTS_DIR


def unpack_model(init_function=None, run_params=None, run_id=None, epoch=250, **dummy_kwargs):
    model = init_function(**run_params)
    chkpt_path = f'{CHECKPOINTS_DIR}/{run_id}-{epoch}.hdf5'
    model.load_weights(chkpt_path)
    return model
