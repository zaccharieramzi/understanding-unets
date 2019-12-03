def unpack_model(init_function=None, run_params=None, run_id=None, epoch=250, **dummy_kwargs):
    model = init_function(**run_params)
    chkpt_path = f'checkpoints/{run_id}-{epoch}.hdf5'
    model.load_weights(chkpt_path)
    return model
