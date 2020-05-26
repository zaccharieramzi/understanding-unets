from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.models.unet import unet
from .multiscale_eval import evaluate_multiscale


def evaluate_unet(run_id, base_n_filters=64, n_epochs=500, **kwargs):
    # model definition
    n_layers = 5
    run_params = {
        'n_layers': n_layers,
        'pool': 'max',
        "layers_n_channels": [base_n_filters * 2**i for i in range(5)],
        'layers_n_non_lins': 2,
        'non_relu_contract': False,
        'bn': True,
    }
    n_channels = 1
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = unet(input_size=(None, None, n_channels), **run_params)
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5'
    model.load_weights(chkpt_path)
    metrics = evaluate_multiscale(
        model,
        n_scales=n_layers,
        dynamic_denoising=False,
        **kwargs,
    )
    return metrics
