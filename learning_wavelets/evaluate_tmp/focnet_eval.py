import tensorflow as tf

from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.models.focnet import focnet
from .multiscale_eval import evaluate_multiscale


def evaluate_focnet(run_id, n_filters=64, beta=0.2, n_epochs=500, **kwargs):
    # model definition
    n_layers = 5
    run_params = {
        'n_filters': n_filters,
        'beta': beta,
    }
    n_channels = 1
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = focnet(**run_params)
        model.build([[None, None, None, 1]])
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5'
    model.load_weights(chkpt_path)
    metrics = evaluate_multiscale(
        model,
        distrib_strat=mirrored_strategy,
        n_scales=n_layers,
        dynamic_denoising=False,
        **kwargs,
    )
    return metrics
