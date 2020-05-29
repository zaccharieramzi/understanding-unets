import tensorflow as tf

from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.models.focnet import FocNet
from .multiscale_eval import evaluate_multiscale


def evaluate_focnet(run_id, n_filters=128, beta=0.2, n_epochs=40, **kwargs):
    # model definition
    run_params = {
        'n_filters': n_filters,
        'beta': beta,
    }
    n_channels = 1
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = FocNet(**run_params)
        model.build(tf.TensorShape([None, None, None, 1]))
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5'
    model.load_weights(chkpt_path)
    metrics = evaluate_multiscale(
        model,
        distrib_strat=mirrored_strategy,
        n_scales=run_params.get('n_scales', 4),
        dynamic_denoising=False,
        **kwargs,
    )
    return metrics
