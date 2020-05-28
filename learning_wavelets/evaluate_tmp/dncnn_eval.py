import tensorflow as tf

from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.models.dncnn import dncnn
from .multiscale_eval import evaluate_multiscale


def evaluate_dncnn(run_id, filters=64, depth=20, bn=False, n_epochs=500, **kwargs):
    # model definition
    run_params = {
        "filters": filters,
        'depth': depth,
        'bn': bn,
    }
    n_channels = 1
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = dncnn(input_size=(None, None, n_channels), **run_params)
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5'
    model.load_weights(chkpt_path)
    metrics = evaluate_multiscale(
        model,
        distrib_strat=mirrored_strategy,
        n_scales=1,
        dynamic_denoising=False,
        **kwargs,
    )
    return metrics
