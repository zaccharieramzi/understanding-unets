from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.models.learnlet_model import Learnlet
from .multiscale_eval import evaluate_multiscale


def evaluate_learnlet(
        run_id,
        denoising_activation='dynamic_soft_thresholding',
        n_filters=256,
        undecimated=True,
        exact_reco=True,
        n_reweights=1,
        n_epochs=500,
        **kwargs,
    ):
    # model definition
    n_scales = 5
    run_params = {
        'denoising_activation': denoising_activation,
        'learnlet_analysis_kwargs':{
            'n_tiling': n_filters,
            'mixing_details': False,
            'skip_connection': True,
            'kernel_size': 11,
        },
        'learnlet_synthesis_kwargs': {
            'res': True,
            'kernel_size': 13,
        },
        'threshold_kwargs':{
            'noise_std_norm': True,
        },
        'n_scales': n_scales,
        'n_reweights_learn': n_reweights,
        'exact_reconstruction': exact_reco,
        'undecimated': undecimated,
        'clip': False,
    }
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = Learnlet(**run_params)
        model.build([[None, None, None, 1], [None, 1]])
    chkpt_path = f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs:02d}.hdf5'
    model.load_weights(chkpt_path)
    metrics = evaluate_multiscale(
        model,
        n_scales=n_scales,
        dynamic_denoising=True,
        **kwargs,
    )
    return metrics
