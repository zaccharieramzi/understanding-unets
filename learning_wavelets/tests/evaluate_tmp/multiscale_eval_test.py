import numpy as np
import tensorflow as tf

from learning_wavelets.evaluate_tmp.multiscale_eval import evaluate_multiscale


def test_evaluate_multiscale_backbone():
    noise_lvl = 0.001
    model = lambda x: x + noise_lvl
    distrib_strat = tf.distribute.MirroredStrategy()
    metrics_names, eval_res = evaluate_multiscale(
        model,
        distrib_strat,
        n_scales=5,
        dynamic_denoising=False,
        noise_stds=[0],
        n_samples=10,
    )
    for metric_name, res in zip(metrics_names, eval_res):
        if metric_name == 'keras_psnr':
            np.testing.assert_almost_equal(
                res,
                10 * np.log10(1 / (noise_lvl)**2),
            )
