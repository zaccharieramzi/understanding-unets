import tensorflow as tf
from tqdm import tqdm

from learning_wavelets.data.datasets import im_dataset_bsd68
from learning_wavelets.models.multiscale import MultiScale
from learning_wavelets.training.compile import default_model_compile


DEFAULT_NOISE_STDS = (0.0001, 5, 15, 20, 25, 30, 50, 55, 60, 75)

def evaluate_multiscale(
        model,
        n_scales=5,
        dynamic_denoising=True,
        noise_stds=DEFAULT_NOISE_STDS,
        n_samples=None,
        batch_size=8,
    ):
    metrics = []
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        multiscale_model = MultiScale(
            model,
            n_scales=n_scales,
            dynamic_denoising=dynamic_denoising,
        )
        default_model_compile(multiscale_model)
    for noise_std in tqdm(noise_stds, 'Noise stds'):
        im_ds = im_dataset_bsd68(
            mode='testing',
            batch_size=batch_size,
            patch_size=None,
            noise_std=noise_std,
            return_noise_level=True,
            n_samples=n_samples,
        )
        metrics.append(model.evaluate(im_ds, verbose=1))
    return metrics
