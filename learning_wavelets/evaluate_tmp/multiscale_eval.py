from tqdm import tqdm

from learning_wavelets.data.datasets import im_dataset_bsd68
from learning_wavelets.models.multiscale import MultiScale
from learning_wavelets.training.compile import default_model_compile


DEFAULT_NOISE_STDS = (0.0001, 5, 15, 20, 25, 30, 50, 55, 60, 75)

def evaluate_multiscale(
        model,
        distrib_strat=None,
        n_scales=5,
        dynamic_denoising=True,
        noise_stds=DEFAULT_NOISE_STDS,
        n_samples=None,
        **dummy_kwargs,
    ):
    metrics = []
    with distrib_strat.scope():
        multiscale_model = MultiScale(
            model,
            n_scales=n_scales,
            dynamic_denoising=dynamic_denoising,
        )
        default_model_compile(multiscale_model)
    for noise_std in tqdm(noise_stds, 'Noise stds'):
        im_ds = im_dataset_bsd68(
            mode='testing',
            batch_size=1,
            patch_size=None,
            noise_std=noise_std,
            return_noise_level=dynamic_denoising,
            n_samples=n_samples,
        )
        metrics.append(multiscale_model.evaluate(im_ds, verbose=1))
    metrics_names = multiscale_model.metrics_names
    return metrics_names, metrics
