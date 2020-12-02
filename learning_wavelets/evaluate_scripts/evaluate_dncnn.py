from tqdm import tqdm

from learning_wavelets.data.datasets import im_dataset_bsd68
from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.evaluate import Metrics
from learning_wavelets.models.dncnn import dncnn


def evaluate_dncnn(
        run_id,
        noise_stds,
        n_epochs=500,
        n_samples=None,
        bn=True,
    ):
    run_params = {
        'filters': 64,
        'depth': 20,
        'bn': bn,
    }
    n_channels = 1
    model = dncnn(input_size=(None, None, n_channels), lr=1e-3, **run_params)
    chkpt_path = f'{CHECKPOINTS_DIR}/checkpoints/{run_id}-{n_epochs}.hdf5'
    model.load_weights(chkpt_path)
    noise_std_metrics = {}
    for noise_std in tqdm(noise_stds, 'Noise stds'):
        im_ds = im_dataset_bsd68(
            mode='testing',
            batch_size=1,
            patch_size=None,
            noise_std=noise_std,
            return_noise_level=False,
            n_pooling=5,
            n_samples=n_samples,
        )
        metrics = Metrics()
        pred_and_gt_shape = [
            (model.predict_on_batch(images_noisy), images_gt, im_shape)
            for images_noisy, images_gt, im_shape in tqdm(im_ds)
        ]
        for im_recos, images, im_shape in pred_and_gt_shape:
            metrics.push(images.numpy(), im_recos.numpy(), im_shape.numpy())
        noise_std_metrics[noise_std] = metrics
    print(noise_std_metrics)
    return noise_std_metrics
