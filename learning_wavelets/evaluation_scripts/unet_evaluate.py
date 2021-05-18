import tensorflow as tf
from tqdm import tqdm

from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_bsd68
from learning_wavelets.evaluate import METRIC_FUNCS, Metrics 
from learning_wavelets.models.unet import unet

tf.random.set_seed(1)


def evaluate_old_unet(
        noise_std_test=30,
        run_id='unet_64_bsd500_0_55_None_1620201240',
        n_epochs=500,
        base_n_filters=64, 
        n_layers=5,
        layers_n_non_lins=2,
        n_samples=None,
    ):
    
    noise_std_test = force_list(noise_std_test)
    
    run_params = {
        'n_layers': n_layers,
        'pool': 'max',
        "layers_n_channels": [base_n_filters * 2**i for i in range(0, n_layers)],
        'layers_n_non_lins': layers_n_non_lins,
        'non_relu_contract': False,
        'bn': True,
    }
    
    n_channels = 1
    model = unet(input_size=(None, None, n_channels), **run_params)
        
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs}.hdf5')
    
    metrics_per_noise_level = {}
    
    for noise_level in noise_std_test:
        val_set = im_dataset_bsd68(
            mode='testing',
            patch_size=None,
            noise_std=noise_level,
            return_noise_level=False,
            n_samples=n_samples,
        )
    
        
        eval_res = Metrics()
        for x, y_true, size in tqdm(val_set.as_numpy_iterator()):
            y_pred = model.predict(x)
            eval_res.push(y_true, y_pred)
        metrics_per_noise_level[noise_level] = (list(eval_res.means().values()), list(eval_res.stddevs().values()))
        
    return METRIC_FUNCS, metrics_per_noise_level


def force_list(x):
    if not isinstance(x, list):
        return [x]
    else:
        return x
