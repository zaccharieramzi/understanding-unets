import tensorflow as tf
from tqdm import tqdm

from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_bsd68
from learning_wavelets.evaluate import METRIC_FUNCS, Metrics 
from learning_wavelets.models.exact_recon_unet import ExactReconUnet

tf.random.set_seed(1)


def evaluate_unet(
        noise_std_test=30,
        run_id='ExactReconUnet_4_bsd500_0_55_None_1620201240',
        n_epochs=500,
        n_output_channels=1,
        kernel_size=3,
        base_n_filters=4, 
        n_layers=4,
        layers_n_non_lins=2,
        non_linearity='relu',
        n_samples=None,
        bn=False,
        exact_recon=False,
        residual=False,
    ):
    
    noise_std_test = force_list(noise_std_test)
    layers_n_channels = [base_n_filters*2**j for j in range(0, n_layers)]
    
    run_params = {
        'n_output_channels': n_output_channels,
        'kernel_size': kernel_size,
        'layers_n_channels': layers_n_channels,
        'layers_n_non_lins': layers_n_non_lins,
        'non_linearity': non_linearity,
        'bn': bn,
        'exact_recon': exact_recon,
        'residual': residual,
    }
    
    model = ExactReconUnet(**run_params)
    
    inputs = [tf.zeros((1, 32, 32, 1)), tf.zeros((1, 1))]
    model(inputs)
        
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs}.hdf5')
    
    metrics_per_noise_level = {}
    
    for noise_level in noise_std_test:
        val_set = im_dataset_bsd68(
            mode='testing',
            patch_size=None,
            noise_std=noise_level,
            n_pooling=5,
            return_noise_level=True,
            n_samples=n_samples,
        )
    
        
        eval_res = Metrics()
        for x, y_true, im_shape in tqdm(val_set.as_numpy_iterator()):
            y_pred = model.predict(x)
            eval_res.push(y_true, y_pred, im_shape=im_shape)
        metrics_per_noise_level[noise_level] = (list(eval_res.means().values()), list(eval_res.stddevs().values()))
        
    return METRIC_FUNCS, metrics_per_noise_level


def force_list(x):
    if not isinstance(x, list):
        return [x]
    else:
        return x
