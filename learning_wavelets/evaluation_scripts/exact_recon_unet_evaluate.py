import tensorflow as tf
from tqdm import tqdm

from learning_wavelets.config import CHECKPOINTS_DIR
from learning_wavelets.data.datasets import im_dataset_bsd68
from learning_wavelets.evaluate import METRIC_FUNCS, Metrics 
from learning_wavelets.models.exact_recon_unet import ExactReconUnet



def evaluate_unet(
        noise_std_test=30,
        run_id='ExactReconUnet_4_bsd500_0_55_None_1620201240',
        n_epochs=500,
        n_output_channels=1,
        kernel_size=3,
        layers_n_channels=[64, 128, 256, 512, 1024],
        layers_n_non_lins=2,
        non_linearity='relu',
    ):
    

    run_params = {
        'n_output_channels': n_output_channels,
        'kernel_size': kernel_size,
        'layers_n_channels': layers_n_channels,
        'layers_n_non_lins': layers_n_non_lins,
        'non_linearity': non_linearity,
    }
    
    model = ExactReconUnet(**run_params)
    
    inputs = [tf.zeros((1, 32, 32, 1)), tf.zeros((1, 1))]
    model(inputs)
        
    model.load_weights(f'{CHECKPOINTS_DIR}checkpoints/{run_id}-{n_epochs}.hdf5')
    
    loss_noise_dic = {}
    
    for noise_level in noise_std_test:
        val_set = im_dataset_bsd68(
            mode='testing',
            batch_size=1,
            patch_size=None,
            noise_std=noise_level,
            return_noise_level=True,
        )
    
        
        eval_res = Metrics()
        for x, y_true, size in tqdm(val_set.as_numpy_iterator()):
            y_pred = model.predict(x)
            eval_res.push(y_true, y_pred)
        loss_noise_dic[noise_level] = METRIC_FUNCS, (list(eval_res.means().values()), list(eval_res.stddevs().values()))
        
    return loss_noise_dic
