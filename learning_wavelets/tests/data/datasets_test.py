import pytest

from learning_wavelets.data.datasets import im_dataset_bsd500, im_dataset_bsd68

@pytest.mark.parametrize('im_ds_kwargs',[
    {},
    {'batch_size': 8},
    {'noise_std': [0, 55]},
    {'n_samples': 10},
    {'return_noise_level': True},
    {'patch_size': None, 'n_pooling': 3},
])
@pytest.mark.parametrize('im_ds_func', [im_dataset_bsd500, im_dataset_bsd68])
def test_init(im_ds_kwargs, im_ds_func):
    im_ds_func(**im_ds_kwargs)
