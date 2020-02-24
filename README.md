# Learnlets

Learnlets are a way to learn a filter bank rather than design one like in the curvelets.

This filter bank will be learned in a denoising setting with backpropagation and gradient descent.

## Requirements
The requirements are listed in `learning_wavelets/requirements.txt`.

## Use

The learnlets are defined in `learning_wavelets/learnlet_model.py`, via the class `Learnlet`.

You can use different types of thresholding listed in `learning_wavelets/keras_utils/thresholding.py`.

## List of saved networks

### Exact reconstruction notebook

|                    Model id                    |                         Params                         |
|:----------------------------------------------:|:------------------------------------------------------:|
| learnlet_dynamic_st_bsd500_0_55_1580806694     | the big classical network, with 256 filters + identity |
| learnlet_subclassing_st_bsd500_0_55_1582195807 | 64 filters, subclassed API, exact recon forced         |

### No threshold notebook

|                  Model id                  |                         Params                         |
|:------------------------------------------:|:------------------------------------------------------:|
| learnlet_dynamic_st_bsd500_0_55_1580806694 | the big classical network, with 256 filters + identity |

### Different training noise standard deviations notebook

|                   Model id                  |                         Params                         |
|:-------------------------------------------:|:------------------------------------------------------:|
| learnlet_dynamic_st_bsd500_0_55_1580806694  | the big classical network, with 256 filters + identity |
| learnlet_dynamic_st_bsd500_20_40_1580492805 | same with training on 20;40 noise std                  |
| learnlet_dynamic_st_bsd500_30_1580668579    | same with training on 30 noise std                     |
| unet_dynamic_st_bsd500_0_55_1576668365      | big classical unet with 64 base filters and batch norm |
| unet_dynamic_st_bsd500_20.0_40.0_1581002329 | same with training on 20;40 noise std                  |
| unet_dynamic_st_bsd500_30.0_30.0_1581002329 | same with training on 30 noise std                     |

### General comparison

|                  Model id                  |                         Params                         |
|:------------------------------------------:|:------------------------------------------------------:|
| learnlet_dynamic_st_bsd500_0_55_1580806694 | the big classical network, with 256 filters + identity |
| unet_dynamic_st_bsd500_0_55_1576668365     | big classical unet with 64 base filters and batch norm |
