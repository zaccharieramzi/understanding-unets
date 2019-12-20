# Learnlets

Learnlets are a way to learn a filter bank rather than design one like in the curvelets.

This filter bank will be learned in a denoising setting with backpropagation and gradient descent.

## Requirements
The requirements are listed in `learning_wavelets/requirements.txt`.

## Use

The learnlets are defined in `learning_wavelets/learned_wavelet.py`, via the function `learnlet`.

You can use different types of thresholding listed in `learning_wavelets/keras_utils/thresholding.py`.
