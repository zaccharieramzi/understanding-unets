import tensorflow_addons as tfa

from learning_wavelets.evaluate.metrics import keras_psnr, keras_ssim

def default_model_compile(model, lr=1e-3):
    model.compile(
        optimizer=tfa.optimizers.RectifiedAdam(lr=lr, clipnorm=1.),
        loss='mean_squared_error',
        metrics=[keras_psnr, keras_ssim],
    )
