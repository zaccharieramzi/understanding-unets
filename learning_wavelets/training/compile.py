import tensorflow_addons as tfa

from learning_wavelets.evaluate_tmp.metrics import keras_psnr, keras_ssim

def default_model_compile(model, lr=1e-3):
    model.compile(
        optimizer=tfa.optimizers.RectifiedAdam(lr=lr),
        loss='mean_squared_error',
        metrics=[keras_psnr, keras_ssim],
    )
