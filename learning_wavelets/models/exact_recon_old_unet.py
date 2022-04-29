"""Largely inspired by https://github.com/zhixuhao/unet/blob/master/model.py"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Dropout, UpSampling2D, Input, AveragePooling2D, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ..evaluate import keras_psnr, keras_ssim


def unet_rec(
        inputs,
        kernel_size=3,
        n_layers=1,
        layers_n_channels=1,
        layers_n_non_lins=1,
        pool='max',
        non_relu_contract=False,
        bn=False,
        use_bias=True,
    ):
    if n_layers == 1:
        last_conv = chained_convolutions(
            inputs,
            n_channels=layers_n_channels[0],
            n_non_lins=layers_n_non_lins[0],
            kernel_size=kernel_size,
            bn=bn,
            use_bias=use_bias,
        )
        output = last_conv
    else:
        # TODO: refactor the following
        n_non_lins = layers_n_non_lins[0]
        n_channels = layers_n_channels[0]
        if non_relu_contract:
            activation = 'linear'
        else:
            activation = 'relu'
        left_u = chained_convolutions(
            inputs,
            n_channels=n_channels,
            n_non_lins=n_non_lins,
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
        )
        if pool == 'average':
            pooling = AveragePooling2D
        else:
            pooling = MaxPooling2D
        rec_input = pooling(pool_size=(2, 2))(left_u)
        rec_output = unet_rec(
            inputs=rec_input,
            kernel_size=kernel_size,
            n_layers=n_layers-1,
            layers_n_channels=layers_n_channels[1:],
            layers_n_non_lins=layers_n_non_lins[1:],
            pool=pool,
            non_relu_contract=non_relu_contract,
            use_bias=use_bias,
        )
        merge = concatenate([
            left_u,
            Conv2D(
                n_channels,
                kernel_size - 1,
                activation='relu',
                padding='same',
                kernel_initializer='glorot_uniform',
                use_bias=use_bias,
            )(UpSampling2D(size=(2, 2))(rec_output))  # up-conv
        ], axis=3)
        output = chained_convolutions(
            merge,
            n_channels=n_channels,
            n_non_lins=n_non_lins,
            kernel_size=kernel_size,
            use_bias=use_bias,
        )
    return output


def exact_recon_old_unet(
        input_size=(256, 256, 1),
        kernel_size=3,
        n_layers=1,
        layers_n_channels=1,
        layers_n_non_lins=1,
        non_relu_contract=False,
        pool='max',
        lr=1e-3,
        bn=False,
        use_bias=True,
        exact_recon=False,
        residual=False,
  ):
    if isinstance(layers_n_channels, int):
        layers_n_channels = [layers_n_channels] * n_layers
    else:
        assert len(layers_n_channels) == n_layers
    if isinstance(layers_n_non_lins, int):
        layers_n_non_lins = [layers_n_non_lins] * n_layers
    else:
        assert len(layers_n_non_lins) == n_layers
    noisy_image = Input(input_size)
    noise_std = Input((1))
    #noisy_image = inputs[0]
    #noise_std = inputs[1]
    output = noisy_image
    output = unet_rec(
        output,
        kernel_size=kernel_size,
        n_layers=n_layers,
        layers_n_channels=layers_n_channels,
        layers_n_non_lins=layers_n_non_lins,
        pool=pool,
        non_relu_contract=non_relu_contract,
        bn=bn,
        use_bias=use_bias,
    )
    output = Conv2D(
        4 * input_size[-1],
        1,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
        use_bias=use_bias,
    )(output)
    output = Conv2D(
        input_size[-1],
        1,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
        use_bias=use_bias,
    )(output)
    noise_std_ = tf.reshape(noise_std, shape=[tf.shape(noise_std)[0], 1, 1, 1])
    if exact_recon:
        output = noisy_image - noise_std_ * output
    elif residual:
        output = noisy_image - output

    model = Model(inputs=(noisy_image, noise_std), outputs=output)
    model.compile(
        optimizer=Adam(lr=lr),
        loss='mean_squared_error',
        metrics=[keras_psnr, keras_ssim],
    )

    return model


def chained_convolutions(
    inputs,
    n_channels=1,
    n_non_lins=1,
    kernel_size=3,
    activation='relu',
    bn=False,
    use_bias=True,
):
    conv = inputs
    for _ in range(n_non_lins):
        conv = Conv2D(
            n_channels,
            kernel_size,
            activation=activation,
            padding='same',
            kernel_initializer='glorot_uniform',
            use_bias=use_bias,
        )(conv)
        if bn:
            conv = BatchNormalization()(conv)
    return conv
