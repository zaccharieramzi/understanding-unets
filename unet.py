"""Largely inspired by https://github.com/zhixuhao/unet/blob/master/model.py"""
from keras import activations
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, concatenate, Dropout, UpSampling2D, Input, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam


def unet_rec(inputs, kernel_size=3, n_layers=1, layers_n_channels=1, layers_n_non_lins=1):
    if n_layers == 1:
        last_conv = chained_convolutions(
            inputs,
            n_channels=layers_n_channels[0],
            n_non_lins=layers_n_non_lins[0],
            kernel_size=kernel_size,
        )
        output = last_conv
    else:
        # TODO: refactor the following
        n_non_lins = layers_n_non_lins[0]
        n_channels = layers_n_channels[0]
        left_u = chained_convolutions(
            inputs,
            n_channels=n_channels,
            n_non_lins=n_non_lins,
            kernel_size=kernel_size,
        )
        rec_input = AveragePooling2D(pool_size=(2, 2))(left_u)
        rec_output = unet_rec(
            inputs=rec_input,
            kernel_size=kernel_size,
            n_layers=n_layers-1,
            layers_n_channels=layers_n_channels[1:],
            layers_n_non_lins=layers_n_non_lins[1:],
        )
        merge = concatenate([left_u, UpSampling2D(size=(2, 2))(rec_output)], axis=3)
        output = chained_convolutions(
            merge,
            n_channels=n_channels,
            n_non_lins=n_non_lins,
            kernel_size=kernel_size,
        )
    return output


def unet(
        with_extra_sigmoid=True,
        pretrained_weights=None,
        input_size=(256, 256, 1),
        kernel_size=3,
        n_layers=1,
        layers_n_channels=1,
        layers_n_non_lins=1,
    ):
    if isinstance(layers_n_channels, int):
        layers_n_channels = [layers_n_channels] * n_layers
    else:
        assert len(layers_n_channels) == n_layers
    if isinstance(layers_n_non_lins, int):
        layers_n_non_lins = [layers_n_non_lins] * n_layers
    else:
        assert len(layers_n_non_lins) == n_layers
    inputs = Input(input_size)
    output = unet_rec(
        inputs,
        kernel_size=kernel_size,
        n_layers=n_layers,
        layers_n_channels=layers_n_channels,
        layers_n_non_lins=layers_n_non_lins,
    )
    if with_extra_sigmoid:
        new_output = Conv2D(
            input_size[-1],
            kernel_size,
            activation='sigmoid',
            padding='same',
            kernel_initializer='he_normal',
        )(output)
        model = Model(inputs=inputs, outputs=new_output)
    else:
        model = Model(input=inputs, outputs=output)
        model.layers[-1].activation = activations.sigmoid
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def old_unet(pretrained_weights=None, input_size=(256, 256, 1), dropout=0.5, kernel_size=3):
    inputs = Input(input_size)
    conv1 = Conv2D(1, kernel_size , activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv5 = Conv2D(1, kernel_size , activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(1, kernel_size , activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv1,up6], axis=3)
    # conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(input_size[-1], kernel_size , activation='sigmoid', padding='same', kernel_initializer='he_normal')(merge6)



    model = Model(input=inputs, output=conv6)

    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')

    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def chained_convolutions(inputs, n_channels=1, n_non_lins=1, kernel_size=3):
    conv = inputs
    for _ in range(n_non_lins):
        conv = Conv2D(
            n_channels,
            kernel_size,
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
        )(conv)
    return conv
