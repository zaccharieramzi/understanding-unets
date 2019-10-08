from keras.layers import Activation, Conv2D

def conv_2d(image, n_channels, kernel_size=3, activation='relu'):
    image = Conv2D(
        n_channels,
        kernel_size,
        activation='linear',
        padding='same',
        kernel_initializer='glorot_uniform',
    )(image)
    image = Activation(activation)(image)
    return image
