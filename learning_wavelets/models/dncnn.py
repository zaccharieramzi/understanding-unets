from tensorflow.keras.layers import  Input, Conv2D, Activation, BatchNormalization, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .evaluate import keras_psnr, keras_ssim


def dncnn(input_size=(None, None, 1), filters=64, depth=20, lr=1e-3, bn=True):
    # the code is from https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_keras/main_train.py
    inpt = Input(shape=input_size)
    # 1st layer, Conv+relu
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        # kernel_initializer='Orthogonal',  # this is only in FFDNet
        padding='same',
    )(inpt)
    x = Activation('relu')(x)
    for i in range(depth-2):
        x = Conv2D(
            filters=filters,
            kernel_size=3,
            # kernel_initializer='Orthogonal',  # this is only in FFDNet
            padding='same',
            use_bias=not bn,
        )(x)
        if bn:
            x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.9)(x)
        x = Activation('relu')(x)
    x = Conv2D(
        filters=input_size[-1],
        kernel_size=3,
        # kernel_initializer='Orthogonal',  # this is only in FFDNet
        padding='same',
        use_bias=True,
    )(x)
    x = Subtract()([inpt, x])
    model = Model(inputs=inpt, outputs=x)
    return model
