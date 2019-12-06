from tensorflow.keras.layers import  Input, Conv2D, Activation, BatchNormalization, Subtract
from tensorflow.keras.models import Model

def dncnn(input_size=(None, None, 1), filters=64, depth=20):
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
            use_bias=False,
        )(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
    x = Conv2D(
        filters=input_size[-1],
        kernel_size=3,
        # kernel_initializer='Orthogonal',  # this is only in FFDNet
        padding='same',
        use_bias=False,
    )(x)
    x = Subtract()([inpt, x])
    model = Model(inputs=inpt, outputs=x)

    return model
