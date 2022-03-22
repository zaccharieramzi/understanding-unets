import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU, PReLU, UpSampling2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.models import Model


class Conv(Layer):
    def __init__(
        self,
        n_filters,
        kernel_size=3,
        non_linearity='relu',
        bn=False,
        use_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.non_linearity = non_linearity
        self.bn = bn
        self.use_bias = use_bias
        self.conv = Conv2D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation=None,
            use_bias=self.use_bias and not self.bn,
        )
        if self.non_linearity == 'lrelu':
            self.act = LeakyReLU(0.1)
        elif self.non_linearity == 'prelu':
            self.act = PReLU(shared_axes=[1, 2])
        else:
            self.act = Activation(self.non_linearity)

        self.bnorm = BatchNormalization(center=self.use_bias)

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.act(outputs)
        if self.bn:
            outputs = self.bnorm(outputs)
        return outputs

class ConvBlock(Layer):
    def __init__(
        self,
        n_filters,
        kernel_size=3,
        non_linearity='relu',
        bn=False,
        use_bias=True,
        n_non_lins=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.non_linearity = non_linearity
        self.n_non_lins = n_non_lins
        self.bn = bn
        self.use_bias = use_bias
        self.convs = [
            Conv(
                n_filters=self.n_filters,
                kernel_size=self.kernel_size,
                non_linearity=self.non_linearity,
                bn=self.bn,
                use_bias=self.use_bias,
            ) for _ in range(self.n_non_lins)
        ]

    def call(self, inputs):
        outputs = inputs
        for conv in self.convs:
            outputs = conv(outputs)
        return outputs

class UpConv(Layer):
    def __init__(self, n_filters, kernel_size=3, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.conv = Conv2D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation=None,
            use_bias=self.use_bias,
        )
        self.up = UpSampling2D(size=(2, 2))

    def call(self, inputs):
        outputs = self.up(inputs)
        outputs = self.conv(outputs)
        return outputs


class ExactReconUnet(Model):
    def __init__(
            self,
            n_output_channels=1,
            kernel_size=3,
            layers_n_channels=[64, 128, 256, 512, 1024],
            layers_n_non_lins=2,
            non_linearity='relu',
            bn=False,
            use_bias=True,
            exact_recon=False,
            residual=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.n_output_channels = n_output_channels
        self.kernel_size = kernel_size
        self.layers_n_channels = layers_n_channels
        self.n_layers = len(self.layers_n_channels)
        self.layers_n_non_lins = layers_n_non_lins
        self.non_linearity = non_linearity
        self.bn = bn
        self.use_bias = use_bias
        self.exact_recon = exact_recon
        self.residual = residual
        self.down_convs = [
            ConvBlock(
                n_filters=n_channels,
                kernel_size=self.kernel_size,
                non_linearity=self.non_linearity,
                n_non_lins=self.layers_n_non_lins,
                bn=self.bn,
                use_bias=self.use_bias,
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        self.down = MaxPooling2D(pool_size=(2, 2), padding='same')
        self.bottom_conv = ConvBlock(
            n_filters=self.layers_n_channels[-1],
            kernel_size=self.kernel_size,
            non_linearity=self.non_linearity,
            n_non_lins=self.layers_n_non_lins,
            bn=self.bn,
            use_bias=self.use_bias,
        )
        self.up_convs = [
            ConvBlock(
                n_filters=n_channels,
                kernel_size=self.kernel_size,
                non_linearity=self.non_linearity,
                n_non_lins=self.layers_n_non_lins,
                bn=self.bn,
                use_bias=self.use_bias,
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        self.ups = [
            UpConv(
                n_filters=n_channels,
                kernel_size=self.kernel_size,
                use_bias=self.use_bias,
            ) for n_channels in self.layers_n_channels[:-1]
        ]
        self.final_conv = Conv2D(
            filters=self.n_output_channels,
            kernel_size=1,
            padding='same',
            activation=None,
            use_bias=self.use_bias,
        )


    def call(self, inputs):
        scales = []
        noisy_image = inputs[0]
        noise_std = inputs[1]
        outputs = noisy_image
        for conv in self.down_convs:
            outputs = conv(outputs)
            scales.append(outputs)
            outputs = self.down(outputs)
        outputs = self.bottom_conv(outputs)
        for scale, conv, up in zip(scales[::-1], self.up_convs[::-1], self.ups[::-1]):
            outputs = up(outputs)
            outputs = tf.concat([outputs, scale], axis=-1)
            outputs = conv(outputs)
        outputs = self.final_conv(outputs)
        noise_std = tf.reshape(noise_std, shape=[tf.shape(noise_std)[0], 1, 1, 1])
        if self.exact_recon:
            outputs = noisy_image - noise_std * outputs
        elif self.residual:
            outputs = noisy_image - outputs
        return outputs
