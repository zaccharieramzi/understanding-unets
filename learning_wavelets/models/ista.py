import tensorflow as tf
from tensorflow.keras.models import Model

from .learnlet_model import Learnlet

class IstaLearnlet(Model):
    __name__ = 'ista_learnlet'
    def __init__(self, n_iterations, forward_operator, adjoint_operator, operator_lips_cst=None, **learnlet_kwargs):
        super(IstaLearnlet, self).__init__()
        self.n_iterations = n_iterations
        self.forward_operator = forward_operator
        self.adjoint_operator = adjoint_operator
        self.learnlet = Learnlet(**learnlet_kwargs)
        if operator_lips_cst is None:
            # TODO: think of sthg better
            operator_lips_cst = 1.0
        self.alphas = [
            self.add_weight(
                shape=(1,),
                initializer=tf.constant_initializer(operator_lips_cst),
                name=f'alpha_{i}',
                constraint=tf.keras.constraints.NonNeg(),
            ) for i in range(self.n_iterations)
        ]

    def call(self, inputs):
        measurements, subsampling_pattern = inputs
        x = self.adjoint_operator(measurements[..., 0], subsampling_pattern)[..., None]
        for i in range(self.n_iterations):
            # ISTA-step
            grad = self.grad(x, inputs)
            alpha = tf.cast(self.alphas[i], grad.dtype)
            x = x - alpha * grad
            # learnlet transform on both the real and imaginary part
            x_real = self.prox(tf.math.real(x), self.alphas[i])
            x_imag = self.prox(tf.math.imag(x), self.alphas[i])
            x = tf.complex(x_real, x_imag)
        x = tf.math.abs(x)
        return x

    def prox(self, x, alpha):
        return self.learnlet([x, alpha])

    def grad(self, x, inputs):
        measurements, subsampling_pattern = inputs
        measurements = measurements[..., 0]
        measurements_residual = self.forward_operator(x[..., 0], subsampling_pattern) - measurements
        return self.adjoint_operator(measurements_residual, subsampling_pattern)[..., None]
