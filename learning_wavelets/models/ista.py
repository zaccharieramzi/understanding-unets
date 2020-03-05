import tensorflow as tf
from tensorflow.keras.models import Model

from .learnlet_model import Learnlet

class IstaLearnlet(Model):
    def __init__(self, n_iterations, forward_operator, adjoint_operator, operator_lips_cst=None, **learnlet_kwargs):
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
        x = self.adjoint_operator(inputs)
        for i in range(self.n_iterations):
            # ISTA-step
            x = self.prox(x - self.alphas[i] * self.grad(x, inputs), self.alphas[i])
        return x

    def prox(self, x, alpha):
        return self.learnlet([x, alpha])

    def grad(self, x, y):
        return self.adjoint_operator(self.forward_operator(x) - y)
