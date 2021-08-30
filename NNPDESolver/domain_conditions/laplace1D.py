import numpy as np
from .domainCondition import Domain_Condition
import tensorflow as tf


class Laplace1D(Domain_Condition):
    def __init__(self, inputs: dict, name="dirichlet"):
        super().__init__(name)
        x_inputs, t_inputs = tf.meshgrid(inputs['x'], inputs['t'])
        self._x_inputs = tf.reshape(x_inputs, [-1])
        self._t_inputs = tf.reshape(t_inputs, [-1])

    def train(self, network, optimizer):
        with tf.GradientTape() as tape:
            inputs = tf.stack([self._x_inputs, self._t_inputs], -1)
            results = network(inputs)

            dphidxx = tf.gradients(tf.gradients(
                results, self._x_inputs)[0], self._x_inputs)[0]

            governing_eq = dphidxx
            loss = tf.reduce_sum(tf.square(governing_eq))

        trainable_vars = network.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))


if __name__ == '__main__':
    x = tf.linspace(0, 1, 3)
    t = tf.linspace(0, 1, 5)
    condition = Laplace1D({'x': x, 't': t})
