import numpy as np
import tensorflow as tf


class Initial_Condition():
    def __init__(self, inputs: dict, value, name="dirichlet"):
        self._name = name
        x_inputs, t_inputs = tf.meshgrid(inputs['x'], inputs['t'])
        self._x_inputs = tf.reshape(x_inputs, [-1])
        self._t_inputs = tf.reshape(t_inputs, [-1])

        self._value = value

    def train(self, network, optimizer):

        # for i in range(5):
        with tf.GradientTape() as tape:
            inputs = tf.stack([self._x_inputs, self._t_inputs], -1)
            results = network(inputs)
            loss = tf.reduce_mean(tf.square(results-self._value))

        trainable_vars = network.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))


if __name__ == '__main__':
    x = tf.linspace(0, 1, 3)
    t = tf.linspace(0, 0, 1)
    condition = Initial_Condition({'x': x, 't': t}, 0.0)
