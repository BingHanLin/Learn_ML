import numpy as np
import tensorflow as tf


class Initial_Condition():
    def __init__(self, inputs: dict, value, name="dirichlet"):
        self._name = name

        x_inputs, t_inputs = tf.meshgrid(inputs['x'], inputs['t'])
        self._x_inputs = tf.reshape(x_inputs, [-1])
        self._t_inputs = tf.reshape(t_inputs, [-1])

        self._value = tf.convert_to_tensor(value, dtype=tf.float32)
        self._value = tf.reshape(self._value, [len(self._value), 1])

    def computeLoss(self, network, optimizer):
        inputs = tf.stack([self._x_inputs, self._t_inputs], -1)
        results = network(inputs)

        loss = tf.reduce_mean(tf.square(results-self._value))

        return loss


if __name__ == '__main__':
    x = tf.linspace(0, 1, 3)
    t = tf.linspace(0, 0, 1)
    condition = Initial_Condition({'x': x, 't': t}, 0.0)
