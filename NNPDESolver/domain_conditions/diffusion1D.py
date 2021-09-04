import numpy as np
from .domainCondition import Domain_Condition
import tensorflow as tf


class Diffusion1D(Domain_Condition):
    def __init__(self, inputs: dict, name="dirichlet"):
        super().__init__(name)
        x_inputs, t_inputs = tf.meshgrid(inputs['x'], inputs['t'])
        self._x_inputs = tf.reshape(x_inputs, [-1])
        self._t_inputs = tf.reshape(t_inputs, [-1])

    def computeLoss(self, network, optimizer):
        inputs = tf.stack([self._x_inputs, self._t_inputs], -1)
        results = network(inputs)

        dphidt = tf.gradients(results, self._t_inputs)[0]
        dphidx = tf.gradients(results, self._x_inputs)[0]
        dphidxx = tf.gradients(dphidx, self._x_inputs)[0]

        # tf.print("results ", results)
        # tf.print("dphidt ", tf.gradients(results, self._t_inputs))
        # tf.print("dphidx ", dphidx)
        # tf.print("dphidxx ", dphidxx)

        governing_eq = dphidt - 0.3*dphidxx

        loss = tf.reduce_sum(tf.square(governing_eq))

        return loss


if __name__ == '__main__':
    x = tf.linspace(0, 1, 3)
    t = tf.linspace(0, 1, 5)
    condition = Diffusion1D({'x': x, 't': t})
