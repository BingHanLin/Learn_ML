import numpy as np
from boundaryCondition import Boundary_Condition
import tensorflow as tf


# class DirichletDataset(torch.utils.data.Dataset):
#     def __init__(self, data, device="cpu"):
#         X = np.stack(np.meshgrid(*data[0]), -1).reshape(-1, len(data[0]))
#         Y = np.stack(data[1], axis=1)
#         assert len(X) == len(
#             Y), "the length of the inputs and outputs don't match"
#         self.X = torch.from_numpy(X).float().to(device)
#         self.Y = torch.from_numpy(Y).float().to(device)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, ix):
#         return self.X[ix], self.Y[ix]


class Dirichlet(Boundary_Condition):
    def __init__(self, inputs: dict, value, name="dirichlet"):
        super().__init__(name)

        self._inputs = np.stack(np.meshgrid(*inputs.values()), -
                                1).reshape(-1, len(inputs.values()))

        self._value = value

    # def train(self, network, optimizer):
    #     with tf.GradientTape() as tape:
    #         results = network(self._inputs)
    #         loss = tf.reduce_mean(tf.square(results[0]-self._value))

    #     trainable_vars = network.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     optimizer.apply_gradients(zip(gradients, trainable_vars))


if __name__ == '__main__':
    x = np.linspace(0, 1, 3)
    t = np.linspace(0, 1, 5)
    u = np.linspace(-1, 1, 10)
    condition = Dirichlet({'x': x, 't': t}, 0.0)
