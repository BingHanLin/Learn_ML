import matplotlib.pyplot as plt
from boundary_conditions import Dirichlet
from domain_conditions import Laplace1D
from initial_conditions import Initial_Condition

from model import Solver_Model
import tensorflow as tf
import numpy as np

lx = 2.0
nx = 20
dx = lx / (nx-1)

lt = 0.04
nt = 4
dt = lt / (nt-1)


x = np.linspace(0.0, lx, nx)
t = np.linspace(0.0, lt, nt)
x_bc1 = np.array([0.0])
x_bc2 = np.array([lx])

t_0 = np.linspace(0.0, 0.0, 1)


bc1 = Dirichlet({'x': x_bc1, 't': t_0}, 0.0)
bc2 = Dirichlet({'x': x_bc2, 't': t_0}, 1.0)
domain = Laplace1D({'x': x, 't': t_0})

model = Solver_Model()
model.add_boundary_condition(bc1)
model.add_boundary_condition(bc2)
model.set_domain_condition(domain)
model.train(10000)

# x = np.linspace(0.0, lx, nx)
# t = np.linspace(0.0, lt, nt)

result = model.predict(x, t_0)

# create a figure and axes
fig = plt.figure(figsize=(8, 5))
ax1 = plt.subplot(1, 1, 1)

# set up the subplots as needed
# ax1.set_xlim((0, 2))
# ax1.set_ylim((-1, 3))
ax1.set_xlabel('Domain')
ax1.set_ylabel('Phi')

# create objects that will change in the animation. These are
# initially empty, and will be given new values for each frame
# in the animation.
line2, = ax1.plot(x,  result, 'r-.', lw=2)

plt.show()

print("end!!!")
