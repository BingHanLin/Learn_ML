import matplotlib.pyplot as plt
from boundary_conditions import Dirichlet
from domain_conditions import HeatTransfer1D
from initial_conditions import Initial_Condition

from model import Solver_Model
import tensorflow as tf
import numpy as np

lx = 1.0
nx = 10

lt = 0.1
nt = 1


x = np.linspace(0.0, lx, nx)
t = np.linspace(0.0, lt, nt)
x_bc1 = np.array([0.0])
x_bc2 = np.array([lx])

t_0 = np.linspace(0.0, 0.0, 1)

phi_init = np.linspace(0.0, np.pi, nx)
phi_init = np.cos(phi_init)+1.0


ic = Initial_Condition({'x': x, 't': t_0}, phi_init)
bc1 = Dirichlet({'x': x_bc1, 't': t[1:]}, 2.0)
bc2 = Dirichlet({'x': x_bc2, 't': t[1:]}, 0.0)
domain = HeatTransfer1D({'x': x, 't': t})

model = Solver_Model()
model.add_initial_condition(ic)
model.add_boundary_condition(bc1)
model.add_boundary_condition(bc2)
model.set_domain_condition(domain)
model.train(1500)

result = model.predict(x, t)

# create a figure and axes
fig = plt.figure(figsize=(8, 5))
ax1 = plt.subplot(1, 1, 1)

# set up the subplots as needed
ax1.set_xlim((0.0, lx))
# ax1.set_ylim((-1, 3))
ax1.set_xlabel('Domain')
ax1.set_ylabel('Phi')

# create objects that will change in the animation. These are
# initially empty, and will be given new values for each frame
# in the animation.
line1, = ax1.plot(x, phi_init, 'gray', lw=2)
line2, = ax1.plot(x,  result[:nx], 'r-.', lw=2)
# line3, = ax1.plot(x,  result[nx:2*nx], 'b:', lw=2)
# line4, = ax1.plot(x,  result[2*nx:3*nx], 'g:', lw=2)
# line5, = ax1.plot(x,  result[3*nx:4*nx], 'y:', lw=2)

plt.show()

print("end!!!")
