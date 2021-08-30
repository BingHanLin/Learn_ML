import matplotlib.pyplot as plt
from boundary_conditions import Dirichlet
from domain_conditions import Advection1D
from initial_conditions import Initial_Condition

from model import Solver_Model
import tensorflow as tf
import numpy as np

lx = 2
nx = 20
dx = lx / (nx-1)

lt = 0.04
nt = 4
dt = lt / (nt-1)


x = np.linspace(0.0, lx, nx)
t = np.linspace(0.0, lt, nt)
x_bc = np.array([0.0, lx])

t_0 = np.linspace(0.0, 0.0, 1)

# sine wave init
phi_init = np.zeros(nx)
sine_wave = np.linspace(0, 2*np.pi, int(1.5 / dx + 1) - int(.5 / dx))
phi_init[int(.5 / dx):int(1.5 / dx + 1)] += np.sin(sine_wave)

phi_init = tf.convert_to_tensor(phi_init, dtype=tf.float32, name="phi_init")
phi_init = tf.reshape(phi_init, [len(phi_init), 1])

ic = Initial_Condition({'x': x, 't': t_0}, phi_init)
bc = Dirichlet({'x': x_bc, 't': t}, 0.0)
domain = Advection1D({'x': x, 't': t})

model = Solver_Model()
model.add_boundary_condition(bc)
model.add_initial_condition(ic)
model.set_domain_condition(domain)
model.train(500)

# x = np.linspace(0.0, lx, nx)
# t = np.linspace(0.0, lt, nt)

result = model.predict(x, t)

# create a figure and axes
fig = plt.figure(figsize=(8, 5))
ax1 = plt.subplot(1, 1, 1)

# set up the subplots as needed
ax1.set_xlim((0, 2))
ax1.set_ylim((-1, 3))
ax1.set_xlabel('Domain')
ax1.set_ylabel('Phi')

# create objects that will change in the animation. These are
# initially empty, and will be given new values for each frame
# in the animation.
n = 2
line1, = ax1.plot(x,  phi_init, 'gray', lw=2)
line2, = ax1.plot(x,  result[n*nx:(n+1)*nx], 'r-.', lw=2)

plt.show()

print("end!!!")
