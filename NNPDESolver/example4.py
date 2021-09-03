from matplotlib import pyplot  # and the useful plotting library
import numpy  # loading our favorite library
import matplotlib.pyplot as plt
from boundary_conditions import Dirichlet
from domain_conditions import HeatTransfer1D
from initial_conditions import Initial_Condition

from model import Solver_Model
import tensorflow as tf
import numpy as np

lx = 1.0
nx = 10

lt = 0.01
nt = 10


x = np.linspace(0.0, lx, nx)
t = np.linspace(0.0, lt, nt)
x_bc1 = np.array([0.0])
x_bc2 = np.array([lx])

t_0 = np.linspace(0.0, 0.0, 1)

phi_init = np.ones_like(x)
phi_init[int(nx * 0.3):int(nx * 0.7)] = 2.0


ic = Initial_Condition({'x': x, 't': t_0}, phi_init)
bc1 = Dirichlet({'x': x_bc1, 't': t}, 1.0)
bc2 = Dirichlet({'x': x_bc2, 't': t}, 1.0)
domain = HeatTransfer1D({'x': x, 't': t})

model = Solver_Model()
model.add_initial_condition(ic)
model.add_boundary_condition(bc1)
model.add_boundary_condition(bc2)
model.set_domain_condition(domain)
model.train(1)

result = model.predict(x, t)

# create a figure and axes
fig = plt.figure(figsize=(8, 5))
ax1 = plt.subplot(1, 1, 1)

# set up the subplots as needed
ax1.set_xlim((0.0, lx))
# ax1.set_ylim((-1, 3))
ax1.set_xlabel('Domain')
ax1.set_ylabel('Phi')


nu = 0.3  # the value of viscosity
dx = lx / (nx - 1)
dt = lt / (nt - 1)


phi_new = phi_init.copy()
# our placeholder array, un, to advance the solution in time
phi_old = np.ones_like(phi_new)

for n in range(nt):  # iterate through time
    phi_old = phi_new.copy()  # copy the existing values of u into un
    for i in range(1, nx - 1):
        phi_new[i] = phi_old[i] + nu * dt / dx**2 * \
            (phi_old[i+1] - 2 * phi_old[i] + phi_old[i-1])


# create objects that will change in the animation. These are
# initially empty, and will be given new values for each frame
# in the animation.
line1, = ax1.plot(x, phi_init, 'black', lw=2)
line2, = ax1.plot(x,  result[:nx], 'r-.', lw=2)
line3, = ax1.plot(x,  result[(nt-1)*nx:nt*nx], 'b:', lw=2)
line4, = ax1.plot(x,  phi_new, 'g:', lw=2)
plt.show()

print("end!!!")
