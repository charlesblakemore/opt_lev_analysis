import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.optimize as opti

sigma = 0.3
mu = 2.0

seps   = np.array([15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
heights = np.array([-3.0, -2.0, -1.0,  0.0,  1.0,  2.0,  3.0])


seps_g, heights_g = np.meshgrid(seps, heights, indexing='ij')




def plane(x, z, a, b, c):
    return a * x + b * z + c


def cost_function(params, Ndof=False):
    a, b, c = params
    cost = 0.0
    N = 0
    func_vals = plane(seps_g, heights_g, a, b, c)
    for gridind, grid in enumerate(rand_grids):
        diff = np.abs(grid - func_vals)
        cost += np.sum(diff**2 / err_grids[gridind]**2)
        N += diff.size
    if Ndof:
        cost *= (1.0 / float(N))
    return cost
        


gridN = 3
rand_grids = []
err_grids = []
for num in range(gridN):
    rand_grids.append( sigma * np.random.randn( *seps_g.shape ) + mu + \
                       1.0 * heights_g)
    err_grids.append( np.abs(sigma * np.random.randn( *seps_g.shape )) )
    #err_grids.append( np.ones_like(seps_g) )





param_vals = np.linspace(-3,3,201)
a_sweep = []
b_sweep = []
c_sweep = []
for val in param_vals:
    a_sweep.append(cost_function([val, 1.0, 2.0]))
    b_sweep.append(cost_function([0.0, val, 2.0]))
    c_sweep.append(cost_function([0.0, 1.0, val]))
plt.plot(param_vals, a_sweep)
plt.plot(param_vals, b_sweep)
plt.plot(param_vals, c_sweep)

res = opti.minimize(cost_function, [0,0,3*mu])

print res.x

fig = plt.figure()
ax = fig.gca(projection='3d')
for gridind, grid in enumerate(rand_grids):
    ax.scatter(seps_g, heights_g, grid, color='C0')
    ax.scatter(seps_g, heights_g, err_grids[gridind], color='C1')
ax.plot_surface(seps_g, heights_g, plane(seps_g, heights_g, *res.x), \
                alpha=0.2, color='k')
plt.show()

