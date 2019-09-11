import bead_sim_funcs as bsf
import matplotlib.pyplot as plt
import numpy as np

ti = 0
tf = 10
dt = 1e-4



mag = 10.
offset = 1.

tarr = np.arange(ti, tf+dt, dt)

f1 = bsf.step_func(ti, tf, dt, mag, offset)
E_arr= bsf.oscE(ti, tf, dt, mag, 1., 0., 0.)
f2 = E_arr[0]

f = f1 * f2
plt.plot(tarr, f)
plt.show()
