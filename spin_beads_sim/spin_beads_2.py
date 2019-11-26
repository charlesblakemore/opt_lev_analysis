######################################################
# Bead spinning simulation script. Does single-threaded
# integration of a dipole subject to thermal torques
# and an external electric field
######################################################


import sys
import numpy as np
import scipy.interpolate as interp
import scipy.signal as signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torsion_noise as tn
import bead_sim_funcs as bsfuncs

### Define the savepath
savepath = '/spinsim_data/alldata_Vxy100Vrotchirp_1kHz_dt5us.npy'

# Random seed. Un-comment to use a fixed seed for numpy's
# quasi-random random number generator in order to check 
# deterministic behavior etc. 
#np.random.seed(123456)

### Get the dipole moment and moment of inertia
p0 = bsfuncs.p0
Ibead = bsfuncs.Ibead

### Define Initial Conditions
theta0 = 0   # rad
phi0 = 0            # rad
p0x = p0 * np.sin(theta0) * np.cos(phi0)
p0y = p0 * np.sin(theta0) * np.sin(phi0)
p0z = p0 * np.cos(theta0)

wx = 0.0        
wy = 0.0  
wz = 0.0 

xi_init = np.array([p0x, p0y, p0z, wx, wy, wz])



### Integration parameters
# dt = 1.0e-5
dt = 5.0e-6      # time-step, s
ti = 0           # initial time, s
tf = 300         # final time, s

### Drive parameters
maxvoltage = 10.0
fieldmag = maxvoltage / 4.0e-3
drvmag = 10.0 * maxvoltage / 4.0e-3


tt = np.arange(ti, tf + dt, dt)
Nsamp = len( tt )


# Compute sampling frequency to limit bandwidth of white noise
Fsamp = 1.0 / dt

#### NORMALIZATION OF TORQUE NOISE COULD BE OFF, SHOULD
#### REALLY LOOK INTO THIS MORE
# Compute torque noise from Alex's functional library
thermtorque = np.array([tn.torqueNoise(Nsamp, Fsamp), \
                        tn.torqueNoise(Nsamp, Fsamp), \
                        tn.torqueNoise(Nsamp, Fsamp)])


# Compute damping coefficient from Alex's functional library
beta = tn.beta()



xchirp = bsfuncs.chirpE(ti, tf, dt, drvmag, 1, 1000, 100, \
                        xchirp=True, ychirp=True, zchirp=False, \
                        steady=True, xphi=-90, yphi=0, twait=0, tmax=250)


efield = xchirp






### UNCOMMENT THESE LINES TO LOOK AT EFIELD BEFORE SIMULATING

# fig, axarr = plt.subplots(3,1,sharex=True,sharey=True)

# start_time = bsfuncs.therm_time+100 - 10000*dt
# x = int(start_time / dt)

# for i in [0,1,2]:
#     axarr[i].plot(tt[x:x+100000], efield[i,x:x+100000])
# plt.show()

# raw_input()




# Keeping this function in the integrator script as it depends on the 
# user-defined electric field and torque noise etc.

def system(xi, t, tind):
    '''Function returns the first order derivatives of the vector
    xi, assuming xi = (px, py, pz, omegax, omegay, omegaz), i.e. the 
    orientation of the dipole and angular momentum of the microsphere.'''

    Ex = efield[0,tind]
    Ey = efield[1,tind]
    Ez = efield[2,tind]

    tx = thermtorque[0,tind]
    ty = thermtorque[1,tind]
    tz = thermtorque[2,tind]

    px = xi[0]
    py = xi[1]
    pz = xi[2]

    wx = xi[3]
    wy = xi[4]
    wz = xi[5]

    # Consider p vector to be like radial vector and thus given angular
    # velocities defined about the origin, the components change as:
    dpx = py * wz - pz * wy
    dpy = pz * wx - px * wz
    dpz = px * wy - py * wx

    # Compute torque as p (cross) E + thermal torque - damping
    torque = [py * Ez - pz * Ey + tx - wx * beta,  \
              pz * Ex - px * Ez + ty - wy * beta,  \
              px * Ey - py * Ex + tz - wz * beta]


    return np.array([-dpx, -dpy, -dpz, torque[0] / (Ibead), \
                        torque[1] / (Ibead), torque[2] / (Ibead)])






time, points, energy_vec = bsfuncs.stepper(xi_init, ti, tf, dt, system, \
                                    bsfuncs.rk4, efield=efield, plot=False)
print(points.shape)



outpoints = np.c_[time, points, energy_vec, efield[0,:], efield[1,:], efield[2,:]]

np.save(savepath, outpoints)
#np.save('./data/points_Ez2500Vm_chirp2.npy', outpoints)

plt.show()




