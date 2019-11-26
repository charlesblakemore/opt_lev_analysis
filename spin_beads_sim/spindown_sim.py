

import sys, time, os
import numpy as np
import dill as pickle

import scipy.interpolate as interp
import scipy.signal as signal
import scipy.constants as constants
import scipy.integrate as integrate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torsion_noise as tn
import bead_util as bu

from numba import jit
from joblib import Parallel, delayed


### Constants
kb = constants.Boltzmann
dipole_units = constants.e * (1e-6)


### Bead-specific constants
p0 = 100.0 #* dipole_units  #  C * m
rhobead = bu.rhobead['val']
mbead_dic = {'val': 84.3e-15, 'sterr': 1.0e-15, 'syserr': 1.5e-15}
mbead = mbead_dic['val']
Ibead = bu.get_Ibead(mbead=mbead_dic)['val']
kappa = bu.get_kappa(mbead=mbead_dic)['val']


### Environmental constants
T = 297
P = 3.5e-6 * 100  # Pressure, converted to pascals
m0 = 18.0 * constants.atomic_mass  # residual gas particl mass, in kg
beta_rot = P * np.sqrt(m0) / kappa


### Intial electric field, and initial conditions
#drive_freq = 50000.0
drive_freq = 110000.5
drive_amp = np.abs(bu.trap_efield([0, 0, 0, 400, -400, 0, 0, 0], nsamp=1)[0])
#N_opt = 2.0 * np.pi * (6000.0) * beta_rot
N_opt = 0
xi_init = np.array([p0, 0.0, 0.0, 0.0, 0.0, 2.0 * np.pi * drive_freq])
# to avoid numerical errors, dipole moment is integrated in units of e * micron
# although anytime a torque appears, the value is cast to the correct SI units
anomaly = 1.0   # Set to 0 if you want to turn of anamolous torques from residual field
real_efield_params = pickle.load(open('./real_drive_dat.p', 'rb'))
efield_rms = real_efield_params[0.0][0]
del real_efield_params[0.0]
del real_efield_params[110000.5]


### Simulation parameters
t_release = 20.0
t_sim = 120.0
#t_sim = 4.0

out_file_length = 2.0
nfiles = int(t_sim / out_file_length)

fsamp = 500000.0
nsamp = int(out_file_length * fsamp)

upsamp = 50.0
fsim = upsamp * fsamp
dt_sim = 1.0 / fsim
nsim = int(t_sim * fsim)

n_mc = 50
#n_mc = 1

# Automatically saves data in /data/old_trap_processed/spinsim_data/
# with subdirectory name defined below
savedir = 'spindowns/sim_110kHz_real-noise'


seed_init = 123456 # for sim_0



############################################################################
############################################################################
############################################################################
############################################################################




real_freqs = list(real_efield_params.keys())
real_freqs.sort(key = lambda x: real_efield_params[x][0])

real_freqs = np.array(real_freqs)
real_amps = np.zeros_like(real_freqs)
real_phases = np.zeros_like(real_freqs)
for freqind, freq in enumerate(real_freqs):
    real_amps[freqind] = real_efield_params[freq][0]
    real_phases[freqind] = real_efield_params[freq][1]





@jit()
def rhs(t, xi):
    '''This function represents the right-hand side of the differential equation
       d(xi)/dt = rhs(t, xi), where xi is a 6-dimensional vector representing the 
       system of a rotating microsphere: {px, py, pz, omegax, omegay, omegaz}, 
       with p the dipole moment and omega the angular velocity. The system is 
       solved in Cartesian coordinates to avoid the branch cuts inherent to 
       integrating phase angles.

       The function computes the following torques:
            thermal torque, white noise with power computed from above global
                                parameters and fluctuation dissipation theorem 
            drag torque, computed as (- beta * omega)
            drive torque, computed as (-1.0) * {px, py, pz} (cross) {Ex, Ey, Ez}
            optical torque, constant torque about the z axis
    '''
    thermal_torque = np.sqrt(4.0 * kb * T * beta_rot * fsim) * np.random.randn(3)
    drag_torque = -1.0 * beta_rot * xi[3:]

    Efield = np.array([drive_amp * np.cos(2.0 * np.pi * drive_freq * t), \
                       drive_amp * np.sin(2.0 * np.pi * drive_freq * t), \
                       0.0])

    Ex = np.sum(real_amps * np.cos( 2.0 * np.pi * real_freqs * t + real_phases)) \
            + efield_rms * np.random.randn()
    anomalous_torque = np.array([0.0, xi[2] * dipole_units * Ex, \
                                    -xi[1] * dipole_units * Ex]) * anomaly

    drive_torque = np.cross(xi[:3]*dipole_units, Efield) * (t <= t_release)

    optical_torque = np.array([0.0, 0.0, N_opt])

    total_torque = drive_torque + drag_torque + thermal_torque + \
                        optical_torque + anomalous_torque

    return np.concatenate( (-1.0*np.cross(xi[:3], xi[3:]), total_torque / Ibead) )



@jit()
def rk4(xi_old, t, delt, system):
    '''4th-order Runge-Kutta integrator method. Takes an input vector
       of the form [x_i, y_i, ..., dx_i/dt, dy_i/dt, ...] and uses a 
       user-defined system which computes the derivatives of each of 
       the coordinates and their corresponding velocities at the given
       instant in time. Essentially, the function 'system' returns:
       [dx_i/dt, dy_i/dt, ..., d^2x_i/dt^2, d^2y_i/dt^2, ...], where 
       the acceleration terms are determined by external fields and 
       thermal fluctuations and such.'''
    k1 = delt * system(t, xi_old)
    k2 = delt * system(t + (delt / 2), xi_old + k1 / 2)
    k3 = delt * system(t + (delt / 2), xi_old + k2 / 2,)
    k4 = delt * system(t + delt, xi_old + k3)
    xi_new = xi_old + (1. / 6.) * (k1 + 2*k2 + 2*k3 + k4)

    ### HARDCODED STUFF IS BAD BUT HERE IT IS ANYWAY
    # Correction to keep dipole magnitude normalized, or small integration 
    # build up. This is only useful for the spcific implementation
    # of electrostatically spinning beads
    ptot = np.sqrt(np.sum(xi_new[:3]**2))
    xi_new[:3] = (p0 / ptot) * xi_new[:3]

    return xi_new




@jit()
def stepper(xi_0, ti, tf, delt, upsamp, system, method):
    '''This function does the full integration. Takes a system function 
    (which is usually defined in the simulation script since it's subject
    to change), an integration method, an external efield to keep track 
    of the dipole's energy, and of course initial conditions.'''

    # Build the array of solution times
    tt = np.arange(ti, tf + delt * (1.0 + upsamp), delt)
    nt = len(tt)
    nt_ds = int(nt / upsamp)

    # Initialize list of 'points' which will contain the solution to our
    # ODE at each time in our discrete-time array
    points = np.zeros((6, nt_ds))
    out_time = np.zeros(nt_ds)
    xi_old = xi_0

    saveind = 0
    for tind, t in enumerate(tt):

        #start_t = time.time()
        # Using the 4th-order Runge-Kutta method, we evaluate the solution
        # iteratively for each time t in our discrete-time array
        xi_new = method(xi_old, t, delt, system)


        if not tind % upsamp:
            points[:,saveind] = xi_new
            out_time[saveind] = t
            saveind += 1

        #points[:,tind] = xi_new

        #energy = 0.5 * Ibead * np.sum(xi_new[3:]**2)
        #            (Ex * xi_new[0] + Ey * xi_new[1] + Ez * xi_new[2])

        #energy_vec[tind] = (energy)

        xi_old = xi_new

    #out_time -= (out_time[1] - out_time[0])

    return out_time, points #, energy_vec



def mc(ind):

    xi_0 = xi_init

    seed = seed_init * (ind + 1)
    np.random.seed(seed)

    base = '/data/old_trap_processed/spinsim_data/'
    base = os.path.join(base, savedir)
    base_filename = os.path.join(base, 'mc_{:d}/'.format(ind))

    bu.make_all_pardirs(os.path.join(base_filename, 'derp.txt'))

    for i in range(nfiles):
        #bu.progress_bar(i, nfiles)

        t0 = i*out_file_length
        tf = (i+1)*out_file_length

        tvec, soln = stepper(xi_0, t0, tf, dt_sim, upsamp, rhs, rk4)

        xi_0 = soln[:,-1]

        tvec = tvec[:-1]
        soln = soln[:,:-1]
        out_arr = np.concatenate( (tvec.reshape((1, len(tvec))), soln) )

        filename = os.path.join(base_filename, 'outdat_{:d}.npy'.format(i))
        np.save(open(filename, 'wb'), out_arr)

    return seed


start = time.time()
print('Starting to process data...')

seeds = Parallel(n_jobs=20)(delayed(mc)(ind) for ind in range(n_mc))
print(seeds)

stop = time.time()
print('Total troglodyte computation time: ', stop-start)


# start = time.time()
# tvec, soln = stepper(xi_init, 0.0, t_sim, dt_sim, upsamp, rhs, rk4)
# stop = time.time()
# print('Total troglodyte computation time: ', stop-start)

# base_filename = '/data/old_trap_processed/spinsim_data/spindowns/test3'
# bu.make_all_pardirs(os.path.join(base_filename, 'derp.txt'))
# out_arr = np.concatenate( (tvec.reshape((1, len(tvec))), soln) )
# for i in range(nfiles):
#     filename = os.path.join(base_filename, 'outdat_{:d}.npy'.format(i))
#     np.save(open(filename, 'wb'), out_arr[:,i*nsamp:(i+1)*nsamp])
#     print(out_arr[:,i*nsamp:(i+1)*nsamp].shape)

# plt.plot(tvec[inds], energy[inds])
# plt.figure()
# plt.plot(soln[0][inds])
# plt.plot(soln[1][inds])
# plt.plot(soln[2][inds])
# plt.figure()
# plt.plot(soln[3][inds])
# plt.plot(soln[4][inds])
# plt.plot(soln[5][inds])
# plt.show()

# start = time.time()
# soln = integrate.solve_ivp(rhs, (0, t_sim), xi_init, \
#                             t_eval=np.linspace(0, t_sim, nsamp+1))
# print soln.message
# stop = time.time()
# print 'Total solv_ivp computation time: ', stop-start


sys.stdout.flush()



