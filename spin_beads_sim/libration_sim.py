
import sys, time, os, itertools, h5py
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

ncore = 20


### Constants
kb = constants.Boltzmann
dipole_units = constants.e * (1e-6)


### Environmental constants
T = 297
m0 = 18.0 * constants.atomic_mass  # residual gas particle mass, in kg


### Bead-specific constants
p0 = 100.0 #* dipole_units  #  C * m
### to avoid numerical errors, dipole moment is integrated in units of e * micron
### although anytime a torque appears, the value is cast to the correct SI units


# rhobead = {'val': 1850.0, 'sterr': 1.0, 'syserr': 1.0}
mbead_dic = {'val': 84.3e-15, 'sterr': 1.0e-15, 'syserr': 1.5e-15}
mbead = mbead_dic['val']
Ibead = bu.get_Ibead(mbead=mbead_dic)['val']
kappa = bu.get_kappa(mbead=mbead_dic)['val']


### Simulation parameters
t_sim = 300.0
# t_sim = 0.1

out_file_length = 2.0
# out_file_length = 0.1
nfiles = int(t_sim / out_file_length)

fsamp = 1.0e6
nsamp = int(out_file_length * fsamp)

upsamp = 10.0
fsim = upsamp * fsamp
dt_sim = 1.0 / fsim
nsim = int(t_sim * fsim)

#N_opt = 2.0 * np.pi * (6000.0) * beta_rot
N_opt = 0

### Pressures, converted to Pascals
pressures = [1.0e-3 * 100]
# pressures = [#1.0e-7 * 100.0, 3.5e-7 * 100.0, \
#              1.0e-6 * 100.0, 3.5e-6 * 100.0, \
#              1.0e-5 * 100.0, 3.5e-5 * 100.0, \
#              1.0e-4 * 100.0, 3.5e-4 * 100.0, \
#              1.0e-3 * 100.0, 3.5e-3 * 100.0, \
#              1.0e-2 * 100.0, 3.5e-2 * 100.0, \
#             ]

drive_freqs = [25000.0]
# drive_freqs = [10000.0, 25000.0, 50000.0, 75000.0, 100000.0]

drive_voltages = [400.0]
# drive_voltages = [50.0, 100.0, 400.0]
# drive_voltages = np.linspace(10, 400, 30)

drive_voltage_noises = [0.0]
# drive_voltage_noises =  [0.0, 0.001, 0.01, 0.1, 0.5]

drive_phase_noises = [0.0]
# drive_phase_noises = [0.0, 0.01 * np.pi, 0.1 * np.pi]

# init_angles = [0.0]
init_angles = np.linspace(0, np.pi/2, 20)

repeats = 1
# repeats = 10

# seed_init = 321654 # for fine amp sweep
# seed_init = 654321 # for amp/amp-noise sweep
# seed_init = 123456 # for pressure sweep
seed_init = 133769 # for angle sweep


### Save path below
# savedir = 'libration_tests/fine_amp_sweep_short'
# savedir = 'libration_tests/high_pressure_sweep'
savedir = 'libration_tests/initial_angle_highp'
# savedir = 'derp_test'

base = '/data/old_trap_processed/spinsim_data/'
base = os.path.join(base, savedir)





############################################################################
############################################################################
############################################################################
############################################################################


### Build a list of all possible parameter combinations
iterproduct = itertools.product(pressures, drive_freqs, drive_voltages, \
                                drive_voltage_noises, drive_phase_noises, \
                                init_angles)

### Add a unique index to each entry in the parameter list to ensure 
### a unique seed for each iteration of the simulation
ind = 0
param_list = []
for repeat in range(repeats):
    for v1, v2, v3, v4, v5, v6 in iterproduct:
        param_list.append([ind, v1, v2, v3, v4, v5, v6])
        ind += 1



@jit()
def rk4(xi_old, t, delt, system, system_stochastic=None):
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

    if system_stochastic is not None:
        xi_new += delt * system_stochastic(t, xi_old)

    ### HARDCODED STUFF IS BAD BUT HERE IT IS ANYWAY
    # Correction to keep dipole magnitude normalized, or small integration 
    # build up. This is only useful for the spcific implementation
    # of electrostatically spinning beads
    ptot = np.sqrt(np.sum(xi_new[:3]**2))
    xi_new[:3] = (p0 / ptot) * xi_new[:3]

    return xi_new




def stepper(xi_0, ti, tf, delt, upsamp, method, system, system_stochastic=None):
    '''This function does the full integration. Takes a system function 
    (which is usually defined in the simulation script since it's subject
    to change), an integration method, an external efield to keep track 
    of the dipole's energy, and of course initial conditions.'''

    ### Build the array of solution times
    nt = (tf - ti) / delt
    tt = np.linspace(ti, tf, int(nt + 1))

    nt_ds = int(nt / upsamp)

    ### Initialize list of 'points' which will contain the solution to our
    ### ODE at each time in our discrete-time array
    xi_old = xi_0
    points = np.zeros((len(xi_0), nt_ds + 1))
    out_time = np.zeros(nt_ds + 1)

    saveind = 0
    for tind, t in enumerate(tt):
        # bu.progress_bar(tind, nt)

        if tind == 0:
            points[:,0] = xi_0
            out_time[0] = ti
            saveind += 1
            continue

        #start_t = time.time()
        ### Using the 4th-order Runge-Kutta method, we evaluate the solution
        ### iteratively for each time t in our discrete-time array
        xi_new = method(xi_old, t, delt, system, \
                        system_stochastic=system_stochastic)

        if not tind % upsamp:
            points[:,saveind] = xi_new
            out_time[saveind] = t
            saveind += 1

        xi_old = xi_new

    return out_time, points 






def run_mc(params):

    ind                  = params[0]
    pressure             = params[1]
    drive_freq           = params[2]
    drive_voltage        = params[3]
    drive_voltage_noise  = params[4]
    drive_phase_noise    = params[5]
    init_angle           = params[6]

    beta_rot = pressure * np.sqrt(m0) / kappa
    drive_amp = np.abs(bu.trap_efield([0, 0, 0, drive_voltage, -1.0*drive_voltage, \
                                        0, 0, 0], nsamp=1)[0])
    drive_amp_noise = drive_voltage_noise * (drive_amp / drive_voltage)

    lib_freq = np.sqrt(drive_amp * p0 * dipole_units / Ibead) / (2.0 * np.pi)

    xi_0 = np.array([p0*np.cos(init_angle), p0*np.sin(init_angle), 0.0, \
                        0.0, 0.0, 2.0 * np.pi * drive_freq])

    seed = seed_init * (ind + 1)

    np.random.seed(seed)

    values_to_save = {}
    values_to_save['mbead'] = mbead
    values_to_save['Ibead'] = Ibead
    values_to_save['p0'] = p0
    values_to_save['fsamp'] = fsamp
    values_to_save['seed'] = seed
    values_to_save['xi_0'] = xi_0
    values_to_save['pressure'] = pressure
    values_to_save['drive_freq'] = drive_freq
    values_to_save['drive_amp'] =  drive_amp
    values_to_save['drive_amp_noise'] =  drive_amp_noise
    values_to_save['drive_phase_noise'] =  drive_phase_noise

    base_filename = os.path.join(base, 'mc_{:d}/'.format(ind))

    bu.make_all_pardirs(os.path.join(base_filename, 'derp.txt'))

    param_path = os.path.join(base_filename, 'params.p')
    pickle.dump(values_to_save, open(param_path, 'wb') )



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
        drag_torque = -1.0 * beta_rot * xi[3:]

        #### Construct the rotating Efield drive
        Efield = np.array([drive_amp * np.cos(2.0 * np.pi * drive_freq * t), \
                           drive_amp * np.sin(2.0 * np.pi * drive_freq * t), \
                           0.0])

        drive_torque = np.cross(xi[:3]*dipole_units, Efield)
        optical_torque = np.array([0.0, 0.0, N_opt])

        total_torque = drive_torque + drag_torque + optical_torque 

        return np.concatenate( (-1.0*np.cross(xi[:3], xi[3:]), total_torque / Ibead) )


    @jit()
    def rhs_stochastic(t, xi):
        '''Basically the same as above rhs() function, but this only includes the 
           stochastic forcing terms. Doesn't update the dipole moment projections,
           just adds more (Delta omega)

           The function computes the following torques:
                thermal torque, white noise with power computed from above global
                                    parameters and fluctuation dissipation theorem 
                drive torque, computed as (-1.0) * {px, py, pz} (cross) {Ex, Ey, Ez}
                                where the Efield only includes noise terms
        '''
        thermal_torque = np.sqrt(4.0 * kb * T * beta_rot * fsim) * np.random.randn(3)

        ### Amplitude noise for all three axes
        an = drive_amp_noise * np.random.randn(3)

        ### Phase noise for the two drive axes
        pn = drive_phase_noise * np.random.randn(2)

        #### Construct the rotating Efield drive
        Efield1 = np.array([drive_amp * np.cos(2.0 * np.pi * drive_freq * t), \
                            drive_amp * np.sin(2.0 * np.pi * drive_freq * t), \
                            0.0])
        Efield2 = np.array([drive_amp * np.cos(2.0 * np.pi * drive_freq * t + pn[0]), \
                            drive_amp * np.sin(2.0 * np.pi * drive_freq * t + pn[1]), \
                            0.0])
        Efield = Efield2 - Efield1 + an

        drive_torque = np.cross(xi[:3]*dipole_units, Efield)

        total_torque = drive_torque + thermal_torque

        return np.concatenate( (np.zeros(3), total_torque / Ibead) )



    for i in range(nfiles):
        #bu.progress_bar(i, nfiles)

        t0 = i*out_file_length
        tf = (i+1)*out_file_length

        tvec, soln = stepper(xi_0, t0, tf, dt_sim, upsamp, rk4, rhs, \
                             system_stochastic=rhs_stochastic)

        xi_0 = soln[:,-1]

        tvec = tvec[:-1]
        soln = soln[:,:-1]
        out_arr = np.concatenate( (tvec.reshape((1, len(tvec))), soln) )

        filename = os.path.join(base_filename, 'outdat_{:d}.h5'.format(i))
        fobj = h5py.File(filename, 'w')
        # group = fobj.create_group('sim_data')
        fobj.create_dataset('sim_data', data=out_arr, compression='gzip', \
                            compression_opts=9)
        fobj.close()

        # filename = os.path.join(base_filename, 'outdat_{:d}.npy'.format(i))
        # np.save(open(filename, 'wb'), out_arr)

    return seed



start = time.time()
print('Starting to process data...')

seeds = Parallel(n_jobs=ncore)(delayed(run_mc)(params) for params in param_list)
print(seeds)

stop = time.time()
print('Total troglodyte computation time: ', stop-start)



sys.stdout.flush()



