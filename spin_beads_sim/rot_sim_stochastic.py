import sys, time, os, itertools, h5py
import numpy as np
import dill as pickle

import matplotlib.pyplot as plt

import scipy.constants as constants
import scipy.signal as signal

from joblib import Parallel, delayed
from numba import jit

import bead_util as bu

import sdeint

ncore = 10

### Time to simulate
t_sim = 100.0

out_file_length = 2.0
nfiles = int(t_sim / out_file_length)


### Sampling frequency consistent with what we actually sample on our
### experiment. Simulate with a timestep 100 times smaller than the final
### sampling frequency of interest to avoid numerical artifacts
fsamp = 500000.0

upsamp = 1.0
fsim = upsamp * fsamp

### Build the array of times at which we'd like our solution
dt_sim = 1.0 / fsim


### Some physical constants associated to our system which determine
### the magnitude of the thermal driving force
m0 = 18.0 * constants.atomic_mass  ### residual gas is primarily water
kb = constants.Boltzmann
T = 297.0


### Properties of the microsphere (in SI mks units) to construct the 
### damping coefficient and other things
mbead_dic = {'val': 84.3e-15, 'sterr': 1.0e-15, 'syserr': 1.5e-15}
mbead = mbead_dic['val']
Ibead = bu.get_Ibead(mbead=mbead_dic)['val']
kappa = bu.get_kappa(mbead=mbead_dic)['val']

p0 = 100.0 * constants.e * (1e-6)



### Parameter lists
pressures = [1.0e-4]
# pressures = [1.0e-5, 2.0e-5, 5.0e-5, \
#              1.0e-4, 2.0e-4, 5.0e-4, \
#              1.0e-3, 2.0e-3, 5.0e-3, \
#              1.0e-2]
pressures = 100.0 * np.array(pressures) 

drive_freqs = [22500.0]

# drive_voltages = [400.0]
drive_voltages = np.linspace(50.0, 400.0, 10)

drive_voltage_noises = [0.0]

drive_phase_noises = [0.0]

# initial_angles = [np.pi/2.0]
initial_angles = [0.0]




seed_init = 123456


### Save path below
# savedir = 'libration_tests/sdeint_ringdown_manyp'
savedir = 'libration_tests/sdeint_amp-sweep'

base = '/data/old_trap_processed/spinsim_data/'
base = os.path.join(base, savedir)


#########################################################################
#########################################################################

iterproduct = itertools.product(pressures, drive_freqs, drive_voltages, \
                                drive_voltage_noises, drive_phase_noises, \
                                initial_angles)

ind = 0
param_list = []
for v1, v2, v3, v4, v5, v6 in iterproduct:
    param_list.append([ind, v1, v2, v3, v4, v5, v6])
    ind += 1


def run_mc(params):

    ind = params[0]
    pressure = params[1]
    drive_freq = params[2]
    drive_voltage = params[3]
    drive_voltage_noise = params[4]
    drive_phase_noise = params[5]
    init_angle = params[6]

    beta_rot = pressure * np.sqrt(m0) / kappa
    drive_amp = np.abs(bu.trap_efield([0, 0, 0, drive_voltage, -1.0*drive_voltage, \
                                       0, 0, 0], nsamp=1)[0])
    drive_amp_noise = drive_voltage_noise * (drive_amp / drive_voltage)

    xi_0 = np.array([np.pi/2.0, init_angle, 0.0, 2.0*np.pi*drive_freq])

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
    values_to_save['drive_amp'] = drive_amp
    values_to_save['drive_amp_noise'] = drive_amp_noise
    values_to_save['drive_phase_noise'] = drive_phase_noise

    base_filename = os.path.join(base, 'mc_{:d}/'.format(ind))

    bu.make_all_pardirs(os.path.join(base_filename, 'derp.txt'))

    param_path = os.path.join(base_filename, 'params.p')
    pickle.dump(values_to_save, open(param_path, 'wb'))

    torque_noise = np.sqrt(4.0 * kb * T * beta_rot)

    B = np.array([[0, 0,   0,   0],
                  [0, 0,   0,   0],
                  [0, 0, 1.0,   0],
                  [0, 0,   0, 1.0]])
    B *= torque_noise / Ibead

    @jit()
    def f(x, t):
        torque_theta = drive_amp * p0 * np.sin(0.5 * np.pi - x[0]) \
                            - 1.0 * beta_rot * x[2]

        E_phi = 2.0 * np.pi * drive_freq * t
        torque_phi = drive_amp * p0 * np.sin(E_phi - x[1]) * np.sin(x[0]) \
                            - 1.0 * beta_rot * x[3]

        return np.array([x[2], x[3], torque_theta / Ibead, torque_phi / Ibead])

    @jit()
    def G(x, t):
        return B


    for i in range(nfiles):
        # start = time.time()
        t0 = i*out_file_length
        tf = (i+1)*out_file_length

        nsamp = int(out_file_length * fsim)
        tvec = np.linspace(t0, tf, nsamp+1)

        ### Solve!
        result = sdeint.itoint(f, G, xi_0, tvec).T

        xi_0 = result[:,-1]

        tvec = tvec[:-1]
        soln = result[:,:-1]

        out_arr = np.concatenate( (tvec.reshape((1, len(tvec))), soln) )

        filename = os.path.join(base_filename, 'outdat_{:d}.h5'.format(i)) 

        fobj = h5py.File(filename, 'w')
        fobj.create_dataset('sim_data', data=out_arr, compression='gzip', \
                            compression_opts=9)
        fobj.close()
        # stop = time.time()
        # print('Time for one file: {:0.1f}'.format(stop-start))

    return seeds

start = time.time()
print('Starting to process data...')

seeds = Parallel(n_jobs=ncore)(delayed(run_mc)(params) for params in param_list)
print(seeds)

stop = time.time()
print('Total troglodyte computation time: {:0.1f}'.format(stop-start))



sys.stdout.flush()
