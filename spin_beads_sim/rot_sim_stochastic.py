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

TEST = False

ncore = 10
# ncore = 1

### Time to thermalize
# user_t_therm = 0.0
user_t_therm = 50.0
variable_thermalization = False

### Time to simulate
# t_sim = 1.5
t_sim = 50.0
# t_sim = 200.0

out_file_length = 2.0
user_nthermfiles = int(user_t_therm / out_file_length)
nfiles = int(t_sim / out_file_length)



### Sampling frequency consistent with what we actually sample on our
### experiment. Simulate with a timestep 100 times smaller than the final
### sampling frequency of interest to avoid numerical artifacts
fsamp = 500000.0
# fsamp = 2000000.0

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


# 
### Parameter lists
pressures = [1.0e-2]
# pressures = [2.0e-4, 5.0e-4, \
             # 1.0e-3, 2.0e-3, 5.0e-3, \
             # 1.0e-2, 2.0e-2, 5.0e-2]
pressures = 100.0 * np.array(pressures) # convert mbar to Pa

drive_freqs = [22500.0]
# drive_freqs = [5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0]

# drive_voltages = [400.0]
drive_voltages = np.linspace(50.0, 400.0, 8)

drive_voltage_noises = [0.0]
# drive_voltage_noises = [0.0, 5.0, 10.0, 15.0, 25.0, 50.0, 100.0]

drive_phase_noises = [0.0]
# drive_phase_noises = [0.0, np.pi/100, np.pi/50, np.pi/10, np.pi/4]

# initial_angles = [0.0]
initial_angles = [np.pi/2.0]

discretized_phases = [0.0]
# discretized_phases = [0.0, np.pi/100, np.pi/50, np.pi/10, np.pi/4]

fterm_noise = False
gterm_noise = False



# seed_init = 123456
seed_init = 654321


### Save path below
# savedir = 'libration_tests/phase_discretization'
# savedir = 'libration_tests/phase_noise_test_gterm'
# savedir = 'libration_tests/amp_noise_test_fterm_hp'
# savedir = 'libration_tests/3d_thermalization'
# savedir = 'libration_tests/rot_freq_sweep_hp'
savedir = 'libration_tests/amp_sweep_hp'
# savedir = 'libration_tests/pressure_sweep_hp'
# savedir = 'libration_tests/sdeint_fieldoff_manyp'
# savedir = 'libration_tests/sdeint_amp-sweep'
# savedir = 'libration_tests/sdeint_concat_test'

base = '/data/spin_sim_data/'
base = os.path.join(base, savedir)


#########################################################################
#########################################################################

iterproduct = itertools.product(pressures, drive_freqs, drive_voltages, \
                                drive_voltage_noises, drive_phase_noises, \
                                initial_angles, discretized_phases)

ind = 0
param_list = []
for v1, v2, v3, v4, v5, v6, v7 in iterproduct:
    param_list.append([ind, v1, v2, v3, v4, v5, v6, v7])
    ind += 1


def run_mc(params):

    ind = params[0]
    pressure = params[1]
    drive_freq = params[2]
    drive_voltage = params[3]
    drive_voltage_noise = params[4]
    drive_phase_noise = params[5]
    init_angle = params[6]
    discretized_phase = params[7]

    beta_rot = pressure * np.sqrt(m0) / kappa
    drive_amp = np.abs(bu.trap_efield([0, 0, 0, drive_voltage, -1.0*drive_voltage, \
                                       0, 0, 0], nsamp=1)[0])
    drive_amp_noise = drive_voltage_noise * (drive_amp / drive_voltage)

    seed = seed_init * (ind + 1)

    xi_0 = np.array([np.pi/2.0, 0.0, 0.0, \
                     0.0, 2.0*np.pi*drive_freq, 0.0])

    time_constant = Ibead / beta_rot

    np.random.seed(seed)

    ### If desired, set a thermalization time equal to 10x the time constant
    ### for this particular pressure and Ibead combination
    if variable_thermalization:
        t_therm = np.min([10.0 * time_constant, 300.0])
        nthermfiles = int(t_therm / out_file_length) + 1
    else:
        t_therm = user_t_therm
        nthermfiles = user_nthermfiles

    values_to_save = {}
    values_to_save['mbead'] = mbead
    values_to_save['Ibead'] = Ibead
    values_to_save['kappa'] = kappa
    values_to_save['beta_rot'] = beta_rot
    values_to_save['p0'] = p0
    values_to_save['fsamp'] = fsamp
    values_to_save['fsim'] = fsim
    values_to_save['seed'] = seed
    values_to_save['xi_0'] = xi_0
    values_to_save['init_angle'] = init_angle
    values_to_save['pressure'] = pressure
    values_to_save['m0'] = m0
    values_to_save['drive_freq'] = drive_freq
    values_to_save['drive_amp'] = drive_amp
    values_to_save['drive_amp_noise'] = drive_amp_noise
    values_to_save['drive_phase_noise'] = drive_phase_noise
    values_to_save['discretized_phase'] = discretized_phase
    values_to_save['t_therm'] = t_therm

    if not TEST:
        base_filename = os.path.join(base, 'mc_{:d}/'.format(ind))

        bu.make_all_pardirs(os.path.join(base_filename, 'derp.txt'))

        param_path = os.path.join(base_filename, 'params.p')
        pickle.dump(values_to_save, open(param_path, 'wb'))


    def E_phi_func(t, t_therm=0.0, init_angle=0.0):
        raw_val = 2.0 * np.pi * drive_freq * (t + t_therm) + init_angle
        if discretized_phase:
            n_disc = int(raw_val / discretized_phase)
            return n_disc * discretized_phase
        else:
            return raw_val

    ### Matrix for the stochastic driving processes
    torque_noise = np.sqrt(4.0 * kb * T * beta_rot)
    # B = np.array([[0, 0,   0,   0],
    #               [0, 0,   0,   0],
    #               [0, 0, 1.0,   0],
    #               [0, 0,   0, 1.0]])
    B = np.array([[0, 0, 0,   0,   0,   0],
                  [0, 0, 0,   0,   0,   0],
                  [0, 0, 0,   0,   0,   0],
                  [0, 0, 0, 1.0,   0,   0],
                  [0, 0, 0,   0, 1.0,   0],
                  [0, 0, 0,   0,   0, 1.0]])
    B *= torque_noise / Ibead

    ### Define the system such that d(xi) = f(xi, t) * dt
    # @jit()
    def f(x, t):
        torque_theta = drive_amp * p0 * np.sin(0.5 * np.pi - x[0]) \
                            - 1.0 * beta_rot * x[3]

        c_amp = drive_amp
        E_phi = E_phi_func(t)
        if fterm_noise:
            c_amp += drive_amp_noise * np.random.randn()
            E_phi += drive_phase_noise * np.random.randn()

        torque_phi = c_amp * p0 * np.sin(E_phi - x[1]) * np.sin(x[0]) \
                            - 1.0 * beta_rot * x[4]

        torque_psi = -1.0 * beta_rot * x[5]

        return np.array([x[3], x[4], x[5], \
                         torque_theta / Ibead, \
                         torque_phi / Ibead, \
                         torque_psi / Ibead])

    ### Define the stochastic portion of the system
    # @jit()
    def G(x, t):
        newB = np.zeros((6,6))

        if gterm_noise:
            E_phi = E_phi_func(t)
            amp_noise_term = drive_amp_noise * p0 * np.sin(E_phi - x[1]) * np.sin(x[0])

            E_phi_rand = drive_phase_noise * np.random.randn()
            phase_noise_term = drive_amp * p0  * np.sin(E_phi_rand) * np.sin(x[0])
            newB[4,4] += amp_noise_term + phase_noise_term

        return B + newB


    ### Thermalize
    xi_init = np.copy(xi_0)
    for i in range(nthermfiles):
        t0 = i*out_file_length
        tf = (i+1)*out_file_length

        nsim = int(out_file_length * fsim)
        tvec = np.linspace(t0, tf, nsim+1)

        result = sdeint.itoint(f, G, xi_init, tvec).T
        xi_init = np.copy(result[:,-1])


    ### Redefine the system taking into account the thermalization time
    ### and the desired phase offset
    # @jit()
    def f(x, t):
        torque_theta = drive_amp * p0 * np.sin(0.5 * np.pi - x[0]) \
                            - 1.0 * beta_rot * x[3]

        c_amp = drive_amp
        E_phi = E_phi_func(t, t_therm=t_therm, init_angle=init_angle)
        if fterm_noise:
            c_amp += drive_amp_noise * np.random.randn()
            E_phi += drive_phase_noise * np.random.randn()

        torque_phi = c_amp * p0 * np.sin(E_phi - x[1]) * np.sin(x[0]) \
                            - 1.0 * beta_rot * x[4]

        torque_psi = -1.0 * beta_rot * x[5]

        return np.array([x[3], x[4], x[5], \
                         torque_theta / Ibead, \
                         torque_phi / Ibead, \
                         torque_psi / Ibead])


    # @jit()
    # def f(x, t):
    #     torque_theta = - 1.0 * beta_rot * x[2]
    #     torque_phi = - 1.0 * beta_rot * x[3]

    #     return np.array([x[2], x[3], torque_theta / Ibead, torque_phi / Ibead])

    ### Define the stochastic portion of the system
    def G(x, t):
        newB = np.zeros((6,6))
        if gterm_noise:
            E_phi = E_phi_func(t, t_therm=t_therm, init_angle=init_angle)
            amp_noise_term = drive_amp_noise * p0 * np.sin(E_phi - x[1]) * np.sin(x[0])

            E_phi_rand = drive_phase_noise * np.random.randn()
            phase_noise_term = drive_amp * p0  * np.sin(E_phi_rand) * np.sin(x[0])

        newB[4,4] += amp_noise_term + phase_noise_term

        return B + newB




    ### Run the simulation with the thermalized solution
    for i in range(nfiles):
        # start = time.time()
        t0 = i*out_file_length
        tf = (i+1)*out_file_length

        nsim = int(out_file_length * fsim)
        tvec = np.linspace(t0, tf, nsim+1)

        ### Solve!
        # print('RUNNING SIM')
        result = sdeint.itoint(f, G, xi_init, tvec).T
        xi_init = np.copy(result[:,-1])

        tvec = tvec[:-1]
        soln = result[:,:-1]

        # print('DOWNSAMPLING')
        nsamp = int(out_file_length * fsamp)
        # soln_ds, tvec_ds = signal.resample(soln, t=tvec, \
        #                                    num=nsamp, axis=-1)
        # soln_ds = signal.decimate(soln, int(upsamp))

        tvec_ds = tvec[::int(upsamp)]
        soln_ds = soln[:,::int(upsamp)]

        # plt.plot(tvec, soln[1])
        # plt.plot(tvec_ds, soln_ds[1])
        # plt.plot(tvec_ds, soln_ds_2[1])

        # plt.show()

        if not TEST:
            out_arr = np.concatenate( (tvec_ds.reshape((1, len(tvec_ds))), soln_ds) )

            filename = os.path.join(base_filename, 'outdat_{:d}.h5'.format(i)) 

            fobj = h5py.File(filename, 'w')
            fobj.create_dataset('sim_data', data=out_arr, compression='gzip', \
                                compression_opts=9)
            fobj.close()

        # stop = time.time()
        # print('Time for one file: {:0.1f}'.format(stop-start))

    return seed

start = time.time()
print('Starting to process data...')

seeds = Parallel(n_jobs=ncore)(delayed(run_mc)(params) for params in param_list)
print(seeds)

stop = time.time()
print('Total troglodyte computation time: {:0.1f}'.format(stop-start))



sys.stdout.flush()
