import os, sys, time, h5py

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.optimize as opti
import scipy.constants as constants

from obspy.signal.detrend import polynomial

import bead_util as bu

import dill as pickle

plt.rcParams.update({'font.size': 14})


base = '/data/old_trap_processed/spinsim_data/libration_tests/'

# dirname = os.path.join(base, 'fine_amp_sweep_short')
# dirname = os.path.join(base, 'high_pressure_sweep')
dirname = os.path.join(base, 'initial_angle_highp')

n_mc = bu.count_subdirectories(dirname)

hdf5 = True

fsamp = 1.0e6


### Constants
dipole_units = constants.e * (1e-6) # to convert e um -> C m
T = 297
m0 = 18.0 * constants.atomic_mass  # residual gas particle mass, in kg

### Bead-specific constants
p0 = 100.0 * dipole_units  #  C * m
# rhobead = {'val': 1850.0, 'sterr': 1.0, 'syserr': 1.0}
mbead_dic = {'val': 84.3e-15, 'sterr': 1.0e-15, 'syserr': 1.5e-15}
mbead = mbead_dic['val']
Ibead = bu.get_Ibead(mbead=mbead_dic)['val']
kappa = bu.get_kappa(mbead=mbead_dic)['val']

############################################################################
############################################################################
############################################################################
############################################################################


amps = []
pressures = []
lib_freqs = []

for i in range(n_mc):
    cdir = os.path.join(dirname, 'mc_{:d}'.format(i))

    param_path = os.path.join(cdir, 'params.p')
    params = pickle.load( open(param_path, 'rb') )

    pressure = params['pressure']   # convert back to mbar
    drive_amp = params['drive_amp']
    fsig = params['drive_freq']

    beta_rot = pressure * np.sqrt(m0) / kappa
    gamma = beta_rot / Ibead

    pressures.append(pressure)
    amps.append(drive_amp)

    if hdf5:
        ext = '.h5'
    else:
        ext = '.npy'

    datfiles, lengths = bu.find_all_fnames(cdir, ext=ext)
    nfiles = lengths[0]

    # gammas = []
    longdat = []
    nsamp = 0
    lib_freqs.append([])

    for fileind, file in enumerate(datfiles[::-1]):
        if fileind > 5:
            break

        bu.progress_bar(fileind, nfiles, suffix='{:d}/{:d}'.format(i+1, n_mc))

        if hdf5:
            fobj = h5py.File(file, 'r')
            dat = np.copy(fobj['sim_data'])
            fobj.close()
        else:
            dat = np.load(file)

        nsamp += dat.shape[1]

        tvec = dat[0]
        px = dat[1]

        crossp = np.abs(px)

        if not len(longdat):
            longdat = crossp
        else:
            longdat = np.concatenate( (longdat, crossp) )

    carrier_amp, carrier_phase \
            = bu.demod(longdat, fsig, fsamp, harmind=2.0, filt=True, \
                       bandwidth=5000.0, plot=False)

    params, cov = bu.fit_damped_osc_amp(carrier_phase, fsamp, plot=False, \
                                        fit_band=200)

    label1 = '$\\gamma = {:0.2f}$ Hz'.format(params[2])
    label2 = 'Expected: $\\gamma = {:0.2f}$ Hz'.format(gamma / (2.0 * np.pi))

    freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)
    asd = np.abs(np.fft.rfft(carrier_phase)) * bu.fft_norm(nsamp, fsamp)
    plt.loglog(freqs, asd)
    plt.loglog(freqs, bu.damped_osc_amp(freqs, *params), \
                label=label1)
    plt.loglog([], color='w', label=label2)
    plt.legend()

    plt.xlim(20, 2000)
    plt.ylim(1e-6, 0.12)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase ASD [rad/$\\sqrt{\mathrm{Hz}}$]')
    plt.tight_layout()

    plt.show()

    input()

        # lib_freqs[-1].append(params[1])
        # gammas.append(np.abs(params[2]))







### Routine to make the libfreq vs efield plot (need to move somewhere else)

# amps = np.array(amps)
# lib_freqs = np.mean(lib_freqs, axis=-1)

# def fit_fun(x, A):
#     return A * np.sqrt(x)

# popt, pcov = opti.curve_fit(fit_fun, amps, lib_freqs)

# plot_x = np.linspace(0, np.max(amps), 200)

# expected_libration = np.sqrt(plot_x * p0 / Ibead) / (2.0 * np.pi)



# plt.plot(amps * 1e-3, lib_freqs, 'o', ms=8, label='$\\omega_0 / 2 \\pi$ from simulation output')
# plt.plot(plot_x * 1e-3, expected_libration, ls='--', lw=2, color='r', \
#          label='Expected value: $( \\sqrt{E d / \\, I} \\,) / \\, 2 \\pi$')
# plt.xlabel('Efield Amplitude [kV/m]')
# plt.ylabel('Libration Frequency [Hz]')
# plt.legend()
# plt.tight_layout()

# plt.show()


# print()
# print('Pressure [mbar] : {:0.3g}'.format(pressure))
# print('   Damping [Hz] : {:0.3g}'.format(np.mean(gammas_hz)))
# sys.stdout.flush()