import os, sys, time, h5py

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.optimize as opti
import scipy.constants as constants

from obspy.signal.detrend import polynomial

import bead_util as bu

import dill as pickle

from joblib import Parallel, delayed

ncore = 10
# ncore = 1

plt.rcParams.update({'font.size': 14})


base = '/data/spin_sim_data/libration_tests/'

# dirname = os.path.join(base, 'high_pressure_sweep')
# dirname = os.path.join(base, 'sdeint_ringdown_manyp_3')
dirname = os.path.join(base, 'amp_noise_test_fterm')
# dirname = os.path.join(base, 'rot_freq_sweep')
n_mc = bu.count_subdirectories(dirname)

hdf5 = True
ext = '.h5'

### Paths for saving
save_base = '/home/cblakemore/opt_lev_analysis/spin_beads_sim/processed_results/'
# save_filename = os.path.join(save_base, 'rot_freq_sweep.p')
save_filename = os.path.join(save_base, 'amp_noise_test_fterm.p')

average = False
concatenate = True
# nspectra_to_combine = 15
nspectra_to_combine = 10
plot_raw_data = False

ncycle_pad = 0.0

downsample = True
downsample_fac = 100

### Use this option with care: if you parallelize and ask it to plot,
### you'll get ncore * (a few) plots up simultaneously
plot_demod = False

### Constants
dipole_units = constants.e * (1e-6) # to convert e um -> C m

### Bead-specific constants
p0 = 100.0 * dipole_units  #  C * m

### Some physical constants associated to our system which determine
### the magnitude of the thermal driving force
m0 = 18.0 * constants.atomic_mass  ### residual gas is primarily water
kb = constants.Boltzmann
T = 297.0

### Properties of the microsphere (in SI mks units) to construct the 
### damping coefficient and other things
mbead_dic = {'val': 84.3e-15, 'sterr': 1.0e-15, 'syserr': 1.5e-15}
mbead = mbead_dic['val']
# Ibead = bu.get_Ibead(mbead=mbead_dic)['val']
# kappa = bu.get_kappa(mbead=mbead_dic)['val']


############################################################################
############################################################################
############################################################################
############################################################################

colors = bu.get_colormap(n_mc, cmap='plasma')[::-1]


def proc_mc(i):
    ### Build the path name, assuming the monte-carlo's are 
    ### zero-indexed
    cdir = os.path.join(dirname, 'mc_{:d}'.format(i))

    ### Load the simulation parameters saved alongside the data
    param_path = os.path.join(cdir, 'params.p')
    params = pickle.load( open(param_path, 'rb') )

    ### Define some values that are necessary for analysis
    pressure = params['pressure']
    drive_amp = params['drive_amp']
    fsig = params['drive_freq']
    drive_freq = params['drive_freq']
    p0 = params['p0']
    Ibead = params['Ibead']
    kappa = params['kappa']
    fsamp = params['fsamp']
    fsamp_ds = fsamp

    try:
        t_therm = params['t_therm']
    except:
        t_therm = 0.0

    try:
        init_angle = params['init_angle']
    except Exception:
        init_angle = np.pi / 2.0

    beta_rot = pressure * np.sqrt(m0) / kappa
    phieq = -1.0 * np.arcsin(2.0 * np.pi * drive_freq * beta_rot / (drive_amp * p0))

    time_constant = Ibead / beta_rot
    gamma_calc = 1.0 / time_constant

    # print(pressure, time_constant, t_therm)

    ### Load the data
    datfiles, lengths = bu.find_all_fnames(cdir, ext=ext, verbose=False, \
                                            sort_time=True, use_origin_timestamp=True)

    ### Invert the data file array so that the last files are processed first
    ### since the last files should be the most thermalized
    datfiles = datfiles[::-1]
    nfiles = lengths[0]

    ### Depeding on the requested behavior, instantiate some arrays
    if concatenate:
        long_t = []
        long_sig = []
        long_sig_2 = []
    if average:
        psd_array = []

    ### Loop over the datafiles 
    for fileind, file in enumerate(datfiles):
        # print(file)
        ### Break the loop if we've acquired enough data
        if fileind > nspectra_to_combine - 1:
            break

        ### Load the data, taking into account the file type
        if hdf5:
            fobj = h5py.File(file, 'r')
            dat = np.copy(fobj['sim_data'])
            fobj.close()
        else:
            dat = np.load(file)

        ### Determine the length of the data
        nsamp = dat.shape[1]
        nsamp_ds = int(nsamp / downsample_fac)

        ### Load the angles and construct the x-component of the dipole
        ### based on the integrated angular positions
        tvec = dat[0]
        theta = dat[1]
        phi = dat[2]
        px = p0 * np.cos(phi) * np.sin(theta)

        E_phi = 2.0 * np.pi * drive_freq * (tvec + t_therm) + init_angle

        ones = np.ones(len(tvec))

        dipole = np.array([ones, theta, phi])
        efield = np.array([ones, ones*(np.pi/2), E_phi])
        lib_angle = bu.angle_between_vectors(dipole, efield, coord='s')

        ### construct an estimate of the cross-polarized light
        crossp = np.sin(phi)**2

        ### Normalize to avoid numerical errors. Try to get the max to sit at 10
        crossp *= (10.0 / np.max(np.abs(crossp)))

        ### Build up the long signal if concatenation is desired
        if concatenate:
            if not len(long_sig):
                long_t = tvec
                long_sig = crossp
                long_phi = phi
                long_lib = lib_angle
            else:
                long_t = np.concatenate((tvec, long_t))
                long_sig = np.concatenate((crossp, long_sig))
                long_phi = np.concatenate((phi, long_phi))
                long_lib = np.concatenate((lib_angle, long_lib ))

        ### Using a hilbert transform, demodulate the amplitude and phase of
        ### the carrier signal. Filter things if desired.
        carrier_amp, carrier_phase \
                = bu.demod(crossp, fsig, fsamp, harmind=2.0, filt=True, \
                           bandwidth=4000.0, plot=plot_demod, ncycle_pad=100, \
                           tukey=True, tukey_alpha=5e-4)

        # carrier_phase = phi - E_phi

        if plot_raw_data:
            phi_lab = '$\\phi$'
            theta_lab = '$\\theta - \\pi / 2$'
            lib_lab = '$\\left| \\measuredangle (\\vec{E}) (\\vec{d}) \\right|$'

            # plt.plot(carrier_phase)
            plt.plot(tvec, carrier_phase, alpha=1.0, label=phi_lab)
            plt.plot(tvec, theta - np.pi/2, alpha=0.7, label=theta_lab)
            plt.plot(tvec, lib_angle, alpha=0.7, label=lib_lab)
            plt.title('In Rotating Frame', fontsize=14)
            plt.xlabel('Time [s]')
            plt.ylabel('Angular Coordinate [rad]')
            plt.legend(loc='lower right', fontsize=12)
            plt.tight_layout()
            # plt.show()

            freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)
            norm = bu.fft_norm(nsamp, fsamp)
            plt.figure()
            plt.loglog(freqs, np.abs(np.fft.rfft(carrier_phase))*norm, label=phi_lab)
            plt.loglog(freqs, np.abs(np.fft.rfft(theta-np.pi/2))*norm, label=theta_lab)
            plt.loglog(freqs, np.abs(np.fft.rfft(lib_angle))*norm, label=lib_lab)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('ASD [rad / $\\sqrt{ \\rm Hz}$]')
            plt.legend(loc='lower right', fontsize=12)
            plt.xlim(330, 1730)
            plt.ylim(5e-7, 3e-2)
            plt.tight_layout()
            plt.show()

            input()

        ### Downsample the data if desired
        if downsample:
            ### Use scipy's fourier-based downsampling
            carrier_phase_ds, tvec_ds = signal.resample(carrier_phase, nsamp_ds, t=tvec, window=None)
            carrier_phase = np.copy(carrier_phase_ds)
            tvec = np.copy(tvec_ds)

            dt = tvec_ds[1] - tvec_ds[0]
            fsamp_ds = 1.0 / dt

            ### Compute the frequencies and ASD values of the downsampled signal
            freqs = np.fft.rfftfreq(nsamp_ds, d=dt)
            carrier_phase_asd = bu.fft_norm(nsamp_ds, fsamp_ds) * np.abs(np.fft.rfft(carrier_phase_ds))

        else:
            ### Compute the frequencies and ASD values
            freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)
            carrier_phase_asd = bu.fft_norm(nsamp, fsamp) * np.abs(np.fft.rfft(carrier_phase))

        ### Add the data from the current file to the array of PSDs
        if average:
            if not len(psd_array):
                psd_array = np.zeros((nspectra_to_combine, len(carrier_phase_asd)), dtype=np.float64)
            psd_array[fileind,:] += carrier_phase_asd**2

    ### Compute the mean and uncertainty of the PSDs, then compute the ASD
    if average:
        avg_psd = np.mean(psd_array, axis=0)
        fit_asd = np.sqrt(avg_psd)

        ### Use propogation of uncertainty and the standard error on the mean
        asd_errs = 0.5 * fit_asd * (np.std(psd_array, axis=0) \
                    * np.sqrt(1.0 / nspectra_to_combine)) / avg_psd

    ### If concatenation was desired, first demodulate the amplitude and phase of the
    ### carrier signal, and then downsample the carrier phase.
    if concatenate:
        ### Compute the new values of nsamp
        nsamp = len(long_sig)
        if downsample:
            nsamp_ds = int(nsamp / downsample_fac)
        else:
            nsamp_ds = nsamp

        ### Hilbert transform demodulation
        carrier_amp_long, carrier_phase_long \
                = bu.demod(long_sig, fsig, fsamp, harmind=2.0, filt=False, \
                           bandwidth=5000.0, plot=False, ncycle_pad=ncycle_pad, \
                           tukey=True, tukey_alpha=1e-4)

        carrier_phase_long = long_phi - 2.0 * np.pi * drive_freq * (long_t + t_therm) - init_angle

        ### Downsampling
        carrier_phase_ds, tvec_ds = signal.resample(carrier_phase_long, nsamp_ds, \
                                                    t=long_t, window=None)
        long_lib_ds, tvec_ds_2 = signal.resample(long_lib, nsamp_ds, \
                                                 t=long_t, window=None)

        ### Compute the ASD of the downsampled signals
        fit_asd = bu.fft_norm(nsamp_ds, fsamp_ds) * np.abs(np.fft.rfft(carrier_phase_ds))
        fit_asd_2 = bu.fft_norm(nsamp_ds, fsamp_ds) * np.abs(np.fft.rfft(long_lib_ds))
        asd_errs = []

        ### Compute the new frequency arrays
        freqs = np.fft.rfftfreq(nsamp_ds, d=1.0/fsamp_ds)

    ### Fit either the averaged ASD or the ASD of the concatenated signal
    params, cov = bu.fit_damped_osc_amp(fit_asd, fsamp_ds, plot=False, \
                                        sig_asd=True, linearize=True, \
                                        asd_errs=asd_errs, fit_band=[500.0,600.0], \
                                        gamma_guess=gamma_calc, \
                                        weight_lowf=True, weight_lowf_val=0.5, \
                                        weight_lowf_thresh=200)


    # plt.loglog(freqs, fit_asd_2)
    # plt.show()

    ### Fit either the averaged ASD or the ASD of the concatenated signal
    params_2, cov_2 = bu.fit_damped_osc_amp(fit_asd_2, fsamp_ds, plot=False, \
                                            sig_asd=True, linearize=True, \
                                            asd_errs=[], fit_band=[1050.0,1150.0], \
                                            gamma_guess=gamma_calc, freq_guess=2.0*params[1], \
                                            weight_lowf=True, weight_lowf_val=0.5, \
                                            weight_lowf_thresh=200)

    outdict = {'pressure': pressure, 'freqs': freqs, \
               'fit_asd': fit_asd, 'params': params, 'cov': cov, \
               'fit_asd_2': fit_asd_2, 'params_2': params_2, 'cov_2': cov_2, \
               'gamma_calc': gamma_calc, 'drive_freq': drive_freq}

    return outdict


### Analyze all the results in parallel
results = Parallel(n_jobs=ncore)( delayed(proc_mc)(ind) for ind in list(range(n_mc))[::1] )


### Loop over the results and plot each one
plt.figure(figsize=(12,6))
colors = bu.get_colormap(len(results), cmap='plasma')
pressures = []
gammas = [[], [], []]
for resultind, result in enumerate(results[::-1]):
    fac = 100.0**resultind
    print(fac)
    pressure = result['pressure']
    pressures.append(pressure)

    gamma_calc = result['gamma_calc']
    gamma_fit = result['params'][2]
    gamma_fit_2 = result['params_2'][2]

    label = '$ \\gamma = {:0.2g}$ Hz, [$\\gamma (p) = {:0.2g}$ Hz]'\
                .format(gamma_fit, gamma_calc / (2.0 * np.pi))
    # label = '$\\omega_0 = {:0.1f}$ Hz'.format(result[6])
    gammas[0].append(gamma_calc)
    gammas[1].append(gamma_fit)
    gammas[2].append(gamma_fit_2)

    freqs = result['freqs']
    fit_asd = result['fit_asd']
    plt.loglog(freqs, fit_asd*fac, color=colors[resultind], alpha=0.7, label=label)
    plt.loglog(freqs, bu.damped_osc_amp(freqs, *result['params'])*fac, \
               color=colors[resultind], lw=3)
plt.legend(fontsize=10, ncol=2, loc='upper left')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase ASD [arb]')
plt.xlim(1.0, 2500)
plt.ylim(1e-7, 3e12)
plt.tight_layout()

gammas = np.array(gammas)
plt.figure()
plt.plot(gammas[1] / gammas[0])
plt.plot(gammas[2] / gammas[0])

plt.show()

### Save the reslts for later analysis if desired
pickle.dump(results, open(save_filename, 'wb'))




