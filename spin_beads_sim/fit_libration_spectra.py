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

plt.rcParams.update({'font.size': 14})


base = '/data/spin_sim_data/libration_tests/'

# dirname = os.path.join(base, 'high_pressure_sweep')
dirname = os.path.join(base, 'sdeint_ringdown_manyp_3')
# dirname = os.path.join(base, 'sdeint_concat_test')
n_mc = bu.count_subdirectories(dirname)

hdf5 = True
ext = '.h5'

### Paths for saving
save_base = '/home/cblakemore/opt_lev_analysis/spin_beads_sim/processed_results/'
save_filename = os.path.join(save_base, 'libration_spectra_manyp_3.p')
# save_filename = os.path.join(save_base, 'libration_concat_test.p')

average = False
concatenate = True
nspectra_to_combine = 10

downsample = True
downsample_fac = 100

### Use this option with care: if you parallelize and ask it to plot,
### you'll get ncore * (a few) plots up simultaneously
plot_demod = False

### Constants
dipole_units = constants.e * (1e-6) # to convert e um -> C m

### Bead-specific constants
p0 = 100.0 * dipole_units  #  C * m


# rhobead = {'val': 1850.0, 'sterr': 1.0, 'syserr': 1.0}
mbead_dic = {'val': 84.3e-15, 'sterr': 1.0e-15, 'syserr': 1.5e-15}
mbead = mbead_dic['val']
Ibead = bu.get_Ibead(mbead=mbead_dic)['val']

############################################################################
############################################################################
############################################################################
############################################################################

colors = bu.get_color_map(n_mc, cmap='plasma')[::-1]


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
    try:
        fsamp = params['fsamp']
        fsamp_ds = fsamp
    except Exception:
        fsamp = 1.0e6
        fsamp_ds = fsamp

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
    if average:
        psd_array = []

    ### Loop over the datafiles 
    for fileind, file in enumerate(datfiles):

        ### Break the loop if we've acquired enough data
        if fileind >= nspectra_to_combine - 1:
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

        ### construct an estimate of the cross-polarized light
        crossp = np.sin(phi)**2
        # crossp = np.abs(px)  

        ### Normalize to avoid numerical errors. Try to get the max to sit at 10
        crossp *= (10.0 / np.max(np.abs(crossp)))

        ### Build up the long signal if concatenation is desired
        if concatenate:
            if not len(long_sig):
                long_t = tvec
                long_sig = crossp
            else:
                long_t = np.concatenate((tvec, long_t))
                long_sig = np.concatenate((crossp, long_sig))

        ### Using a hilbert transform, demodulate the amplitude and phase of
        ### the carrier signal. Filter things if desired.
        carrier_amp, carrier_phase \
                = bu.demod(crossp, fsig, fsamp, harmind=2.0, filt=True, \
                           bandwidth=4000.0, plot=plot_demod, ncycle_pad=100, \
                           tukey=True, tukey_alpha=1e-3)

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
        nsamp_ds = int(nsamp / downsample_fac)

        ### Hilbert transform demodulation
        carrier_amp_long, carrier_phase_long \
                = bu.demod(long_sig, fsig, fsamp, harmind=2.0, filt=True, \
                           bandwidth=4000.0, plot=plot_demod, ncycle_pad=100, \
                           tukey=True, tukey_alpha=1e-4)

        ### Downsampling
        carrier_phase_ds, tvec_ds = signal.resample(carrier_phase_long, nsamp_ds, \
                                                    t=long_t, window=None)

        ### Compute the ASD of the downsampled signals
        fit_asd = bu.fft_norm(nsamp_ds, fsamp_ds) * np.abs(np.fft.rfft(carrier_phase_ds))
        asd_errs = []

        ### Compute the new frequency arrays
        freqs = np.fft.rfftfreq(nsamp_ds, d=1.0/fsamp_ds)

    ### Fit either the averaged ASD or the ASD of the concatenated signal
    params, cov = bu.fit_damped_osc_amp(fit_asd, fsamp_ds, plot=False, \
                                        sig_asd=True, linearize=True, \
                                        asd_errs=asd_errs, fit_band=[100.0,1000.0])

    return [pressure, freqs, fit_asd, params, cov]


### Analyze all the results in parallel
results = Parallel(n_jobs=ncore)( delayed(proc_mc)(ind) for ind in list(range(n_mc))[::-1] )


### Loop over the results and plot each one
colors = bu.get_color_map(len(results), cmap='plasma')
pressures = []
gammas = []
for resultind, result in enumerate(results):
    pressures.append(result[0])
    label = '$ p = {:0.1f}$ mbar'.format(pressures[-1]*0.01)
    gammas.append(result[3][2])
    plt.loglog(result[1], result[2], color=colors[resultind], alpha=0.7, label=label)
    plt.loglog(result[1], bu.damped_osc_amp(result[1], *result[3]), \
               color=colors[resultind], lw=3)
plt.legend(fontsize=10, loc='upper left')

plt.figure()
plt.plot(pressures, gammas)

plt.show()

### Save the reslts for later analysis if desired
pickle.dump(results, open(save_filename, 'wb'))




