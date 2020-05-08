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


base = '/data/old_trap_processed/spinsim_data/libration_tests/'

# dirname = os.path.join(base, 'high_pressure_sweep')
dirname = os.path.join(base, 'initial_angle_manyp_1')
n_mc = bu.count_subdirectories(dirname)

hdf5 = True
ext = '.h5'

### Paths for saving
save_base = '/home/cblakemore/opt_lev_analysis/spin_beads_sim/processed_results/'
save_filename = os.path.join(save_base, 'libration_spectra_manyp_1.p')

nspectra_to_avg = 10

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
    cdir = os.path.join(dirname, 'mc_{:d}'.format(i))

    param_path = os.path.join(cdir, 'params.p')
    params = pickle.load( open(param_path, 'rb') )

    pressure = params['pressure']
    drive_amp = params['drive_amp']
    fsig = params['drive_freq']
    try:
        fsamp = params['fsamp']
    except Exception:
        fsamp = 1.0e6

    datfiles, lengths = bu.find_all_fnames(cdir, ext=ext, verbose=False, \
                                            sort_time=True, use_origin_timestamp=True)
    datfiles = datfiles[::-1]
    nfiles = lengths[0]

    psd_array = []
    for fileind, file in enumerate(datfiles):

        if fileind >= nspectra_to_avg - 1:
            break

        if hdf5:
            fobj = h5py.File(file, 'r')
            dat = np.copy(fobj['sim_data'])
            fobj.close()
        else:
            dat = np.load(file)

        nsamp = dat.shape[1]
        nsamp_ds = int(nsamp / downsample_fac)

        tvec = dat[0]
        px = dat[1]

        crossp = np.abs(px)
        carrier_amp, carrier_phase \
                = bu.demod(crossp, fsig, fsamp, harmind=2.0, filt=True, \
                           bandwidth=4000.0, plot=plot_demod)

        if downsample:
            carrier_phase_ds, tvec_ds = signal.resample(carrier_phase, nsamp_ds, t=tvec, window=None)

            dt = tvec_ds[1] - tvec_ds[0]
            fsamp_ds = 1.0 / dt

            freqs = np.fft.rfftfreq(nsamp_ds, d=dt)
            carrier_phase_asd = bu.fft_norm(nsamp_ds, fsamp_ds) * np.abs(np.fft.rfft(carrier_phase_ds))
        else:
            freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)
            carrier_phase_asd = bu.fft_norm(nsamp, fsamp) * np.abs(np.fft.rfft(carrier_phase))

        if not len(psd_array):
            psd_array = np.zeros((nspectra_to_avg, len(carrier_phase_asd)), dtype=np.float64)
        psd_array[fileind,:] += carrier_phase_asd**2

    if downsample:
        fsamp = fsamp_ds
    avg_psd = np.mean(psd_array, axis=0)
    avg_asd = np.sqrt(avg_psd)

    asd_errs = 0.5 * avg_asd * (np.std(psd_array, axis=0) * np.sqrt(1.0 / nspectra_to_avg)) / avg_psd

    params, cov = bu.fit_damped_osc_amp(avg_asd, fsamp, plot=False, \
                                        sig_asd=True, linearize=True, \
                                        asd_errs=asd_errs, fit_band=[100.0,1000.0])

    return [pressure, freqs, avg_asd, params, cov]



results = Parallel(n_jobs=ncore)( delayed(proc_mc)(ind) for ind in list(range(n_mc))[::-1] )

colors = bu.get_color_map(len(results), cmap='plasma')
pressures = []
gammas = []
for resultind, result in enumerate(results):
    pressures.append(result[0])
    gammas.append(result[3][2])
    plt.loglog(result[1], result[2], color=colors[resultind], alpha=0.7)
    plt.loglog(result[1], bu.damped_osc_amp(result[1], *result[3]), \
               color=colors[resultind], lw=3)

plt.figure()
plt.plot(pressures, gammas)

plt.show()


pickle.dump(results, open(save_filename, 'wb'))

for ind, result in enumerate(results[::-1]):
    pressure, all_t, all_amp = result
    # lab = '{:0.3g} mbar'.format(pressure)
    plt.plot(all_t, all_amp, color=colors[ind])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude of Phase Modulation [rad]')
# plt.legend()
plt.tight_layout()
plt.show()


