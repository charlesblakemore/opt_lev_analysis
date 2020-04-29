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
ncore = 20

plt.rcParams.update({'font.size': 14})


base = '/data/old_trap_processed/spinsim_data/libration_tests/'

dirname = os.path.join(base, 'high_pressure_sweep')
n_mc = bu.count_subdirectories(dirname)

hdf5 = True

fsamp = 1.0e6


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

    pressure = params['pressure'] * 1e-2   # convert back to mbar
    drive_amp = params['drive_amp']
    fsig = params['drive_freq']

    if hdf5:
        ext = '.h5'
    else:
        ext = '.npy'

    datfiles, lengths = bu.find_all_fnames(cdir, ext=ext, verbose=False, \
                                            sort_time=True, use_origin_timestamp=True)
    nfiles = lengths[0]

    all_amp = np.array([])
    all_t = np.array([])

    for fileind, file in enumerate(datfiles):
        # bu.progress_bar(fileind, nfiles, suffix='{:d}/{:d}'.format(i+1, n_mc))

        if hdf5:
            fobj = h5py.File(file, 'r')
            dat = np.copy(fobj['sim_data'])
            fobj.close()
        else:
            dat = np.load(file)

        nsamp = dat.shape[1]

        tvec = dat[0]
        px = dat[1]

        crossp = np.abs(px)

        carrier_amp, carrier_phase \
                = bu.demod(crossp, fsig, fsamp, harmind=2.0, filt=True, \
                           bandwidth=4000.0, plot=False)

        params, cov = bu.fit_damped_osc_amp(carrier_phase, fsamp, plot=False)

        libration_amp, libration_phase \
                = bu.demod(carrier_phase, params[1], fsamp, harmind=1.0, \
                           filt=True, filt_band=[300, 2000], plot=False)

        tvec_cut = tvec[5000:nsamp-5000]
        amp_cut = libration_amp[5000:nsamp-5000]

        step = int(len(amp_cut) / 100)
        amp_cut_ds = amp_cut[::step]
        tvec_cut_ds = tvec_cut[::step]

        all_amp = np.concatenate( (all_amp, amp_cut_ds) )
        all_t = np.concatenate( (all_t, tvec_cut_ds) )


    return [pressure, all_t, all_amp]



results = Parallel(n_jobs=ncore)( delayed(proc_mc)(ind) for ind in list(range(n_mc)) )

pickle.dump(results, open('./derp.p', 'wb'))

for ind, result in enumerate(results[::-1]):

    pressure, all_t, all_amp = result
    plt.plot(all_t, all_amp, color=colors[ind], \
             label='{:0.3g} mbar'.format(pressure))

plt.xlabel('Time [s]')
plt.ylabel('Amplitude of Phase Modulation [rad]')
plt.legend()
plt.tight_layout()

plt.show()


