import sys, os, time, itertools, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as opti
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed
# ncore = 1
ncore = 30

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 14})
np.random.seed(12345)

# # date = '20200727'
# # date = '20200924'
# base = '/data/old_trap/{:s}/bead1/spinning/'.format(date)
# # meas = 'dds_phase_modulation_sweep/trial_0000'
# meas = 'dds_phase_modulation_sweep_8Vpp/'

date = '20230410'
base = f'/data/old_trap/{date}/bead1/spinning/'

# meas = 'phase_modulation_sweeps/10Hz_to_700Hz_0007'
meas = 'phase_modulation_sweeps/700Hz_to_10Hz_0007'

dir_name = os.path.join(base, meas)
file_inds = (0, 500)
file_step = 1

# mbead = bu.get_mbead(date=date)
mbead = {'val': 380e-15, 'sterr': 0.0, 'syserr': 0.0}

Ibead = bu.get_Ibead(mbead=mbead, \
                     rhobead=bu.rhobead['german7'], \
                     date=date)

### Filter constants
fspin = 25000
wspin = 2.0*np.pi*fspin
bandwidth = 10000.0
drive_bandwidth = 2000.0

### Sideband filtering overrides basic filtering behavior
sideband_filter = True
sideband_filter_nharm = 5
sideband_filter_bw = 10.0

# notch_freqs = []
notch_freqs = [49989.9, 50010.4]

# notch_qs = []
notch_qs = [50000.0, 50000.0]

detrend = True
force_2pi_wrap = False

### Boolean flags for various sorts of plotting (used for debugging usually)
plot_demod = False

### Should probably measure these monitor factors
tabor_mon_fac = 100
#tabor_mon_fac = 100 * (1.0 / 0.95)


#########################
### Plotting behavior ###
#########################
output_band = (3, 2000)

### Full spectra plot limits
xlim = (0.5, 5000)
ylim = (3e-4, 5e0)


if plot_demod:
    ncore = 1


def proc_file(file):

    fobj = bu.hsDat(file, load=True)

    vperp = fobj.dat[:,0]
    elec3 = fobj.dat[:,1] * tabor_mon_fac

    inds = np.abs(full_freqs - fspin) < 200.0

    elec3_fft = np.fft.rfft(elec3)
    # true_fspin = full_freqs[np.argmax(np.abs(elec3_fft))]
    true_fspin = 25000.1

    pm_freq = fobj.pm_freq    

    bandwidth = 10.0*pm_freq
    # bandwidth = 100.0

    phase_hpf = 0.5*pm_freq

    drive_amp, drive_phase_mod = \
                    bu.demod(elec3, true_fspin, fsamp, plot=False, \
                             filt=True, bandwidth=drive_bandwidth, \
                             notch_freqs=notch_freqs, notch_qs=notch_qs, \
                             pre_tukey=False, post_tukey=False, \
                             tukey_alpha=5.0e-3, \
                             detrend=detrend, harmind=1.0, \
                             force_2pi_wrap=force_2pi_wrap, \
                             optimize_frequency=False, unwrap=False,\
                             pad=True, npad=1, pad_mode='reflect', \
                             phase_hp=True, phase_hpf=phase_hpf)

    amp, phase_mod = bu.demod(vperp, true_fspin, fsamp, plot=plot_demod, \
                              filt=True, bandwidth=bandwidth, \
                              notch_freqs=notch_freqs, notch_qs=notch_qs, \
                              pre_tukey=False, post_tukey=False, \
                              tukey_alpha=5.0e-3, \
                              detrend=detrend, harmind=2.0, \
                              force_2pi_wrap=force_2pi_wrap, \
                              optimize_frequency=False, unwrap=False,\
                              pad=True, npad=1, pad_mode='reflect', \
                              phase_hp=True, phase_hpf=phase_hpf, \
                              sideband_filter=sideband_filter, \
                              sideband_filter_freq=pm_freq, \
                              sideband_filter_nharm=sideband_filter_nharm, \
                              sideband_filter_bw=sideband_filter_bw)

    phase_mod_fft = np.fft.rfft(phase_mod)[out_inds] * fac
    drive_phase_mod_fft = np.fft.rfft(drive_phase_mod)[out_inds] * fac

    return (phase_mod_fft, drive_phase_mod_fft, pm_freq, fobj.time)




def proc_directory(dir_name):

    files, _ = bu.find_all_fnames(dir_name, ext='.h5', \
                                  sort_time=True, \
                                  skip_subdirectories=False)
    files = files[file_inds[0]:file_inds[1]:file_step]
    nfiles = len(files)

    results = Parallel(n_jobs=ncore)( delayed(proc_file)(file) \
                                        for file in tqdm(files) )

    phase_mod_results, drive_phase_mod_results, pm_freqs, times \
        = map(list,zip(*results))

    times = np.array(times) * 1e-9
    times -= times[0]



    fobj = bu.hsDat(files[0], load=False, load_attribs=True)

    nsamp = fobj.nsamp
    fsamp = fobj.fsamp
    fac = bu.fft_norm(nsamp, fsamp)

    time_vec = np.arange(nsamp) * (1.0 / fsamp)
    full_freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

    out_inds = (full_freqs > output_band[0]) * (full_freqs < output_band[1])
    out_freqs = full_freqs[out_inds]
    

    tf_freqs = []
    tf_vals = []
    drive_vals = []
    for i in range(nfiles):

        phase_mod_fft = phase_mod_results[i]
        drive_phase_mod_fft = drive_phase_mod_results[i]

        try:
            pm_freq = pm_freqs[i]
            pm_ind = np.argmin(np.abs(out_freqs - pm_freq))
        except:
            pm_ind = np.argmax(np.abs(drive_phase_mod_fft)[1:]) + 1
            pm_freq = out_freqs[pm_ind]

        tf_val = phase_mod_fft[pm_ind] / drive_phase_mod_fft[pm_ind]

        tf_freqs.append(pm_freq)
        tf_vals.append(tf_val)
        drive_vals.append(np.abs(drive_phase_mod_fft[pm_ind]))

    tf_freqs = np.array(tf_freqs)
    tf_vals = np.array(tf_vals)
    drive_vals = np.array(drive_vals)

    sorter = np.argsort(tf_freqs)
    tf_freqs = tf_freqs[sorter]
    tf_vals = tf_vals[sorter]
    drive_vals = drive_vals[sorter]

    duplicates = []
    for i, tf_freq in enumerate(tf_freqs):
        inds = np.arange(len(tf_freqs))[tf_freqs == tf_freq]
        nfreq = len(inds)
        if nfreq > 1:
            small_drive_ind = inds[np.argmax(drive_vals[inds])]
            for ind in inds:
                if ind != small_drive_ind:
                    duplicates.append(ind)
    duplicates.sort()
    duplicates = duplicates[::-1]
    for ind in duplicates:
        tf_freqs = np.delete(tf_freqs, ind)
        tf_vals = np.delete(tf_vals, ind)
        drive_vals = np.delete(drive_vals, ind)



fig, axarr = plt.subplots(2,1,sharex=True,figsize=(8,5), \
                          gridspec_kw={'height_ratios': [2,1]})

axarr[0].set_title('Libration Response to E-field Phase Modulation', fontsize=16)

axarr[0].loglog(tf_freqs, np.abs(tf_vals), 'o', ms=6)
axarr[0].set_ylabel('TF Mag [rad/rad]')

phase = np.angle(tf_vals)
pos = phase > 0.5*np.pi
phase -= pos * 2.0 * np.pi

axarr[1].semilogx(tf_freqs, phase, 'o', ms=6)
axarr[1].set_ylabel('TF Phase [rad]')
axarr[1].set_xlabel('Frequency [Hz]')
axarr[1].set_yticks([-np.pi, -np.pi/2.0, 0])
axarr[1].set_yticklabels(['$-\\pi$', '$-\\frac{\\pi}{2}$', '0'])
axarr[1].set_yticks([], minor=True)
axarr[1].set_yticklabels([], minor=True)

fig.tight_layout()

plt.show()

