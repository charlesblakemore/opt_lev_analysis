import os, sys, time, itertools, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as optimize
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed
# ncore = 1
ncore = 25

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 14})
np.random.seed(12345)



#############################
### Which data to analyze ###
#############################
# dir_name = '/data/old_trap/20200727/bead1/spinning/sideband_test_1Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/sideband_test_5Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/sideband_test_7Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/amplitude_impulse_1Vpp-to-7Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/wobble_slow_2'
# dir_name = '/data/old_trap/20200727/bead1/spinning/wobble_fast'
# file_inds = (0, 1000)
file_step = 1

# init_features = [21.3, 1178.8]
# init_features = []

try:
    trial_ind = int(sys.argv[1])
except:
    trial_ind = 1

# dir_name = '/data/old_trap/20200727/bead1/spinning/phase_impulse_+90deg'
# dir_name = '/data/old_trap/20200727/bead1/spinning/phase_impulse_-90deg'
# dir_name = '/data/old_trap/20200727/bead1/spinning/arb_phase_impulse_+90deg'
dir_name = '/data/old_trap/20200727/bead1/spinning/arb_phase_impulse_many_2/trial_{:04d}'.format(trial_ind)
file_inds = (0, 50)

# init_features = [25.8, 1355.3]
init_features = [50.0, 1385.0]
# init_features = []

### Filter constants
fspin = 19000
wspin = 2.0*np.pi*fspin
bandwidth = 10000.0

### Some crude limits to keep from loading too much into memory
output_band = (0, 5000)
drive_output_band = (15000, 45000)

### Boolean flags for various sorts of plotting (used for debugging usually)
plot_demod = False
plot_peaks = False

### Should probably measure these monitor factors
tabor_mon_fac = 100
#tabor_mon_fac = 100 * (1.0 / 0.95)



#############################
### Peak finding settings ###
#############################
window = 100
delta_fac = 4.0
lower_delta_fac = 4.0

exclude_df = 10



#################################
### Feature tracking settings ###
#################################
track_features = True
track_drive_features = True
allow_new_features = True
allowed_jumps = [1.1, 0.3]  # allowed jump between integrations given as a fraction of the feature
feature_base = '/data/old_trap_processed/spinning/feature_tracking/'
# phase_feature_savepath = os.path.join(feature_base, '20200727/phase_impulse_+90deg.p')
# drive_feature_savepath = os.path.join(feature_base, '20200727/phase_impulse_+90deg_drive.p')
# phase_feature_savepath = os.path.join(feature_base, '20200727/phase_impulse_-90deg.p')
# drive_feature_savepath = os.path.join(feature_base, '20200727/phase_impulse_-90deg_drive.p')
# phase_feature_savepath = os.path.join(feature_base, '20200727/arb_phase_impulse_+90deg.p')
# drive_feature_savepath = os.path.join(feature_base, '20200727/arb_phase_impulse_+90deg_drive.p')
phase_feature_savepath = os.path.join(feature_base, \
                                '20200727/arb_phase_impulse_many_2_{:04d}.p'.format(trial_ind))
drive_feature_savepath = os.path.join(feature_base, \
                                '20200727/arb_phase_impulse_many_2_{:04d}_drive.p'.format(trial_ind))





########################################################################
########################################################################
########################################################################

if plot_demod or plot_peaks:
    ncore = 1

date = re.search(r"\d{8,}", dir_name)[0]

files, _ = bu.find_all_fnames(dir_name, ext='.h5', sort_time=True)
files = files[file_inds[0]:file_inds[1]:file_step]

Ibead = bu.get_Ibead(date=date)


def line(x, a, b):
    return a*x + b



fobj = bu.hsDat(files[0], load=True, load_attribs=True)

nsamp = fobj.nsamp
fsamp = fobj.fsamp
fac = bu.fft_norm(nsamp, fsamp)

time_vec = np.arange(nsamp) * (1.0 / fsamp)
full_freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

out_inds = (full_freqs > output_band[0]) * (full_freqs < output_band[1])
drive_out_inds = (full_freqs > drive_output_band[0]) * (full_freqs < drive_output_band[1])

inds = np.abs(full_freqs - fspin) < 200.0

vperp = fobj.dat[:,0]
elec3 = fobj.dat[:,1]*tabor_mon_fac

elec3_fft = np.fft.rfft(elec3)*fac
true_fspin = np.average(full_freqs[inds], weights=np.abs(elec3_fft)[inds])

amp, phase_mod = bu.demod(vperp, true_fspin, fsamp, plot=plot_demod, \
                          filt=True, bandwidth=bandwidth,
                          tukey=True, tukey_alpha=5.0e-4, \
                          detrend=True, detrend_order=1, harmind=2.0)

phase_mod_fft = np.fft.rfft(phase_mod)[out_inds] * fac
freqs = full_freqs[out_inds]

first_fft = (freqs, phase_mod_fft)
first_drive_fft = (full_freqs, elec3_fft)




times = []
for file in files:
    fobj = bu.hsDat(file, load=False, load_attribs=True)
    times.append(fobj.time)
times = np.array(times) * 1e-9
times -= times[0]




def proc_file(file):

    fobj = bu.hsDat(file, load=True)

    vperp = fobj.dat[:,0]
    elec3 = fobj.dat[:,1]*tabor_mon_fac

    inds = np.abs(full_freqs - fspin) < 200.0

    elec3_fft = np.fft.rfft(elec3)*fac
    true_fspin = np.average(full_freqs[inds], weights=np.abs(elec3_fft)[inds])

    amp, phase_mod = bu.demod(vperp, true_fspin, fsamp, plot=plot_demod, \
                              filt=True, bandwidth=bandwidth,
                              tukey=True, tukey_alpha=5.0e-4, \
                              detrend=True, detrend_order=1, harmind=2.0)

    phase_mod_fft = np.fft.rfft(phase_mod)[out_inds] * fac
    freqs = full_freqs[out_inds]

    drive_fft = elec3_fft[drive_out_inds] * fac
    drive_freqs = full_freqs[drive_out_inds]


    upper_ind = np.argmin(np.abs(freqs - 200.0))

    ### Fit a power law to the sideband ASD, ignoring the DC bin
    popt, pcov = optimize.curve_fit(line, np.log(freqs[1:upper_ind]), \
                                    np.log(np.abs(phase_mod_fft[1:upper_ind])), \
                                    maxfev=10000, p0=[0.0, 0.0])

    ### Remove the power law from the data
    if np.abs(popt[0]) > 0.5:
        phase_mod_fft *= 1.0 / (np.exp(popt[1]) * freqs**popt[0])

    ### Find the peaks
    phase_mod_peaks = \
            bu.find_fft_peaks(freqs, phase_mod_fft, window=window, \
                              lower_delta_fac=lower_delta_fac, \
                              delta_fac=delta_fac, \
                              exclude_df=exclude_df)
    drive_peaks = \
            bu.find_fft_peaks(drive_freqs, drive_fft, window=window, \
                              lower_delta_fac=10.0, delta_fac=10.0, \
                              exclude_df=exclude_df)

    ### Put the power law back in to the peak amplitudes so they can
    ### be plotted over the original data with the plot_pdet() functions
    if len(phase_mod_peaks):
        if np.abs(popt[0]) > 0.5:
            phase_mod_peaks[:,1] *= (np.exp(popt[1]) * phase_mod_peaks[:,0]**popt[0])

    if plot_peaks:
        bu.plot_pdet([phase_mod_peaks, []], freqs, np.abs(phase_mod_fft), \
                     loglog=True, show=False)
        bu.plot_pdet([drive_peaks, []], drive_freqs, np.abs(drive_fft), \
                     loglog=True, show=True)

    return (phase_mod_peaks, drive_peaks)



results = Parallel(n_jobs=ncore)(delayed(proc_file)(file) for file in tqdm(files))


phase_peaks_all = []
drive_peaks_all = []
for phase_peaks, drive_peaks in results:
    phase_peaks_all.append(phase_peaks)
    drive_peaks_all.append(drive_peaks)




phase_feature_lists = \
    bu.track_spectral_feature(phase_peaks_all, first_fft=first_fft, \
                              init_features=init_features, \
                              allowed_jumps=allowed_jumps)

bu.make_all_pardirs(phase_feature_savepath)
pickle.dump(phase_feature_lists, open(phase_feature_savepath, 'wb'))

if track_drive_features:
    drive_feature_lists = \
        bu.track_spectral_feature(drive_peaks_all, first_fft=first_drive_fft, \
                                  init_features=[fspin, 2.0*fspin], \
                                  allowed_jumps=0.01)

    pickle.dump(drive_feature_lists, open(drive_feature_savepath, 'wb'))



