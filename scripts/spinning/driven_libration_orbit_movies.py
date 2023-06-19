import sys, os, time, itertools, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import dill as pickle

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as opti
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

np.random.seed(12345)

ncore = 30

# date = '20230410'
# base = f'/data/old_trap/{date}/bead1/spinning/'

# # meas = 'phase_modulation_sweeps/10Hz_to_700Hz_0007'
# meas = 'phase_modulation_sweeps/700Hz_to_10Hz_0007'


date = '20230531'
base = f'/data/old_trap/{date}/bead1/spinning/phase_modulation_sweeps/'

meas = '10Hz_to_700Hz_8Vpp_500sec_settle_50mrad'


# desired_drive_freqs = [650.0]
# t_movie = 0.015
# data_decimate = 1
# frame_decimate = 10
# fps = 20

desired_drive_freqs = [50.0]
t_movie = 0.5
data_decimate = 2
frame_decimate = 100
fps = 20

# desired_drive_freqs = [100.0]
# t_movie = 0.06
# data_decimate = 1
# frame_decimate = 40
# fps = 20

t_offset = 0.0
# t_offset = 2.0

markersize = 30


check_frame_count = True

movie_dir = f'/home/cblakemore/plots/{date}/spinning/libration_orbits'

# suffix = None
suffix = meas.split('/')[-1]

add_freq_to_suffix = True


xlabel = 'Libration $\\phi$ [rad]'
ylabel = '$\\partial \\phi / \\partial t$ [Hz]'


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




def get_pm_freqs(dir_name):

    files, _ = bu.find_all_fnames(dir_name, ext='.h5', \
                                  sort_time=True, \
                                  skip_subdirectories=False)

    pm_freqs = {}
    for file in files:
        fobj = bu.hsDat(file, load=False, load_attribs=True)
        pm_freq = fobj.pm_freq
        pm_freqs[pm_freq] = file

    return pm_freqs




def proc_file(file, movie_dir, suffix, add_freq_to_suffix):

    fobj = bu.hsDat(file, load=True)
    nsamp = fobj.nsamp
    fsamp = fobj.fsamp

    fac = bu.fft_norm(nsamp, fsamp)

    vperp = fobj.dat[:,0]
    elec3 = fobj.dat[:,1] * tabor_mon_fac

    full_freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)
    elec3_fft = np.fft.rfft(elec3)
    true_fspin = bu.find_freq(elec3[:int(0.1*fsamp)], \
                              fsamp, freq_guess=25000.1)
    pm_freq = fobj.pm_freq
    if add_freq_to_suffix and suffix is not None:
        suffix += f'_freq{int(pm_freq):d}Hz'
    if t_offset:
        suffix += (f'_toff{t_offset:0.1f}').replace('.', '_')

    bandwidth = 10.0*pm_freq
    phase_hpf = 0.5*pm_freq

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

    tind = int(t_offset * fsamp)
    nplot = int(t_movie * fsamp)

    plot_t = (np.arange(nplot) * (1.0 / fsamp))[::data_decimate]
    plot_t += t_offset

    plot_x = phase_mod[tind:tind+nplot:data_decimate]
    plot_y = np.gradient(plot_x, 1.0/fsamp) / (2.0*np.pi)

    if check_frame_count:
        print("You're about to try making a movie with:")
        print(f"  {int(nplot/(frame_decimate*data_decimate)):d} frames")
        print("")
        input("Press [Enter] to continue...")

    annotate_list = [f'Orbit at {pm_freq:0.1f} Hz']

    bu.make_all_pardirs(os.path.join(movie_dir, 'test.file'))
    bu.animate_trajectory(plot_x, plot_y, fps=fps, \
                          markersize=markersize, \
                          savepath=movie_dir, suffix=suffix, \
                          annotate_time=True, plot_t=plot_t, \
                          annotate_list=annotate_list, \
                          ncore=ncore, xlabel=xlabel, \
                          ylabel=ylabel, frame_decimate=frame_decimate)






dir_name = os.path.join(base, meas)
pm_freqs = get_pm_freqs(dir_name)

pm_freqs_arr = list(pm_freqs.keys())
pm_freqs_arr.sort()
pm_freqs_arr = np.array(pm_freqs_arr)

files_to_plot = []
for desired_freq in desired_drive_freqs:
    closest_freq = \
        pm_freqs_arr[np.argmin(np.abs(pm_freqs_arr-desired_freq))]
    relevant_file = pm_freqs[closest_freq]
    if relevant_file not in files_to_plot:
        files_to_plot.append(relevant_file)

for file in files_to_plot:
    proc_file(file, movie_dir, suffix, add_freq_to_suffix)











