import os, sys, time, itertools, re, warnings
import numpy as np
import dill as pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

try:
    trial_ind = int(sys.argv[1])
except:
    trial_ind = 1

# dir_name = '/data/old_trap/20200727/bead1/spinning/phase_impulse_+90deg'
# dir_name = '/data/old_trap/20200727/bead1/spinning/phase_impulse_-90deg'
# dir_name = '/data/old_trap/20200727/bead1/spinning/arb_phase_impulse_+90deg'
dir_name = '/data/old_trap/20200727/bead1/spinning/arb_phase_impulse_many_2/trial_{:04d}'.format(trial_ind)
file_inds = (0, 50)
file_step = 1

# libration_guess = 0.0
# libration_guess = 1335.8
libration_guess = 1385.0

### Carrier filter constants
fspin = 19000
wspin = 2.0*np.pi*fspin
bandwidth = 10000.0

# libration_fit_band = []
libration_filt_band = [1000.0, 1400.0]
libration_bandwidth = 1000

# colorbar_limits = []
colorbar_limits = [1200, 1400]

### Boolean flags for various sorts of plotting (used for debugging usually)
plot_carrier_demod = False
plot_libration_demod = False
plot_downsample = False

plot_lib_amp = False

### Should probably measure these monitor factors
tabor_mon_fac = 100
#tabor_mon_fac = 100 * (1.0 / 0.95)

out_nsamp = 100000
out_cut = 100

xlim = (0, 1500)
# ylim = (-0.25, 1.8)
ylim = (-1.8, 1.8)

# yticks = []
yticks = [-np.pi/2.0, 0.0, np.pi/2.0]
yticklabels = ['$-\\pi/2$', '0', '$\\pi/2$']


save_sequence = True
# fig_basepath = '/home/cblakemore/plots/20200727/spinning/phase_impulse_+90deg/ringdown_amp/'
# fig_basepath = '/home/cblakemore/plots/20200727/spinning/phase_impulse_-90deg/ringdown_amp/'
# fig_basepath = '/home/cblakemore/plots/20200727/spinning/arb_phase_impulse_+90deg/ringdown_amp/'
fig_basepath = '/home/cblakemore/plots/20200727/spinning/arb_phase_impulse_many_2/trial_{:04d}/ringdown_amp/'.format(trial_ind)



########################################################################
########################################################################
########################################################################

def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0 * (x - mu)**2 / (2.0 * sigma**2)) + c



if plot_carrier_demod or plot_libration_demod or plot_downsample:
    ncore = 1

files, _ = bu.find_all_fnames(dir_name, ext='.h5', sort_time=True)
files = files[file_inds[0]:file_inds[1]:file_step]

times = []
for file in files:
    fobj = bu.hsDat(file, load=False, load_attribs=True)
    times.append(fobj.time)
times = np.array(times) * 1e-9
times -= times[0]


fobj = bu.hsDat(files[0], load=True, load_attribs=True)

nsamp = fobj.nsamp
fsamp = fobj.fsamp
fac = bu.fft_norm(nsamp, fsamp)

time_vec = np.arange(nsamp) * (1.0 / fsamp)
full_freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

vperp = fobj.dat[:,0]
elec3 = fobj.dat[:,1]

inds = np.abs(full_freqs - fspin) < 200.0

elec3_fft = np.fft.rfft(elec3)
true_fspin = np.average(full_freqs[inds], weights=np.abs(elec3_fft)[inds])


if not libration_guess:
    amp, phase_mod = bu.demod(vperp, true_fspin, fsamp, plot=plot_carrier_demod, \
                              filt=True, bandwidth=bandwidth,
                              tukey=True, tukey_alpha=5.0e-4, \
                              detrend=True, detrend_order=1, harmind=2.0)

    phase_mod_fft = np.fft.rfft(phase_mod) * fac

    fig, ax = plt.subplots(1,1)
    ax.loglog(full_freqs, np.abs(phase_mod_fft))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Phase ASD [rad/$\\sqrt{\\rm Hz}$]')
    ax.set_title('Identify Libration Frequency')
    fig.tight_layout()
    plt.show()

    libration_guess = float( input('Libration guess: ') )




def proc_file(file):

    fobj = bu.hsDat(file, load=True)

    vperp = fobj.dat[:,0]
    elec3 = fobj.dat[:,1]

    inds = np.abs(full_freqs - fspin) < 200.0

    elec3_fft = np.fft.rfft(elec3)
    true_fspin = np.average(full_freqs[inds], weights=np.abs(elec3_fft)[inds])

    carrier_amp, carrier_phase_mod = \
            bu.demod(vperp, true_fspin, fsamp, plot=plot_carrier_demod, \
                     filt=True, bandwidth=bandwidth,
                     tukey=True, tukey_alpha=5.0e-4, \
                     detrend=True, detrend_order=1, harmind=2.0)

    b1, a1 = signal.butter(3, np.array(libration_filt_band)*2.0/fsamp, btype='bandpass')
    carrier_phase_mod_filt = signal.filtfilt(b1, a1, carrier_phase_mod)

    if len(libration_filt_band):
        libration_inds = (full_freqs > libration_filt_band[0]) \
                                * (full_freqs < libration_filt_band[1])
    else:
        libration_inds = np.abs(full_freqs - libration_guess) < 0.5*libration_bandwidth

    phase_mod_fft = np.fft.rfft(carrier_phase_mod) * fac

    fit_x = full_freqs[libration_inds]
    fit_y = np.abs(phase_mod_fft[libration_inds])

    try:
        try:
            peaks = bu.find_fft_peaks(fit_x, fit_y, delta_fac=5.0, window=50)
        except:
            peaks = bu.find_fft_peaks(fit_x, fit_y, delta_fac=3.0, window=100)

        ind = np.argmax(peaks[:,1])
        true_libration_freq = peaks[ind,0]

    except:
        # p0 = [np.max(fit_y), np.average(fit_x, weights=fit_y), 1.0, 0.0]
        # popt, pcov = optimize.curve_fit(gauss, fit_x, fit_y, p0=p0)

        # true_libration_freq = popt[1]

        true_libration_freq = fit_x[np.argmax(fit_y)]


    libration_amp, libration_pahse = \
            bu.demod(carrier_phase_mod, true_libration_freq, fsamp, \
                     plot=plot_libration_demod, filt=True, \
                     filt_band=libration_filt_band, \
                     bandwidth=libration_bandwidth, \
                     tukey=True, tukey_alpha=5.0e-4, \
                     detrend=True, detrend_order=1.0, harmind=1.0)

    libration_ds, time_vec_ds = \
            signal.resample(carrier_phase_mod_filt, t=time_vec, num=out_nsamp)
    libration_amp_ds, time_vec_ds = \
            signal.resample(libration_amp, t=time_vec, num=out_nsamp)

    libration_ds = libration_ds[out_cut:int(-1*out_cut)]
    libration_amp_ds = libration_amp_ds[out_cut:int(-1*out_cut)]
    time_vec_ds = time_vec_ds[out_cut:int(-1*out_cut)]

    if plot_downsample:
        plt.plot(time_vec, carrier_phase_mod, color='C0', label='Original')
        plt.plot(time_vec_ds, libration_ds, color='C0', ls='--', label='Downsampled')
        plt.plot(time_vec, libration_amp, color='C1')#, label='Original')
        plt.plot(time_vec_ds, libration_amp_ds, color='C1', ls='--')#, label='Downsampled')
        plt.legend()
        plt.show()

        input()

    return (time_vec_ds, libration_ds, libration_amp_ds, true_libration_freq)




all_amp = Parallel(n_jobs=ncore)(delayed(proc_file)(file) for file in tqdm(files))

lib_freqs = []
for i, (tvec, lib, amp, lib_freq) in enumerate(all_amp):

    if len(colorbar_limits):
        if lib_freq < colorbar_limits[0]:
            lib_freq = colorbar_limits[0]
        elif lib_freq > colorbar_limits[1]:
            lib_freq = colorbar_limits[1]

    lib_freqs.append(lib_freq)

    if i == len(all_amp) - 1:
        final_time = tvec[-1] + times[-1]

max_lib_freq = np.max(lib_freqs)
min_lib_freq = np.min(lib_freqs)
pad = 0.1 * (max_lib_freq - min_lib_freq)

def freq_to_yval(freq):
    scaled = (freq - min_lib_freq + pad) / (max_lib_freq - min_lib_freq + 2.0*pad)
    return scaled * (ylim[1] - ylim[0]) + ylim[0]

if len(colorbar_limits):
    vmin, vmax = colorbar_limits
else:
    vmin = min_lib_freq - pad
    vmax = max_lib_freq + pad

norm = colors.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(1,1,figsize=(8,4))
divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size='2%', pad=0.05)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.plasma, \
                                norm=norm, orientation='vertical')
cb1.set_label('Libration Frequency [Hz]')

fig.add_axes(ax_cb)

ax.set_xlim(0, final_time)
ax.set_ylim(*ylim)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Phase Modulation [rad]')
if len(yticks):
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_yticks([], minor=True)
    ax.set_yticklabels([], minor=True)
fig.tight_layout()

vline = ax.axvline(0, color='k', zorder=3)
hline = ax.axhline(0, color='k', zorder=3)

my_colors = []
for i, (tvec, lib, amp, lib_freq) in enumerate(all_amp):
    lib_freq = lib_freqs[i]
    color = bu.get_single_color(lib_freq, cmap='plasma', \
                                vmin=vmin, vmax=vmax)
    my_colors.append(color)
    t0 = times[i]
    # ax.plot(tvec+t0, amp, color=color, zorder=2)
    # if not plot_just_amp:
    ax.plot(tvec+t0, lib, alpha=1.0, color=color, zorder=1)


if save_sequence:

    for i, t0 in enumerate(times):
        figname = os.path.join(fig_basepath, 'image_{:04d}.png'.format(i))
        if i == 0:
            bu.make_all_pardirs(figname)

        vline.remove()
        hline.remove()
        vline = ax.axvline(t0, color='k', zorder=3)
        hline = ax.axhline(freq_to_yval(lib_freqs[i]), xmin=0.975, \
                           xmax=1.0, color=my_colors[i], zorder=3, lw=3)

        fig.canvas.draw_idle()
        fig.savefig(figname)
        print('Saved:  image_{:04d}.png'.format(i))

else:
    vline.remove()
    hline.remove()
    plt.show()














