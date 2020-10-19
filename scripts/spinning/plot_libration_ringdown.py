import os, sys, time, itertools, re, warnings
import numpy as np
import dill as pickle
from iminuit import Minuit, describe

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
ncore = 24

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 14})
np.random.seed(12345)



#############################
### Which data to analyze ###
#############################

try:
    meas_ind = int(sys.argv[1])
except:
    meas_ind = 0

try:
    trial_ind = int(sys.argv[2])
except:
    trial_ind = 0

# date = '20200727'
date = '20200924'

bead = 'bead1'

base = '/data/old_trap/{:s}/{:s}/spinning/'.format(date, bead)

# meas = 'arb_phase_impulse_many_2/trial_{:04d}'.format(trial_ind)
# file_inds = (7, 46)
# if meas_ind:
#     meas_ind += 1
#     meas = 'dds_phase_impulse_many_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
#     # meas = 'dds_phase_impulse_lower_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
#     # meas = 'dds_phase_impulse_low_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
#     # meas = 'dds_phase_impulse_mid_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
#     # meas = 'dds_phase_impulse_high_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
# else:
#     meas = 'dds_phase_impulse_many/trial_{:04d}'.format(trial_ind)
#     # meas = 'dds_phase_impulse_lower_dg/trial_{:04d}'.format(trial_ind)
#     # meas = 'dds_phase_impulse_low_dg/trial_{:04d}'.format(trial_ind)
#     # meas = 'dds_phase_impulse_mid_dg/trial_{:04d}'.format(trial_ind)
#     # meas = 'dds_phase_impulse_high_dg/trial_{:04d}'.format(trial_ind)
# file_inds = (12, 36)
# # file_inds = (12, 30)


if meas_ind:
    meas_ind += 1
    # meas = 'dds_phase_impulse_1Vpp_lower_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
    # meas = 'dds_phase_impulse_1Vpp_low_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
    # meas = 'dds_phase_impulse_1Vpp_mid_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
    # meas = 'dds_phase_impulse_1Vpp_high_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
else:
    meas = 'dds_phase_impulse_3Vpp/trial_{:04d}'.format(trial_ind)
    # meas = 'dds_phase_impulse_1Vpp_lower_dg/trial_{:04d}'.format(trial_ind)
    # meas = 'dds_phase_impulse_1Vpp_low_dg/trial_{:04d}'.format(trial_ind)
    # meas = 'dds_phase_impulse_1Vpp_mid_dg/trial_{:04d}'.format(trial_ind)
    # meas = 'dds_phase_impulse_1Vpp_high_dg/trial_{:04d}'.format(trial_ind)
# file_inds = (8, 100)
# file_inds = (8, 90)
file_inds = (8, 50)
# file_inds = (8, 25)


dir_name = os.path.join(base, meas)
file_step = 1

save_sequence = False
color_with_libration = False
plot_base = '/home/cblakemore/plots/{:s}/spinning/'.format(date)
# fig_basepath = os.path.join(plot_base, 'phase_impulse_+90deg/ringdown_amp/')
# fig_basepath = os.path.join(plot_base, 'phase_impulse_-90deg/ringdown_amp/')
# fig_basepath = os.path.join(plot_base, 'arb_phase_impulse_+90deg/ringdown_amp/')
fig_basepath = os.path.join(plot_base, meas, 'ringdown_amp')


# libration_guess = 0.0
# libration_guess = 1335.8
# libration_guess = 1385.0
# libration_guess = 1298.0
libration_guess = 297.9

### Carrier filter constants
# fspin = 19000
fspin = 25000
wspin = 2.0*np.pi*fspin
bandwidth = 10000.0

# libration_fit_band = []
# libration_filt_band = [1000.0, 1450.0]
# libration_filt_band = [900.0, 1350.0]
# libration_filt_band = [700.0, 1000.0]
# libration_filt_band = [175.0, 400.0]
libration_filt_band = [350.0, 600.0]
libration_bandwidth = 1000

notch_freqs = [49020.3]
notch_qs = [10000.0]

# colorbar_limits = []
# colorbar_limits = [1200, 1400]
colorbar_limits = [900, 1350]

### Boolean flags for various sorts of plotting (used for debugging usually)
plot_carrier_demod = False
plot_libration_demod = False
plot_downsample = False

plot_lib_amp = False

### Should probably measure these monitor factors
tabor_mon_fac = 100
#tabor_mon_fac = 100 * (1.0 / 0.95)

# out_nsamp = 200000
out_nsamp = 50000
out_cut = 100

xlim = (-100, 200)
# ylim = (-0.25, 1.8)
ylim = (-1.8, 1.8)

# yticks = []
yticks = [-np.pi/2.0, 0.0, np.pi/2.0]
yticklabels = ['$-\\pi/2$', '0', '$\\pi/2$']

fit_ringdown = True
# base_ringdown_fit_time = 1100.0
base_ringdown_fit_time = 200.0
# base_ringdown_fit_time = 120.0
adjust_fit_time = True
# ringdown_scale_fac = 2.3
ringdown_scale_fac = 1.0
# ringdown_scale_fac = 0.5
initial_offset = 0.0

plot_rebin = False
plot_ringdown_fit = True
close_xlim = True
show = False


processed_base = '/data/old_trap_processed/spinning/{:s}/'.format(date)

save_ringdown = True
# ringdown_dict_path = '/data/old_trap_processed/spinning/20200727/arb_libration_ringdowns.p'
ringdown_data_path = os.path.join(processed_base, 'dds_libration_ringdowns_3Vpp_less_pts.p')



########################################################################
########################################################################
########################################################################

bu.make_all_pardirs(ringdown_data_path)

def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0 * (x - mu)**2 / (2.0 * sigma**2)) + c

def ngauss(x, A, mu, sigma, c, n):
    return A * np.exp(-1.0 * np.abs(x - mu)**n / (2.0 * sigma**n)) + c



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

    try:
        phi_dg = fobj.attribs['phi_dg']
    except:
        phi_dg = 0.0

    inds = np.abs(full_freqs - fspin) < 200.0

    elec3_fft = np.fft.rfft(elec3)
    true_fspin = np.average(full_freqs[inds], weights=np.abs(elec3_fft)[inds])


    carrier_amp, carrier_phase_mod = \
            bu.demod(vperp, true_fspin, fsamp, plot=plot_carrier_demod, \
                     filt=True, bandwidth=bandwidth,
                     notch_freqs=notch_freqs, notch_qs=notch_qs, \
                     tukey=True, tukey_alpha=5.0e-4, \
                     detrend=True, detrend_order=1, harmind=2.0)


    # b1, a1 = signal.butter(3, np.array(libration_filt_band)*2.0/fsamp, btype='bandpass')
    sos = signal.butter(3, libration_filt_band, btype='bandpass', fs=fsamp, output='sos')
    # carrier_phase_mod_filt = signal.filtfilt(b1, a1, carrier_phase_mod)
    carrier_phase_mod_filt = signal.sosfiltfilt(sos, carrier_phase_mod)

    if len(libration_filt_band):
        libration_inds = (full_freqs > libration_filt_band[0]) \
                                * (full_freqs < libration_filt_band[1])
    else:
        libration_inds = np.abs(full_freqs - libration_guess) < 0.5*libration_bandwidth

    phase_mod_fft = np.fft.rfft(carrier_phase_mod) * fac

    lib_fit_x = full_freqs[libration_inds]
    lib_fit_y = np.abs(phase_mod_fft[libration_inds])

    try:
        try:
            peaks = bu.find_fft_peaks(lib_fit_x, lib_fit_y, delta_fac=5.0, window=50)
            ind = np.argmax(peaks[:,1])
        except:
            peaks = bu.find_fft_peaks(lib_fit_x, lib_fit_y, delta_fac=3.0, window=100)
            ind = np.argmax(peaks[:,1])

        true_libration_freq = peaks[ind,0]

    except:
        true_libration_freq = lib_fit_x[np.argmax(lib_fit_y)]

    libration_amp, libration_phase = \
            bu.demod(carrier_phase_mod, true_libration_freq, fsamp, \
                     plot=plot_libration_demod, filt=True, \
                     filt_band=libration_filt_band, \
                     bandwidth=libration_bandwidth, \
                     tukey=False, tukey_alpha=5.0e-4, \
                     detrend=False, detrend_order=1.0, harmind=1.0)

    libration_ds, time_vec_ds = \
            signal.resample(carrier_phase_mod_filt, t=time_vec, num=out_nsamp)
    libration_amp_ds, time_vec_ds = \
            signal.resample(libration_amp, t=time_vec, num=out_nsamp)

    libration_ds = libration_ds[out_cut:int(-1*out_cut)]
    libration_amp_ds = libration_amp_ds[out_cut:int(-1*out_cut)]
    time_vec_ds = time_vec_ds[out_cut:int(-1*out_cut)]

    if plot_downsample:
        plt.plot(time_vec, carrier_phase_mod_filt, color='C0', label='Original')
        plt.plot(time_vec_ds, libration_ds, color='C0', ls='--', label='Downsampled')
        plt.plot(time_vec, libration_amp, color='C1')#, label='Original')
        plt.plot(time_vec_ds, libration_amp_ds, color='C1', ls='--')#, label='Downsampled')
        plt.legend()
        plt.show()

        input()

    return (time_vec_ds, libration_ds, libration_amp_ds, true_libration_freq, phi_dg)




all_amp = Parallel(n_jobs=ncore)(delayed(proc_file)(file) for file in tqdm(files))

phi_dgs = []
lib_freqs = []
init_std = 0.0
found_start = False
for i, (tvec, lib, amp, lib_freq, phi_dg) in enumerate(all_amp):

    if i < 5:
        init_std += np.std(lib) / 5.0
    elif not found_start:
        if np.sum(amp > np.pi/4.0) > 0:
            initial_time = tvec[0] + times[i]
            chirp_start = tvec[np.argmax(np.abs(lib))] + times[i]
            found_start = True

    if len(colorbar_limits):
        if lib_freq < colorbar_limits[0]:
            lib_freq = colorbar_limits[0]
        elif lib_freq > colorbar_limits[1]:
            lib_freq = colorbar_limits[1]

    phi_dgs.append(phi_dg)
    lib_freqs.append(lib_freq)

    if i == len(all_amp) - 1:
        final_time = tvec[-1] + times[-1]

if not found_start:
    print('DID NOT DETECT A RINGDOWN!')
    exit()

initial_time -= chirp_start
times -= chirp_start
meas_phi_dg = np.mean(phi_dgs)

print(meas_phi_dg)

if adjust_fit_time:
    if meas_phi_dg == 0.0:
        ringdown_fit_time = base_ringdown_fit_time
    else:
        ringdown_fit_time = np.min([ringdown_scale_fac / meas_phi_dg, base_ringdown_fit_time])

    if close_xlim:
        upper = np.min([1.75*ringdown_fit_time, final_time-chirp_start])
        xlim = (-0.1*ringdown_fit_time, upper)
    else:
        xlim = (-0.5 * ringdown_fit_time, ringdown_fit_time)

    if ringdown_fit_time >= 10.0:
        nbins = 1000
    else:
        nbins = np.max([int(1000 * ringdown_fit_time / 10.0), 50])



if fit_ringdown:
    fit_times = []
    fit_amps = []
    fit_errs = []
    first = False
    for i, (tvec, lib, amp, lib_freq, phi_dg) in enumerate(all_amp):

        if times[i] < -3.0:
            continue

        if times[i] <= ringdown_fit_time:
            if not first:
                lower_ind = np.argmax(np.abs(lib))
                first = True
            else:
                lower_ind = 0

            if initial_offset and times[i] < initial_offset:
                continue

            upper_ind = np.argmin(np.abs(tvec + times[i] - ringdown_fit_time))

            fit_time = tvec[lower_ind:upper_ind] + times[i]
            fit_amp = amp[lower_ind:upper_ind]

            fit_time_rebin, fit_amp_rebin, fit_err_rebin = \
                    bu.rebin(fit_time, fit_amp, nbins=nbins, plot=plot_rebin, \
                             correlated_errs=True)

            fit_times.append(fit_time_rebin)
            fit_amps.append(fit_amp_rebin)
            fit_errs.append(fit_err_rebin)

    print('Fitting ringdown...', end=' ')
    sys.stdout.flush()

    fit_x = np.concatenate(fit_times)
    fit_y = np.concatenate(fit_amps)
    fit_err = np.concatenate(fit_errs)

    plot_x = np.linspace(fit_x[0], fit_x[-1], 500)

    fit_func = lambda x, amp0, t0, tau, c: amp0 * np.exp(-1.0 * (x - t0) / tau) + c

    npts = len(fit_x)
    def chi_sq(amp0, t0, tau, c):
        resid = np.abs(fit_y - fit_func(fit_x, amp0, t0, tau, c))**2
        variance = fit_err**2
        prior1 = np.abs(amp0 - np.pi/2.0)**2 / np.mean(variance)
        prior2 = np.abs(c - init_std)**2 / init_std**2
        return (1.0 / (npts - 1.0)) * np.sum(resid / variance) + prior1 + prior2

    m = Minuit(chi_sq,
               amp0 = np.pi/2.0, # set start parameter
               fix_amp0 = False,
               limit_amp0 = (np.pi/4.0, 6.0*np.pi/10.0), # if you want to limit things
               t0 = 0.0,
               fix_t0 = False,
               limit_t0 = (fit_x[0]-5.0, fit_x[0]+5.0),
               tau = 0.25*ringdown_fit_time,
               fix_tau = False,
               limit_tau = (0.0, 500.0),
               c = 0.0,
               fix_c = False,
               limit_c = (-np.pi/8.0, np.pi/8.0),
               errordef = 1,
               print_level = 0, 
               pedantic=False)
    m.migrad(ncall=500000)

    try:
        minos = m.minos()
        ringdown_fit = np.array([minos['amp0']['min'], minos['t0']['min'], \
                                 minos['tau']['min'], minos['c']['min']])
        ringdown_unc = np.array([np.mean(np.abs([minos['amp0']['lower'], minos['amp0']['upper']])), \
                                 np.mean(np.abs([minos['t0']['lower'], minos['t0']['upper']])), \
                                 np.mean(np.abs([minos['tau']['lower'], minos['tau']['upper']])), \
                                 np.mean(np.abs([minos['c']['lower'], minos['c']['upper']]))])
    except:
        print('MINOS FAILED!')
        ringdown_fit = np.array([m.values['amp0'], m.values['t0'], \
                                 m.values['tau'], m.values['c']])
        ringdown_unc = np.array([m.errors['amp0'], m.errors['t0'], \
                                 m.errors['tau'], m.errors['c']])

    print('Done!')

    if save_ringdown:

        try:
            ringdown_dict = pickle.load(open(ringdown_data_path, 'rb'))
        except FileNotFoundError:
            ringdown_dict = {}

        if phi_dg not in list(ringdown_dict.keys()):
            ringdown_dict[meas_phi_dg] = {}
            ringdown_dict[meas_phi_dg]['paths'] = []
            ringdown_dict[meas_phi_dg]['data'] = []
            ringdown_dict[meas_phi_dg]['fit'] = []
            ringdown_dict[meas_phi_dg]['unc'] = []
            ringdown_dict[meas_phi_dg]['chi_sq'] = []

        saved = False
        for pathind, path in enumerate(ringdown_dict[meas_phi_dg]['paths']):
            if path == dir_name:
                saved = True
                print('Already saved this one... overwriting!')
                break

        if saved:
            ringdown_dict[meas_phi_dg]['data'][pathind] = np.array([fit_x, fit_y, fit_err])
            ringdown_dict[meas_phi_dg]['fit'][pathind] = ringdown_fit
            ringdown_dict[meas_phi_dg]['unc'][pathind] = ringdown_unc
            ringdown_dict[meas_phi_dg]['chi_sq'][pathind] = m.fval

        else:
            ringdown_dict[meas_phi_dg]['paths'].append(dir_name)
            ringdown_dict[meas_phi_dg]['data'].append( np.array([fit_x, fit_y, fit_err]) )
            ringdown_dict[meas_phi_dg]['fit'].append( ringdown_fit )
            ringdown_dict[meas_phi_dg]['unc'].append( ringdown_unc )
            ringdown_dict[meas_phi_dg]['chi_sq'].append( m.fval )

        pickle.dump(ringdown_dict, open(ringdown_data_path, 'wb'))





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

if color_with_libration:
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size='2%', pad=0.05)
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.plasma, \
                                    norm=norm, orientation='vertical')
    cb1.set_label('Libration Frequency [Hz]')

    fig.add_axes(ax_cb)

# ax.set_xlim(0, final_time)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Phase Modulation [rad]')
if len(yticks):
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_yticks([], minor=True)
    ax.set_yticklabels([], minor=True)
fig.tight_layout()

vline = ax.axvline(0, color='k', zorder=4)
hline = ax.axhline(0, color='k', zorder=4)

my_colors = []
for i, (tvec, lib, amp, lib_freq, phi_dg) in enumerate(all_amp):
    lib_freq = lib_freqs[i]
    if color_with_libration:
        color = bu.get_single_color(lib_freq, cmap='plasma', \
                                    vmin=vmin, vmax=vmax)
    else:
        color = 'C0'
    my_colors.append(color)
    t0 = times[i]
    # ax.plot(tvec+t0, amp, color=color, zorder=2)
    # if not plot_just_amp:
    ax.plot(tvec+t0, lib, alpha=1.0, color=color, zorder=1)
    ax.plot(tvec+t0, amp, ls=':', alpha=1.0, color='k', zorder=2)

if plot_ringdown_fit:
    if ringdown_fit[2] >= 1e2:
        label = '$\\tau = {:d} \\pm {:0.2g}$ s'\
                    .format(int(ringdown_fit[2]), ringdown_unc[2])
    else:
        label = '$\\tau = {:0.3g} \\pm {:0.2g}$ s'\
                    .format(bu.round_sig(ringdown_fit[2], 3), ringdown_unc[2])
    ax.plot(plot_x, fit_func(plot_x, *ringdown_fit), ls='--', color='r', lw=3, \
            zorder=3, label=label)

    if meas_phi_dg != 0.0:
        ax.plot([0], [0], color='w', alpha=0.0, label='$g_d = {:.2g}$'.format(meas_phi_dg))

    ax.legend(loc='upper right')


if save_sequence:

    for i, t0 in enumerate(times):
        figname = os.path.join(fig_basepath, 'image_{:04d}.png'.format(i))
        if i == 0:
            bu.make_all_pardirs(figname)

        vline.remove()
        vline = ax.axvline(t0, color='k', zorder=4)
        if color_with_libration:
            hline.remove()
            hline = ax.axhline(freq_to_yval(lib_freqs[i]), xmin=0.975, \
                               xmax=1.0, color=my_colors[i], zorder=4, lw=3)
        elif i == 0:
            hline.remove()

        fig.canvas.draw_idle()
        fig.savefig(figname)
        print('Saved:  image_{:04d}.png'.format(i))

else:
    vline.remove()
    hline.remove()

    figname = os.path.join(plot_base, meas+'_ringdown_gd-{:.2g}.svg'.format(meas_phi_dg))
    bu.make_all_pardirs(figname)
    fig.savefig(figname)

    if show:
        plt.show()














