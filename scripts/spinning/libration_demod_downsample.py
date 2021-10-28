import os, sys, time, itertools, re, warnings
import numpy as np
import dill as pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as optimize
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed
ncore = 1
# ncore = 30

warnings.filterwarnings('ignore')



#############################
### Which data to analyze ###
#############################

# try:
#     meas_ind = int(sys.argv[1])
# except:
#     meas_ind = 0

try:
    volt_level = int(sys.argv[1])
except:
    volt_level = 1

try:
    trial_ind = int(sys.argv[2])
except:
    trial_ind = 0


date = '20200727'
# date = '20200924'
# date = '20201030'

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


# if meas_ind:
#     meas_ind += 1
#     meas = 'dds_phase_impulse_3Vpp_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
#     # meas = 'dds_phase_impulse_3Vpp_lower_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
#     # meas = 'dds_phase_impulse_3Vpp_low_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
#     # meas = 'dds_phase_impulse_3Vpp_mid_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
#     # meas = 'dds_phase_impulse_3Vpp_high_dg_{:d}/trial_{:04d}'.format(meas_ind, trial_ind)
# else:
#     meas = 'dds_phase_impulse_3Vpp/trial_{:04d}'.format(trial_ind)
#     # meas = 'dds_phase_impulse_3Vpp_lower_dg/trial_{:04d}'.format(trial_ind)
#     # meas = 'dds_phase_impulse_3Vpp_low_dg/trial_{:04d}'.format(trial_ind)
#     # meas = 'dds_phase_impulse_3Vpp_mid_dg/trial_{:04d}'.format(trial_ind)
#     # meas = 'dds_phase_impulse_3Vpp_high_dg/trial_{:04d}'.format(trial_ind)
# # file_inds = (8, 100)
# # file_inds = (8, 90)
# file_inds = (8, 50)
# # file_inds = (8, 25)

meas = 'dds_phase_impulse_{:d}Vpp/trial_{:04d}'.format(volt_level, trial_ind)
file_inds = (8, 100)

dir_name = os.path.join(base, meas)
file_step = 1

save_sequence = False
color_with_libration = False
plot_base = '/home/cblakemore/plots/{:s}/spinning/'.format(date)
# fig_basepath = os.path.join(plot_base, 'phase_impulse_+90deg/ringdown_amp/')
# fig_basepath = os.path.join(plot_base, 'phase_impulse_-90deg/ringdown_amp/')
# fig_basepath = os.path.join(plot_base, 'arb_phase_impulse_+90deg/ringdown_amp/')
fig_basepath = os.path.join(plot_base, meas, 'ringdown_amp')


processed_base = '/data/old_trap_processed/spinning/{:s}/'.format(date)
save_ringdown = False
# ringdown_dict_path = '/data/old_trap_processed/spinning/20200727/arb_libration_ringdowns.p'
ringdown_data_path = os.path.join(processed_base, 'dds_libration_ringdowns_manyVpp.p')


dipole = bu.get_dipole(date, substrs=[])
rhobead = bu.rhobead['bangs5']

libration_guess = 0.0
# libration_guess = 1335.8
# libration_guess = 1385.0
# libration_guess = 1298.0
# libration_guess = 297.9   # 20200924 1Vpp
# libration_guess = 512.0   # 20200924 3Vpp

### Carrier filter constants
# fspin = 19000
fspin = 25000.0
fspin = 30000.0
wspin = 2.0*np.pi*fspin
bandwidth = 10000.0

# libration_fit_band = []
# libration_filt_band = [1000.0, 1450.0]
# libration_filt_band = [900.0, 1350.0]
# libration_filt_band = [700.0, 1000.0]
# libration_filt_band = [175.0, 400.0]   # 20200924 1Vpp
# libration_filt_band = [350.0, 600.0]   # 20200924 3Vpp
libration_bandwidth = 400

notch_freqs = [49020.3]
notch_qs = [10000.0]

# colorbar_limits = []
# colorbar_limits = [1200, 1400]
colorbar_limits = [900, 1350]

### Boolean flags for various sorts of plotting (used for debugging usually)
plot_carrier_demod = False
plot_libration_demod = False
plot_downsample = True

plot_lib_amp = False

### Should probably measure these monitor factors
# tabor_mon_fac = 100
# tabor_mon_fac = 100.0 * (1.0 / 0.95)
tabor_mon_fac = 100.0 * (53000.0 / 50000.0)

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
initial_offset = 0.0

plot_rebin = False
plot_ringdown_fit = True
close_xlim = True
save_ringdown_figs = False
show = True



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
weights = np.abs(elec3_fft[inds])**2
true_fspin = np.sum(full_freqs[inds] * weights) / np.sum(weights)


if not libration_guess:

    try:
        elec3_cut = tabor_mon_fac * elec3[:int(fsamp)]
        zeros = np.zeros_like(elec3_cut)
        voltages = [zeros, zeros, zeros, elec3_cut, zeros, zeros, zeros, zeros]
        efield = bu.trap_efield(voltages, only_x=True)[0]

        ### Factor of 2.0 for two opposing electrodes, only one of which is
        ### digitized due to the (sampling rate limitations
        efield_amp, _ = bu.get_sine_amp_phase(efield)
        efield_amp *= 2.0

        Ibead = bu.get_Ibead(date=date, rhobead=rhobead)['val']

        libration_guess = np.sqrt(efield_amp * dipole['val'] / Ibead) / (2.0 * np.pi)

    except:

        amp, phase_mod = bu.demod(vperp, true_fspin, fsamp, plot=plot_carrier_demod, \
                                  filt=True, bandwidth=bandwidth,
                                  tukey=True, tukey_alpha=5.0e-4, \
                                  detrend=True, harmind=2.0)

        phase_mod_fft = np.fft.rfft(phase_mod) * fac

        fig, ax = plt.subplots(1,1)
        ax.loglog(full_freqs, np.abs(phase_mod_fft))
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Phase ASD [rad/$\\sqrt{\\rm Hz}$]')
        ax.set_title('Identify Libration Frequency')
        fig.tight_layout()
        plt.show()

        libration_guess = float( input('Libration guess: ') )



# libration_filt_band = [np.min([125.0, libration_guess-200.0]), libration_guess+100.0]


def proc_file(file):

    print(file)
    input()

    fobj = bu.hsDat(file, load=True)

    vperp = fobj.dat[:,0]
    elec3 = fobj.dat[:,1]

    try:
        phi_dg = fobj.attribs['phi_dg']
    except:
        phi_dg = 0.0

    inds = np.abs(full_freqs - fspin) < 200.0

    cut = 1e5
    zeros = np.zeros_like(elec3[:cut])
    voltages = [zeros, zeros, zeros, elec3[:cut], zeros, zeros, zeros, zeros]
    efield = bu.trap_efield(voltages, only_x=True)[0]
    drive_amp, drive_phase = bu.get_sine_amp_phase(efield)
    drive_amp *= 2.0

    elec3_fft = np.fft.rfft(elec3)
    true_fspin = np.average(full_freqs[inds], weights=np.abs(elec3_fft[inds])**2)


    carrier_amp, carrier_phase_mod = \
            bu.demod(vperp, true_fspin, fsamp, plot=plot_carrier_demod, \
                     filt=True, bandwidth=bandwidth,
                     notch_freqs=notch_freqs, notch_qs=notch_qs, \
                     tukey=True, tukey_alpha=5.0e-4, \
                     detrend=True, harmind=2.0)


    sos = signal.butter(3, libration_filt_band, btype='bandpass', \
                        fs=fsamp, output='sos')
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
                     detrend=False, harmind=1.0)

    libration_ds, time_vec_ds = \
            signal.resample(carrier_phase_mod_filt, t=time_vec, num=out_nsamp)
    libration_amp_ds, time_vec_ds = \
            signal.resample(libration_amp, t=time_vec, num=out_nsamp)

    libration_ds = libration_ds[out_cut:int(-1*out_cut)]
    libration_amp_ds = libration_amp_ds[out_cut:int(-1*out_cut)]
    time_vec_ds = time_vec_ds[out_cut:int(-1*out_cut)]

    if plot_downsample:
        plt.plot(time_vec, carrier_phase_mod_filt, color='C0', \
                 lw=1, label='Original')
        plt.plot(time_vec_ds, libration_ds, color='C0', lw=3, \
                 ls='--', label='Downsampled')
        plt.plot(time_vec, libration_amp, color='C1', lw=1)
        plt.plot(time_vec_ds, libration_amp_ds, color='C1', lw=3, ls='--')
        plt.legend()
        plt.show()

        input()

    return (time_vec_ds, libration_ds, libration_amp_ds, \
                true_libration_freq, phi_dg, drive_amp, file)




all_amp = Parallel(n_jobs=ncore)(delayed(proc_file)(file) for file in tqdm(files))



for entry in all_amp:
    time_vec, lib_vec, lib_amp_vec, lib_freq, phi_dg, drive_amp, filename = entry









