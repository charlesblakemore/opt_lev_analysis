import os, time, itertools
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as optimize
import scipy.signal as signal
import scipy.constants as constants

from tqdm import tqdm
from joblib import Parallel, delayed
# ncore = 1
ncore = 25

plot_raw_dat = False
plot_demod = False
plot_phase = False
plot_sideband_fit = False
plot_efield_estimation = False
plot_final_result = True

# fc = 220000.0
# fc = 110000.0
# fc = 100000.0
# fspin = 30000   # For 20200727 data
fspin = 25000.0
wspin = 2.0*np.pi*fspin
bandwidth = 1000.0
high_pass = 50.0
detrend = True
force_2pi_wrap = False

allowed_freqs = (100.0, 500.0)

notch_freqs = []
notch_qs = []

### BAD SCIENCE ALERT!!!
###   adjustment of the noise color to help extract the libration
###   feature. Sometimes a ~(1 / f^k) spectrum results from low
###   frequency drifts. This adjusts for that problem by scaling 
###   the entire spectrum so the libration feature has a symmetric
###   background.
correct_noise_color = False
noise_color_power = 0.9

# Should probably measure these monitor factors
tabor_mon_fac = 100
tabor_mon_fac = 100 * (1.0 / 0.95)

# base_path = '/daq2/20190626/bead1/spinning/wobble/wobble_slow_after-highp_later/'
# base_save_path = '/processed_data/spinning/wobble/20190626/after-highp_slow_later/'

# base_path = '/data/old_trap/20190905/bead1/spinning/wobble/before_pramp/'
# base_save_path = '/data/old_trap_processed/spinning/wobble/20190905/before_pramp/'

# date = '20200322'

# base_path = '/data/old_trap/20200322/gbead1/spinning/wobble/50kHz_yz_1/'
# base_save_path = '/data/old_trap_processed/spinning/wobble/20200322/50kHz_yz_1/'

### Need to look this over a bit, but should only affect the 'cleanup' step that
### happens at the end to remove misidentified peaks
invert_order = True

# date = '20200727'
# meas_list = [\
#              # 'wobble_fast', \
#              # 'wobble_large-step_many-files', \
#              # 'wobble_slow', \
#              # 'wobble_slow_2', \
#              # 'adiabatic_wobble', \
#              'wobble_slow_after'
#             ]

# date = '20200924'
# meas_list = [\
#              'dipole_meas/initial', \
#             ]

date = '20230327'
bead = 'bead1'
meas_list = [\
             'dipole_meas/initial_zadj', \
            ]

base = f'/data/old_trap/{date}/{bead}/spinning/'
savebase = '/data/old_trap_processed/spinning/wobble/'

paths = []
save_paths = []
for meas in meas_list:
    base_path = os.path.join(base, meas)
    base_save_path = os.path.join(savebase, date, meas)
    for root, dirnames, filenames in os.walk(base_path):
        if len(dirnames):
            for dirname in dirnames:
                paths.append(os.path.join(base_path, dirname))
                save_paths.append(os.path.join(base_save_path, dirname + '.npy'))
        elif len(filenames) and root not in paths:
            paths.append(base_path)
            save_paths.append(os.path.join(base_save_path, 'wobble_0.npy'))
npaths = len(paths)

paths, save_paths = (list(t) for t in zip(*sorted(zip(paths, save_paths))))

save = False
load = False

timer = False

# print(path_dict)
# input()



##########################################################
##########################################################
##########################################################


if plot_raw_dat or plot_demod or plot_phase or plot_sideband_fit \
        or plot_efield_estimation:
    ncore = 1

bu.make_all_pardirs(save_paths[0])

# Ibead = bu.get_Ibead(date=date)
Ibead = bu.get_Ibead(date=date, rhobead={'val': 1850.0, 'sterr': 0.0, 'syserr': 0.0})

def sqrt(x, A, x0, b):
    return A * np.sqrt(x-x0) + b

def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0*(x-mu)**2 / (2.0*sigma**2)) + c

def lorentzian(x, A, mu, gamma, c):
    return (A / np.pi) * (gamma**2 / ((x-mu)**2 + gamma**2)) + c

def sine(x, A, f, phi, c):
    return A * np.sin(2*np.pi*f*x + phi) + c

def simple_pow(x, A, pow):
    return A * (x**pow)


all_data = []


for pathind, path in enumerate(paths):
    if load:
        continue
    files, lengths = bu.find_all_fnames(path, sort_time=True, \
                                        skip_subdirectories=True)
    if invert_order:
        files = files[::-1]

    # print(files)
    # input()

    fobj = bu.hsDat(files[0], load=True)
    nsamp = fobj.nsamp
    fsamp = fobj.fsamp

    time_vec = np.arange(nsamp) * (1.0 / fsamp)
    freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

    upper = fspin + 0.25 * bandwidth
    lower = fspin - 0.25 * bandwidth

    sos = signal.butter(3, [lower, upper], fs=fsamp, \
                        btype='bandpass', output='sos')

    sos_hp = signal.butter(3, high_pass, fs=fsamp, \
                           btype='high', output='sos')


    def proc_file(file):
        proc_file_start = time.time()
        fobj = bu.hsDat(file, load=True)

        vperp = fobj.dat[:,0]
        elec3 = fobj.dat[:,1]

        elec3_filt = signal.sosfiltfilt(sos, elec3)

        if plot_raw_dat:
            fac = bu.fft_norm(nsamp, fsamp)
            plt.plot(time_vec[:10000], elec3[:10000])
            plt.plot(time_vec[:10000], elec3_filt[:10000])
            plt.figure()
            plt.plot(time_vec[:10000], vperp[:10000])
            plt.figure()
            plt.loglog(freqs, fac * np.abs(np.fft.rfft(vperp)))
            plt.figure()
            plt.loglog(freqs, fac * np.abs(np.fft.rfft(elec3)))
            plt.loglog(freqs, fac * np.abs(np.fft.rfft(elec3_filt)))
            plt.show() 

            input()

        inds = np.abs(freqs - fspin) < 100.0

        elec3_fft = np.fft.rfft(elec3_filt)[inds]
        weights = np.abs(elec3_fft)**2
        # true_fspin = freqs[inds][np.argmax(weights)]
        true_fspin = np.sum(freqs[inds] * weights) / np.sum(weights)

        demod_start = time.time()
        amp, phase_mod, demod_debug = \
                bu.demod(vperp, true_fspin, fsamp, plot=plot_demod, \
                         filt=True, bandwidth=bandwidth, \
                         notch_freqs=notch_freqs, notch_qs=notch_qs, \
                         tukey=True, tukey_alpha=5.0e-4, \
                         detrend=detrend, harmind=2.0, \
                         force_2pi_wrap=force_2pi_wrap, debug=True)
        demod_end = time.time()

        ### Add back the residual frequency offset that sometimes remains.
        ### This is observed as a linear trend in the demodulated phase
        ### and if we detrend, then that value naturally comes out and is
        ### used to infer the actual spinning frequency
        # true_fspin += demod_debug['residual_freq']

        extraction_start = time.time()
        phase_mod_filt = signal.sosfiltfilt(sos_hp, phase_mod)

        amp_asd = np.abs(np.fft.rfft(amp))
        phase_asd = np.abs(np.fft.rfft(phase_mod))
        phase_asd_filt = np.abs(np.fft.rfft(phase_mod_filt))

        phase_asd_filt_2 = np.abs(np.fft.rfft(phase_mod_filt))

        if correct_noise_color:
            phase_asd = phase_asd * freqs**noise_color_power
            phase_asd_filt = phase_asd_filt * freqs**noise_color_power
            phase_asd_filt_2 = phase_asd_filt_2 * freqs**noise_color_power

        if plot_phase:
            plt.plot(np.arange(len(phase_mod))/fsamp, phase_mod)
            plt.xlabel('Time [s]')
            plt.ylabel('Phase [rad]')
            plt.tight_layout()

            plt.figure()
            plt.loglog(freqs, phase_asd)
            plt.loglog(freqs, phase_asd_filt)
            plt.loglog(freqs, phase_asd_filt_2)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Phase ASD [arb]')
            plt.tight_layout()

            plt.show()

            input()

        freq_mask = (freqs > allowed_freqs[0]) * (freqs < allowed_freqs[1])

        max_ind = np.argmax(phase_asd_filt_2 * freq_mask)
        max_freq = freqs[max_ind]

        p0 = [10000, max_freq, 5.0, 0]

        try:
            popt, pcov = \
                optimize.curve_fit(lorentzian, freqs[max_ind-30:max_ind+30], \
                                   phase_asd_filt_2[max_ind-30:max_ind+30], \
                                   p0=p0, maxfev=10000)

            fit_max = popt[1]
            fit_std = np.abs(popt[2])
        except:
            print('bad fit...')
            fit_max = max_freq
            fit_std = 5.0*(freqs[1]-freqs[0])
            popt = p0

        if plot_sideband_fit:
            plot_freqs = np.linspace(freqs[max_ind-30], freqs[max_ind+30], 100)
            plt.loglog(freqs, phase_asd_filt_2)
            plt.loglog(plot_freqs, lorentzian(plot_freqs, *popt))
            # print(fit_max, fit_std)
            plt.show()

            input()
        extraction_end = time.time()

        # if fit_max < 10:
        #     return 

        # if len(wobble_freq):
        #     if (np.abs(fit_max - wobble_freq[-1]) / wobble_freq[-1]) > 0.1:
        #         # plt.loglog(freqs, phase_asd)
        #         # plt.loglog(freqs, phase_asd_filt)
        #         # plt.loglog(freqs, phase_asd_filt_2)
        #         # plt.show()
        #         return


        drive_data_start = time.time()
        elec3_filt_fft = np.fft.rfft(elec3_filt)

        fit_ind = 100000
        zeros = np.zeros(fit_ind)
        voltage = np.array([zeros, zeros, zeros, elec3_filt[:fit_ind], \
                   zeros, zeros, zeros, zeros])
        efield = bu.trap_efield(voltage*tabor_mon_fac, only_x=True)

        efield_amp, efield_unc, _, _ = \
            bu.get_sine_amp_phase(efield[0], int_band=1000.0/fsamp, \
                                  plot=plot_efield_estimation, \
                                  freq=true_fspin/fsamp, incoherent=True)

        drive_data_end = time.time()

        if timer:
            print()
            print('     Demod : {:0.4f}'.format(demod_end - demod_start))
            print('Extraction : {:0.4f}'.format(extraction_end - extraction_start))
            print('Drive data : {:0.4f}'.format(drive_data_end - drive_data_start))
            print('     TOTAL : {:0.4f}'.format(drive_data_end - proc_file_start))



        return [2.0*efield_amp, np.sqrt(2)*efield_unc, fit_max, fit_std]



    field_strength = []
    field_err = []

    wobble_freq = []
    wobble_err = []

    nfiles = len(files)
    suffix = '%i / %i' % (pathind+1, npaths)

    # print ncore
    results = Parallel(n_jobs=ncore)(delayed(proc_file)(file) for file in tqdm(files))

    out_arr = np.array(results)
    out_arr = out_arr.T

    if save:
        print()
        print('Saving processed data: ')
        print('    {:s}'.format(save_paths[pathind]))
        print()
        bu.make_all_pardirs(save_paths[pathind])
        np.save(save_paths[pathind], out_arr)

        # print('Saving: ', save_path)
        # np.save(save_path, out_arr)

    all_data.append(out_arr)




if load:
    for save_path in save_paths:
        saved_arr = np.load(save_path)
        all_data.append(saved_arr)

if plot_final_result:

    # plt.figure(constrained_layout=True)

    popt_arr = []
    colors = bu.get_colormap(len(all_data), cmap='inferno')
    for arrind, arr in enumerate(all_data):
        field_strength = arr[0]
        field_err = arr[1]
        wobble_freq = arr[2]
        wobble_err = arr[3]

        # plt.scatter(np.arange(arr.shape[1]), field_strength)

        # plt.figure()
        plt.errorbar(field_strength*1e-3, wobble_freq, fmt='o', \
                     ms=5, alpha=0.6, \
                     yerr=wobble_err, color=colors[arrind])

        # fit_func = lambda x, A, x0, b: sqrt(x, A, x0, b)
        # p0 = [10, 0, 0]

        fit_func = lambda x, A: sqrt(x, A, 0, 0)
        p0 = [10]

        try:
            popt, pcov = \
                optimize.curve_fit(fit_func, field_strength, 2*np.pi*wobble_freq, \
                                   p0=p0, sigma=2*np.pi*wobble_err)
        except:
            popt = p0
            continue

        popt_arr.append(popt)
        # print('')
        # print(popt)
        # print((1.0 / (len(field_strength) - 1)) * \
        #     np.sum((2*np.pi*wobble_freq - sqrt(field_strength, *popt))**2 / (2*np.pi*wobble_err)**2))
        # print('')

        plot_x = np.linspace(0, np.max(field_strength), 100) * 1e-3
        plot_x[0] = 1.0e-9 * plot_x[1]
        plot_y = fit_func(plot_x*1e3, *popt) / (2.0 * np.pi)

        plt.plot(plot_x, plot_y, '--', lw=2, color=colors[arrind])

    popt = np.mean(np.array(popt_arr), axis=0)
    popt_err = np.std(np.array(popt_arr), axis=0)

    # try:
    d = (popt[0])**2 * Ibead['val']
    d_err = (popt_err[0])**2 * Ibead['val']

    d_units = constants.e * 1e-6

    print('Dipole [e*um]:')
    print(d/d_units, d_err/d_units)
    # except: 
    #     2+2

    plt.xlabel('Field Strength [kV/m]')
    plt.ylabel('Libration Frequency [Hz]')

    plt.tight_layout()

    plt.show()