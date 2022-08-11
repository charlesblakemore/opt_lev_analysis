import os, time
import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as opti
import scipy.signal as signal

np.random.seed(12345)

plot_raw_dat = False
plot_phase = False
plot_sideband_fit = False

fc = 100000.0
wfc = 2.0*np.pi*fc
bandwidth = 1000.0
high_pass = 10.0

tabor_mon_fac = 100

base_path = '/daq2/20190626/bead1/spinning/wobble/wobble_slow_after-highp_later/'
base_save_path = '/processed_data/spinning/wobble/20190626/after-highp_slow_later/'

# base_path = '/daq2/20190626/bead1/spinning/pramp/SF6/wobble_3/'
# base_save_path = '/processed_data/spinning/wobble/20190626/SF6_pramp_3/'

paths = []
save_paths = []
for root, dirnames, filenames in os.walk(base_path):
    for dirname in dirnames:
        paths.append(base_path + dirname)
        save_paths.append(base_save_path + dirname + '.npy')
bu.make_all_pardirs(save_paths[0])
npaths = len(paths)

save = True
load = False

#####################################################

mbead = 85.0e-15 # convert picograms to kg
rhobead = 1550.0 # kg/m^3

rbead = ( (mbead / rhobead) / ((4.0/3.0)*np.pi) )**(1.0/3.0)
Ibead = 0.4 * mbead * rbead**2

def sqrt(x, A, x0, b):
    return A * np.sqrt(x-x0) #+ b

def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0*(x-mu)**2 / (2.0*sigma**2)) + c

def sine(x, A, f, phi, c):
    return A * np.sin(2*np.pi*f*x + phi) + c

def simple_pow(x, A, pow):
    return A * (x**pow)


all_data = []
for pathind, path in enumerate(paths):
    if load:
        continue
    files, lengths = bu.find_all_fnames(path, sort_time=True)

    fobj = hsDat(files[0])
    nsamp = fobj.attribs["nsamp"]
    fsamp = fobj.attribs["fsamp"]

    time_vec = np.arange(nsamp) * (1.0 / fsamp)
    freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

    upper1 = (2.0 / fsamp) * (fc + 0.5 * bandwidth)
    lower1 = (2.0 / fsamp) * (fc - 0.5 * bandwidth)

    upper2 = (2.0 / fsamp) * (0.5*fc + 0.25 * bandwidth)
    lower2 = (2.0 / fsamp) * (0.5*fc - 0.25 * bandwidth)

    notch = (2.0 / fsamp) * (58.8)
    Q_notch = 10


    b1, a1 = signal.butter(3, [lower1, upper1], \
                           btype='bandpass')
    b2, a2 = signal.butter(3, [lower2, upper2], \
                           btype='bandpass')

    b3, a3 = signal.butter(3, (2.0/fsamp)*high_pass, btype='high')

    bn, an = signal.iirnotch(notch, Q_notch)

    field_strength = []
    field_err = []

    wobble_freq = []
    wobble_err = []

    nfiles = len(files)
    suffix = '%i / %i' % (pathind+1, npaths)

    for fileind, file in enumerate(files):
        bu.progress_bar(fileind, nfiles, suffix=suffix)

        start = time.time()
        fobj = hsDat(file)

        vperp = fobj.dat[:,0]
        elec3 = fobj.dat[:,1]

        vperp_filt = signal.filtfilt(b1, a1, vperp)
        elec3_filt = signal.filtfilt(b2, a2, elec3)

        p0 = [1, fc, 0, 0]
        # popt, pcov = opti.curve_fit(sine, time, vperp_filt, p0=p0)
        # true_fc = popt[1]

        true_fc = fc

        if plot_raw_dat:
            plt.plot(time_vec[:10000], vperp_filt[:10000])
            plt.figure()
            plt.loglog(freqs, np.abs(np.fft.rfft(vperp)))
            plt.loglog(freqs, np.abs(np.fft.rfft(vperp_filt)))
            plt.figure()
            plt.loglog(freqs, np.abs(np.fft.rfft(elec3)))
            plt.loglog(freqs, np.abs(np.fft.rfft(elec3_filt)))
            plt.show() 

        hilbert = signal.hilbert(vperp_filt)
        phase = np.unwrap(np.angle(hilbert)) - 2.0*np.pi*true_fc*time_vec
        phase = (phase + np.pi) % (2.0*np.pi) - np.pi
        phase = np.unwrap(phase)
        phase_mod = polynomial(phase, order=8, plot=False)
        phase_mod *= signal.tukey(len(phase), alpha=1e-3)
        #phase_mod = bu.polynomial(phase, order=1, plot=True) #- np.mean(phase)

        amp = np.abs(hilbert)

        phase_mod = phase

        phase_mod_filt = signal.filtfilt(b3, a3, phase_mod)
        phase_mod_filt_2 = signal.lfilter(bn, an, phase_mod_filt)
        #phase_mod_filt = phase_mod

        amp_asd = np.abs(np.fft.rfft(amp))
        phase_asd = np.abs(np.fft.rfft(phase_mod))
        phase_asd_filt = np.abs(np.fft.rfft(phase_mod_filt))
        phase_asd_filt_2 = np.abs(np.fft.rfft(phase_mod_filt_2))

        if plot_phase:
            plt.plot(freqs, phase_asd)
            plt.figure()
            plt.loglog(freqs, phase_asd)
            plt.loglog(freqs, phase_asd_filt)
            plt.loglog(freqs, phase_asd_filt_2)
            plt.show()

        max_ind = np.argmax(phase_asd_filt_2)
        max_freq = freqs[max_ind]

        p0 = [10000, max_freq, 2.0, 0]

        try:
            popt, pcov = opti.curve_fit(gauss, freqs[max_ind-20:max_ind+20], \
                                        phase_asd_filt_2[max_ind-20:max_ind+20], p0=p0)

            fit_max = popt[1]
            fit_std = popt[2]
        except:
            fit_max = max_freq
            fit_std = 10.0*(freqs[1]-freqs[0])
            popt = p0

        if plot_sideband_fit:
            plot_freqs = np.linspace(freqs[max_ind-10], freqs[max_ind+10], 100)
            plt.loglog(freqs, phase_asd_filt_2)
            plt.loglog(plot_freqs, gauss(plot_freqs, *popt))
            plt.show()

        if fit_max < 10:
            continue

        if len(wobble_freq):
            if (np.abs(fit_max - wobble_freq[-1]) / wobble_freq[-1]) > 0.1:
                # plt.loglog(freqs, phase_asd)
                # plt.loglog(freqs, phase_asd_filt)
                # plt.loglog(freqs, phase_asd_filt_2)
                # plt.show()
                continue

        wobble_freq.append(fit_max)
        wobble_err.append(fit_std)


        elec3_filt_fft = np.fft.rfft(elec3_filt)

        zeros = np.zeros(nsamp)
        voltage = np.array([zeros, zeros, zeros, elec3_filt, \
                   zeros, zeros, zeros, zeros])
        efield = bu.trap_efield(voltage*tabor_mon_fac)
        #efield_mag = np.linalg.norm(efield, axis=0)

        start_sine = time.time()
        max_ind = np.argmax(np.abs(elec3_filt_fft))
        freq_guess = freqs[max_ind]
        phase_guess = np.mean(np.angle(elec3_filt_fft[max_ind-2:max_ind+2]))
        amp_guess = np.sqrt(2) * np.std(efield[0])
        p0 = [amp_guess, freq_guess, phase_guess, 0]

        fit_ind = int(0.01 * len(time_vec))
        popt, pcov = opti.curve_fit(sine, time_vec[:fit_ind], efield[0][:fit_ind], p0=p0)
        amp_fit = popt[0]
        amp_err = np.sqrt(pcov[0,0])

        field_strength.append(2.0*amp_fit)
        field_err.append(np.sqrt(2)*amp_err)
        stop_sine = time.time()

    out_arr = np.array([field_strength, field_err, wobble_freq, wobble_err])

    if save:
        np.save(save_paths[pathind], out_arr)

    all_data.append(out_arr)


if load:
    for save_path in save_paths:
        field_strength, field_err, wobble_freq, wobble_err = np.load(save_path)
        arr = np.array([field_strength, field_err, wobble_freq, wobble_err])
        all_data.append(arr)


popt_arr = []
colors = bu.get_colormap(len(all_data), cmap='inferno')
for arrind, arr in enumerate(all_data):
    field_strength = arr[0]
    field_err = arr[1]
    wobble_freq = arr[2]
    wobble_err = arr[3]

    try:
        popt, pcov = opti.curve_fit(sqrt, field_strength, 2*np.pi*wobble_freq, \
                                    p0=[10,0,0], sigma=2*np.pi*wobble_err)
    except:
        continue

    popt_arr.append(popt)
    print()
    print(popt)
    print()

    plot_x = np.linspace(0, np.max(field_strength), 100)
    plot_x[0] = 1.0e-9 * plot_x[1]
    plot_y = sqrt(plot_x, *popt)

    plt.plot(plot_x, plot_y, '--', lw=2, color=colors[arrind])
    plt.errorbar(field_strength, 2*np.pi*wobble_freq, alpha=0.6, \
                 yerr=wobble_err, color=colors[arrind])

popt = np.mean(np.array(popt_arr), axis=0)
popt_err = np.std(np.array(popt_arr), axis=0)

# 1e-3 to account for 
d = (popt[0])**2 * Ibead
d_err = (popt_err[0])**2 * Ibead

plt.show()