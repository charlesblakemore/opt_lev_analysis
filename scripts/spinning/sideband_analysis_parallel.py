import os, time, itertools
import numpy as np
import matplotlib.pyplot as plt

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as opti
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed
ncore = 30

np.random.seed(12345)

plot_raw_dat = False
plot_demod = False
plot_phase = True
plot_sideband_fit = False

cleanup_outarr = False

# fc = 220000.0
# fc = 110000.0
# fc = 100000.0
# fspin = 30000
fspin = 25000.0
wspin = 2.0*np.pi*fspin
bandwidth = 6000.0
high_pass = 50.0
detrend = True
force_2pi_wrap = False

allowed_freqs = (150.0, 2000.0)

apply_notch = False
notch_nharm = 15
notch_q = 7.5
notch_init = 10.0
notch_range = 20.0

notch_freqs = []
notch_qs = []

### BAD SCIENCE ALERT!!!
###   adjustment of the noise color to help extract the libration
###   feature. Sometimes a ~(1 / f^k) spectrum results from low
###   low frequency drifts. This adjusts for that.
correct_noise_color = False
noise_color_power = 0.9

# Should probably measure these monitor factors
tabor_mon_fac = 100
tabor_mon_fac = 100 * (1.0 / 0.95)

# base_path = '/daq2/20190626/bead1/spinning/wobble/wobble_slow_after-highp_later/'
# base_save_path = '/processed_data/spinning/wobble/20190626/after-highp_slow_later/'

# base_path = '/data/old_trap/20190905/bead1/spinning/wobble/before_pramp/'
# base_save_path = '/data/old_trap_processed/spinning/wobble/20190905/before_pramp/'

# #date = '20190626'
# #date = '20190905'
# date = '20191017'
# #gases = ['He', 'N2', 'Ar', 'Kr', 'Xe', 'SF6']
# #gases = ['He', 'N2']
# gases = ['He']
# inds = [1, 2, 3]
# #inds = [1]#,2]

# path_dict = {}
# for gas in gases:
#     if gas not in list(path_dict.keys()):
#         path_dict[gas] = {}
#     for ind in inds:
#         #if ind not in path_dict[gas].keys():
#         #    path_dict[gas][ind] = []

#         base_path = '/data/old_trap/{:s}/bead1/spinning/pramp/{:s}/wobble_{:d}/'\
#                             .format(date, gas, ind)
#         base_save_path = '/data/old_trap_processed/spinning/wobble/{:s}/{:s}_pramp_{:d}/'\
#                             .format(date, gas, ind)

#         paths = []
#         save_paths = []
#         for root, dirnames, filenames in os.walk(base_path):
#             for dirname in dirnames:
#                 #if '0001' not in dirname:
#                 #    continue
#                 paths.append(base_path + dirname)
#                 save_paths.append(base_save_path + dirname + '.npy')
#         bu.make_all_pardirs(save_paths[0])
#         npaths = len(paths)
#         paths, save_paths = (list(t) for t in zip(*sorted(zip(paths, save_paths))))

#         path_dict[gas][ind] = (paths, save_paths)


# date = '20200322'

# base_path = '/data/old_trap/20200322/gbead1/spinning/wobble/50kHz_yz_1/'
# base_save_path = '/data/old_trap_processed/spinning/wobble/20200322/50kHz_yz_1/'

# date = '20200727'
# meas = 'wobble_slow_2/'
# date = '20200924'
# date = '20201030'
date = '20201113'
meas = 'dipole_meas'
# meas = 'dipole_meas/initial'

base = '/data/old_trap/{:s}/bead1/spinning/'.format(date)
base_path = os.path.join(base, meas)
invert_order = True

base_save_path = os.path.join('/data/old_trap_processed/spinning/wobble/', date, meas)

path_dict = {}
paths = []
save_paths = []
print(base_path)
for root, dirnames, filenames in os.walk(base_path):
    for dirname in dirnames:
        print(dirname)
        paths.append(os.path.join(base_path, dirname))
        save_paths.append(os.path.join(base_save_path, dirname + '.npy'))
npaths = len(paths)

# paths = [base_path]
# save_paths = [base_save_path]

paths, save_paths = (list(t) for t in zip(*sorted(zip(paths, save_paths))))

path_dict['XX'] = {}
path_dict['XX'][1] = (paths, save_paths)
gases = ['XX']
inds = [1]

save = True
load = False

#####################################################

if plot_raw_dat or plot_demod or plot_phase or plot_sideband_fit:
    ncore = 1

bu.make_all_pardirs(save_paths[0])

Ibead = bu.get_Ibead(date=date)
# Ibead = bu.get_Ibead(date=date, rhobead={'val': 1850.0, 'sterr': 0.0, 'syserr': 0.0})

def sqrt(x, A, x0, b):
    return A * np.sqrt(x-x0) #+ b

def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0*(x-mu)**2 / (2.0*sigma**2)) + c

def lorentzian(x, A, mu, gamma, c):
    return (A / np.pi) * (gamma**2 / ((x-mu)**2 + gamma**2)) + c

def sine(x, A, f, phi, c):
    return A * np.sin(2*np.pi*f*x + phi) + c

def simple_pow(x, A, pow):
    return A * (x**pow)


all_data = []
for meas in itertools.product(gases, inds):
    gas, ind = meas
    paths, save_paths = path_dict[gas][ind]


    for pathind, path in enumerate(paths):
        if load:
            continue
        files, lengths = bu.find_all_fnames(path, sort_time=True)
        if invert_order:
            files = files[::-1]

        fobj = bu.hsDat(files[0], load=True)
        nsamp = fobj.nsamp
        fsamp = fobj.fsamp

        time_vec = np.arange(nsamp) * (1.0 / fsamp)
        freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

        upper1 = (2.0 / fsamp) * (2.0*fspin + 0.5 * bandwidth)
        lower1 = (2.0 / fsamp) * (2.0*fspin - 0.5 * bandwidth)

        upper2 = (2.0 / fsamp) * (fspin + 0.25 * bandwidth)
        lower2 = (2.0 / fsamp) * (fspin - 0.25 * bandwidth)

        notch_fit_inds = np.abs(freqs - notch_init) < notch_range


        b1, a1 = signal.butter(3, [lower1, upper1], \
                               btype='bandpass')
        b2, a2 = signal.butter(3, [lower2, upper2], \
                               btype='bandpass')

        b3, a3 = signal.butter(3, (2.0/fsamp)*high_pass, btype='high')

        def proc_file(file):
            fobj = bu.hsDat(file, load=True)

            vperp = fobj.dat[:,0]
            elec3 = fobj.dat[:,1]

            elec3_filt = signal.filtfilt(b2, a2, elec3)

            if plot_raw_dat:
                fac = bu.fft_norm(nsamp, fsamp)
                plt.plot(time_vec[:10000], elec3[:10000])
                plt.figure()
                plt.plot(time_vec[:10000], vperp[:10000])
                plt.figure()
                plt.loglog(freqs, fac * np.abs(np.fft.rfft(vperp)))
                plt.figure()
                plt.loglog(freqs, fac * np.abs(np.fft.rfft(elec3)))
                plt.loglog(freqs, fac * np.abs(np.fft.rfft(elec3_filt)))
                plt.show() 

                input()

            inds = np.abs(freqs - fspin) < 200.0

            elec3_fft = np.fft.rfft(elec3)
            true_fspin = freqs[np.argmax(np.abs(elec3_fft))]

            amp, phase_mod = bu.demod(vperp, true_fspin, fsamp, plot=plot_demod, \
                                  filt=True, bandwidth=bandwidth, \
                                  notch_freqs=notch_freqs, notch_qs=notch_qs, \
                                  tukey=True, tukey_alpha=5.0e-4, \
                                  detrend=detrend, detrend_order=1, harmind=2.0, \
                                  force_2pi_wrap=force_2pi_wrap)

            phase_mod_filt = signal.filtfilt(b3, a3, phase_mod)
            #phase_mod_filt = phase_mod

            amp_asd = np.abs(np.fft.rfft(amp))
            phase_asd = np.abs(np.fft.rfft(phase_mod))
            phase_asd_filt = np.abs(np.fft.rfft(phase_mod_filt))

            # popt_n, pcov_n = opti.curve_fit(gauss, freqs[notch_fit_inds], \
            #                                     phase_asd_filt[notch_fit_inds], \
            #                                     p0=[10000, notch_init, 2, 0])


            if apply_notch:            
                notch = freqs[np.argmax(phase_asd_filt * notch_fit_inds)]
                notch_digital = (2.0 / fsamp) * (notch)

                for i in range(notch_nharm):
                    bn, an = signal.iirnotch(notch_digital*(i+1), notch_q)
                    phase_mod_filt = signal.lfilter(bn, an, phase_mod_filt)

            phase_asd_filt_2 = np.abs(np.fft.rfft(phase_mod_filt))

            if correct_noise_color:
                phase_asd = phase_asd * freqs**noise_color_power
                phase_asd_filt = phase_asd_filt * freqs**noise_color_power
                phase_asd_filt_2 = phase_asd_filt_2 * freqs**noise_color_power

            if plot_phase:
                plt.plot(freqs, phase_mod)
                plt.xlabel('Frequency [Hz]')
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
                popt, pcov = opti.curve_fit(lorentzian, freqs[max_ind-30:max_ind+30], \
                                            phase_asd_filt_2[max_ind-30:max_ind+30], p0=p0, \
                                            maxfev=10000)

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
                print(fit_max, fit_std)
                plt.show()

                input()

            # if fit_max < 10:
            #     return 

            # if len(wobble_freq):
            #     if (np.abs(fit_max - wobble_freq[-1]) / wobble_freq[-1]) > 0.1:
            #         # plt.loglog(freqs, phase_asd)
            #         # plt.loglog(freqs, phase_asd_filt)
            #         # plt.loglog(freqs, phase_asd_filt_2)
            #         # plt.show()
            #         return


            elec3_filt_fft = np.fft.rfft(elec3_filt)

            fit_ind = 100000
            short_freqs = np.fft.rfftfreq(fit_ind, d=1.0/fsamp)
            zeros = np.zeros(fit_ind)
            voltage = np.array([zeros, zeros, zeros, elec3_filt[:fit_ind], \
                       zeros, zeros, zeros, zeros])
            efield = bu.trap_efield(voltage*tabor_mon_fac, only_x=True)

            #efield_mag = np.linalg.norm(efield, axis=0)

            efield_asd = bu.fft_norm(fit_ind, fsamp) * np.abs(np.fft.rfft(efield[0]))

            # max_ind = np.argmax(np.abs(elec3_filt_fft))
            short_max_ind = np.argmax(efield_asd)
            # freq_guess = freqs[max_ind]
            # phase_guess = np.mean(np.angle(elec3_filt_fft[max_ind-2:max_ind+2]))
            # amp_guess = np.sqrt(2) * np.std(efield[0])
            # p0 = [amp_guess, freq_guess, phase_guess, 0]

            # popt_l, pcov_l = opti.curve_fit(lorentzian, short_freqs[short_max_ind-100:short_max_ind+100], \
            #                                 efield_asd[short_max_ind-100:short_max_ind+100], \
            #                                 p0=[amp_guess, freq_guess, 100, 0], maxfev=10000)

            # start_sine = time.time()
            # popt, pcov = opti.curve_fit(sine, time_vec[:fit_ind], efield[0], \
            #                                 sigma=0.01*efield[0], p0=p0)

            # print popt[0], popt_l[0] * (fsamp / fit_ind)

            #print popt[0], np.sqrt(pcov[0,0])
            # amp_fit = efield_asd[short_max_ind] * np.sqrt(2.0 * fsamp / fit_ind)
            amp_fit = np.sqrt(2) * np.std(efield[0])
            # plt.plot(efield[0])
            # plt.show()

            err_val = np.mean(np.array([efield_asd[short_max_ind-10:short_max_ind], \
                                efield_asd[short_max_ind+1:short_max_ind+11]]).flatten())
            amp_err = np.sqrt((err_val * fsamp / fit_ind)**2)# + (0.01*amp_fit)**2)
            # print amp_fit, amp_err
            # stop_sine = time.time()
            # print "Field sampling: ", stop_sine - start_sine

            return [2.0*amp_fit, np.sqrt(2)*amp_err, fit_max, fit_std]



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

        if cleanup_outarr:
            clean = False
            new_nfiles = nfiles
            print('Cleaning...', end=' ')
            while not clean:
                for i in range(new_nfiles):
                    if i == 0:
                        continue
                    if i == new_nfiles - 1:
                        clean = True
                        break
                    cond1 = (np.abs(out_arr[2][i] - out_arr[2][i-1]) / 
                                                np.max(out_arr[2][i-1:i+1])) > 0.2
                    cond2 = out_arr[3][i] > 2.0 * out_arr[2][i]
                    if cond1 or cond2:
                        out_arr_1 = out_arr[:,:i]
                        out_arr_2 = out_arr[:,i+1:]
                        out_arr = np.concatenate((out_arr_1, out_arr_2), axis=-1)
                        new_nfiles = len(out_arr[0])
                        print(i, end=' ')
                        break
            print()

        # plt.hist(out_arr[3] / out_arr[2])
        # plt.show()

        if save:
            print(('Saving: ', save_paths[pathind]))
            np.save(save_paths[pathind], out_arr)
            # print('Saving: ', save_path)
            # np.save(save_path, out_arr)

        all_data.append(out_arr)




if load:
    for save_path in save_paths:
        saved_arr = np.load(save_path)
        #field_strength, field_err, wobble_freq, wobble_err = np.load(save_path)
        #arr = np.array([field_strength, field_err, wobble_freq, wobble_err])
        #arr = saved_arr.T
        all_data.append(saved_arr)


popt_arr = []
colors = bu.get_color_map(len(all_data), cmap='inferno')
for arrind, arr in enumerate(all_data):
    field_strength = arr[0]
    field_err = arr[1]
    wobble_freq = arr[2]
    wobble_err = arr[3]

    # plt.scatter(np.arange(arr.shape[1]), field_strength)

    # plt.figure()
    plt.errorbar(field_strength, 2*np.pi*wobble_freq, alpha=0.6, \
                 yerr=wobble_err, color=colors[arrind])

    p0 = [10, 0, 0]
    try:
        popt, pcov = opti.curve_fit(sqrt, field_strength, 2*np.pi*wobble_freq, \
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

    plot_x = np.linspace(0, np.max(field_strength), 100)
    plot_x[0] = 1.0e-9 * plot_x[1]
    plot_y = sqrt(plot_x, *popt)

    plt.plot(plot_x, plot_y, '--', lw=2, color=colors[arrind])

popt = np.mean(np.array(popt_arr), axis=0)
popt_err = np.std(np.array(popt_arr), axis=0)

# 1e-3 to account for 
try:
    d = (popt[0])**2 * Ibead['val']
    d_err = (popt_err[0])**2 * Ibead['val']
except: 
    2+2

plt.show()