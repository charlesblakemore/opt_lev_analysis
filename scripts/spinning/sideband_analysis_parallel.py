import os, time, itertools
import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as opti
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed
ncore = 20

np.random.seed(12345)

plot_raw_dat = True
plot_phase = False
plot_sideband_fit = False

cleanup_outarr = True

fc = 100000.0
wfc = 2.0*np.pi*fc
bandwidth = 2000.0
high_pass = 50.0

notch_init = 80.0
notch_range = 30.0
notch_2harm = True

# Should probably measure these monitor factors
tabor_mon_fac = 100
#tabor_mon_fac = 100 * (1.0 / 0.95)

# base_path = '/daq2/20190626/bead1/spinning/wobble/wobble_slow_after-highp_later/'
# base_save_path = '/processed_data/spinning/wobble/20190626/after-highp_slow_later/'

# base_path = '/data/old_trap/20190905/bead1/spinning/wobble/before_pramp/'
# base_save_path = '/data/old_trap_processed/spinning/wobble/20190905/before_pramp/'

#date = '20190626'
#date = '20190905'
date = '20191017'
#gases = ['He', 'N2', 'Ar', 'Kr', 'Xe', 'SF6']
#gases = ['He', 'N2']
gases = ['He']
inds = [1, 2, 3]
#inds = [1]#,2]

path_dict = {}
for gas in gases:
    if gas not in list(path_dict.keys()):
        path_dict[gas] = {}
    for ind in inds:
        #if ind not in path_dict[gas].keys():
        #    path_dict[gas][ind] = []

        base_path = '/data/old_trap/{:s}/bead1/spinning/pramp/{:s}/wobble_{:d}/'\
                            .format(date, gas, ind)
        base_save_path = '/data/old_trap_processed/spinning/wobble/{:s}/{:s}_pramp_{:d}/'\
                            .format(date, gas, ind)

        paths = []
        save_paths = []
        for root, dirnames, filenames in os.walk(base_path):
            for dirname in dirnames:
                #if '0001' not in dirname:
                #    continue
                paths.append(base_path + dirname)
                save_paths.append(base_save_path + dirname + '.npy')
        bu.make_all_pardirs(save_paths[0])
        npaths = len(paths)
        paths, save_paths = (list(t) for t in zip(*sorted(zip(paths, save_paths))))

        path_dict[gas][ind] = (paths, save_paths)



# base_path = '/data/old_trap/20191010/bead1/spinning/wobble/init/'
# base_save_path = '/data/old_trap_processed/spinning/wobble/20191010/init/'

# paths = []
# save_paths = []
# for root, dirnames, filenames in os.walk(base_path):
#     for dirname in dirnames:
#         paths.append(base_path + dirname)
#         save_paths.append(base_save_path + dirname + '.npy')
# bu.make_all_pardirs(save_paths[0])
# npaths = len(paths)
# paths, save_paths = (list(t) for t in zip(*sorted(zip(paths, save_paths))))

# path_dict['XX'] = {}
# path_dict['XX'][1] = (paths, save_paths)
# gases = ['XX']
# inds = [1]

save = True
load = False

#####################################################

if plot_raw_dat or plot_phase or plot_sideband_fit:
    ncore = 1


Ibead = bu.get_Ibead(date=date)

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

        fobj = hsDat(files[0])
        nsamp = fobj.attribs["nsamp"]
        fsamp = fobj.attribs["fsamp"]

        time_vec = np.arange(nsamp) * (1.0 / fsamp)
        freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

        upper1 = (2.0 / fsamp) * (fc + 0.5 * bandwidth)
        lower1 = (2.0 / fsamp) * (fc - 0.5 * bandwidth)

        upper2 = (2.0 / fsamp) * (0.5*fc + 0.25 * bandwidth)
        lower2 = (2.0 / fsamp) * (0.5*fc - 0.25 * bandwidth)
        Q_notch = 10

        notch_fit_inds = np.abs(freqs - notch_init) < notch_range


        b1, a1 = signal.butter(3, [lower1, upper1], \
                               btype='bandpass')
        b2, a2 = signal.butter(3, [lower2, upper2], \
                               btype='bandpass')

        b3, a3 = signal.butter(3, (2.0/fsamp)*high_pass, btype='high')

        def proc_file(file):
            fobj = hsDat(file)

            vperp = fobj.dat[:,0]
            elec3 = fobj.dat[:,1]

            vperp_filt = signal.filtfilt(b1, a1, vperp)
            elec3_filt = signal.filtfilt(b2, a2, elec3)

            true_fc = fc

            if plot_raw_dat:
                fac = bu.fft_norm(nsamp, fsamp)
                plt.plot(time_vec[:10000], elec3[:10000])
                plt.figure()
                plt.plot(time_vec[:10000], vperp_filt[:10000])
                plt.figure()
                plt.loglog(freqs, fac * np.abs(np.fft.rfft(vperp)))
                plt.loglog(freqs, fac * np.abs(np.fft.rfft(vperp_filt)))
                plt.figure()
                plt.loglog(freqs, fac * np.abs(np.fft.rfft(elec3)))
                plt.loglog(freqs, fac * np.abs(np.fft.rfft(elec3_filt)))
                plt.show() 

            #start_hilbert = time.time()
            hilbert = signal.hilbert(vperp_filt)
            phase = np.unwrap(np.angle(hilbert)) - 2.0*np.pi*true_fc*time_vec
            #stop_hilbert = time.time()
            #print "Hilbert time: ", stop_hilbert - start_hilbert

            #start_polyfit = time.time()
            phase = (phase + np.pi) % (2.0*np.pi) - np.pi
            phase = np.unwrap(phase)
            phase_mod = polynomial(phase, order=3, plot=False)
            phase_mod *= signal.tukey(len(phase), alpha=1e-3)
            #phase_mod = signal.detrend(phase) * signal.tukey(len(phase), alpha=1e-3)
            #stop_polyfit = time.time()
            #print 'Poly fit: ', stop_polyfit - start_polyfit
            #phase_mod = bu.polynomial(phase, order=1, plot=True) #- np.mean(phase)

            amp = np.abs(hilbert)

            phase_mod = phase

            phase_mod_filt = signal.filtfilt(b3, a3, phase_mod)
            #phase_mod_filt = phase_mod

            amp_asd = np.abs(np.fft.rfft(amp))
            phase_asd = np.abs(np.fft.rfft(phase_mod))
            phase_asd_filt = np.abs(np.fft.rfft(phase_mod_filt))

            # popt_n, pcov_n = opti.curve_fit(gauss, freqs[notch_fit_inds], \
            #                                     phase_asd_filt[notch_fit_inds], \
            #                                     p0=[10000, notch_init, 2, 0])

            notch = freqs[np.argmax(phase_asd_filt * notch_fit_inds)]

            #start_filter_build = time.time()
            notch_digital = (2.0 / fsamp) * (notch)
            bn, an = signal.iirnotch(notch_digital, Q_notch)
            if notch_2harm:
                bn2, an2 = signal.iirnotch(2.0*notch_digital, Q_notch)
            #stop_filter_build = time.time()
            #print "Filter build: ", stop_filter_build - start_filter_build

            phase_mod_filt_2 = signal.lfilter(bn, an, phase_mod_filt)
            if notch_2harm:
                phase_mod_filt_2 = signal.lfilter(bn2, an2, phase_mod_filt_2   )
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

            if fit_max < 10:
                return 

            if len(wobble_freq):
                if (np.abs(fit_max - wobble_freq[-1]) / wobble_freq[-1]) > 0.1:
                    # plt.loglog(freqs, phase_asd)
                    # plt.loglog(freqs, phase_asd_filt)
                    # plt.loglog(freqs, phase_asd_filt_2)
                    # plt.show()
                    return


            elec3_filt_fft = np.fft.rfft(elec3_filt)

            fit_ind = 30000
            short_freqs = np.fft.rfftfreq(fit_ind, d=1.0/fsamp)
            zeros = np.zeros(fit_ind)
            voltage = np.array([zeros, zeros, zeros, elec3_filt[:fit_ind], \
                       zeros, zeros, zeros, zeros])
            efield = bu.trap_efield(voltage*tabor_mon_fac, only_x=True)

            #efield_mag = np.linalg.norm(efield, axis=0)

            efield_asd = bu.fft_norm(fit_ind, fsamp) * np.abs(np.fft.rfft(efield[0]))
            # plt.loglog(short_freqs, efield_asd)
            # plt.show()

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
            amp_fit = efield_asd[short_max_ind] * np.sqrt(2.0 * fsamp / fit_ind)

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


        # for fileind, file in enumerate(files):
        #     bu.progress_bar(fileind, nfiles, suffix=suffix)

        #     out = 

        #     field_strength.append(2.0*amp_fit)
        #     field_err.append(np.sqrt(2)*amp_err)

        #     wobble_freq.append(fit_max)
        #     wobble_err.append(fit_std)

        # out_arr = np.array([field_strength, field_err, wobble_freq, wobble_err])

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
                    cond1 = (np.abs(out_arr[2][i] - out_arr[2][i-1]) / out_arr[2][i]) > 0.1
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

    plt.errorbar(field_strength, 2*np.pi*wobble_freq, alpha=0.6, \
                 yerr=wobble_err, color=colors[arrind])

    try:
        popt, pcov = opti.curve_fit(sqrt, field_strength, 2*np.pi*wobble_freq, \
                                    p0=[10,0,0], sigma=2*np.pi*wobble_err)
    except:
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
d = (popt[0])**2 * Ibead['val']
d_err = (popt_err[0])**2 * Ibead['val']

plt.show()