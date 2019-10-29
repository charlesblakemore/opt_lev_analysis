import os, time, sys
import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
from iminuit import Minuit, describe

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet
import dill as pickle

import scipy.optimize as opti
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed


plt.rcParams.update({'font.size': 14})

#f_rot = 210000.0
f_rot = 110000.0
#f_rot = 50000.0
bandwidth = 500

debug = False
plot_raw_dat = False
plot_phase = False


high_pass = 10.0


# path = '/data/old_trap/0190626/bead1/spinning/ringdown/50kHz_ringdown'
# after_path = '/data/old_trap/20190626/bead1/spinning/ringdown/after_pramp'

# paths = [path, path+'2', after_path]

# save_base = '/data/old_trap_processed/spinning/ringdown/20190626/'

date = '20191017'
base_path = '/data/old_trap/{:s}/bead1/spinning/ringdown/'.format(date)
#base_path = '/data/old_trap/{:s}/bead1/spinning/ringdown_manual/'.format(date)

save_base = '/data/old_trap_processed/spinning/ringdown/{:s}/'.format(date)
#save_base = '/data/old_trap_processed/spinning/ringdown_manual/{:s}/'.format(date)

paths = [#base_path + '110kHz_start_1', \
         #base_path + '110kHz_start_2', \
         #base_path + '110kHz_start_3', \
         #base_path + '110kHz_start_4', \
         base_path + '110kHz_start_5', \
         base_path + '110kHz_start_6', \
         ]



# base_path = '/daq2/20190626/bead1/spinning/wobble/wobble_slow_after-highp_later/'
# base_save_path = '/processed_data/spinning/wobble/20190626/after-highp_slow_later/'

# paths = []
# save_paths = []
# for root, dirnames, filenames in os.walk(base_path):
#     for dirname in dirnames:
#         paths.append(base_path + dirname)
#         save_paths.append(base_save_path + dirname + '.npy')
# bu.make_all_pardirs(save_paths[0])
npaths = len(paths)

save = True
load = False
no_fits = True

lin_fit_seconds = 10
exp_fit_seconds = 500


############################

def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0*(x-mu)**2 / (2.0*sigma**2)) + c

def ngauss(x, A, mu, sigma, c, n=2):
    return A * np.exp(-1.0*np.abs(x-mu)**n / (2.0*sigma**n)) + c

def lorentzian(x, A, mu, gamma, c):
    return A * (gamma**2 / ((x-mu)**2 + gamma**2)) + c

def line(x, a, b):
    return a * x + b

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b, c, d):
    return a * np.exp( b * (x + c) ) + d


def rebin(a, *args):
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape)/np.asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.mean(%d)'%(i+1) for i in range(lenShape)]
    #print ''.join(evList)

    evList2 = ['a.reshape('] + \
              ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
              [')'] + ['.std(%d)'%(i+1) for i in range(lenShape)]
   # print ''.join(evList2)

    return eval(''.join(evList)), eval(''.join(evList2))



# times = np.arange(100000)
# vec = np.random.randn(100000)
# vec_ds, vec_err_ds = rebin(vec, 1000)
# vec_err_ds *= np.sqrt(1000.0/100000.0)
# times_ds, times_err_ds = rebin(times, 1000)
# #plt.plot(times, vec)
# plt.errorbar(times_ds, vec_ds, yerr=vec_err_ds)
# plt.show()

# raw_input()


def proc_dir(path):
    fc = 2.0*f_rot
    wc = 2.0*np.pi*fc

    strs = path.split('/')
    if len(strs[-1]) == 0:
        dirname = strs[-2]
    else:
        dirname = strs[-1]

    out_f = save_base + dirname
    bu.make_all_pardirs(out_f)

    # if load:
    #     outdict[out_f] = pickle.load(open(out_f + '_all.p', 'rb'))
    #     all_time = outdict[out_f]['all_time']
    #     all_freq = outdict[out_f]['all_freq']
    #     all_freq_err = outdict[out_f]['all_freq_err']
    #     plt.errorbar(all_time.flatten(), all_freq.flatten(), yerr=all_freq_err.flatten())
    #     plt.show()
    #     continue

    files, lengths = bu.find_all_fnames(path, sort_time=True)

    #files = files[:1000]

    fobj = hsDat(files[0])
    nsamp = fobj.attribs["nsamp"]
    fsamp = fobj.attribs["fsamp"]
    t0 = fobj.attribs["time"]

    time_vec = np.arange(nsamp) * (1.0 / fsamp)
    freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

    # upper1 = (2.0 / fsamp) * (fc + 0.5 * bandwidth)
    # lower1 = (2.0 / fsamp) * (fc - 0.5 * bandwidth)
    upper1 = (2.0 / fsamp) * (fc + 5 * bandwidth)
    lower1 = (2.0 / fsamp) * (fc - 5 * bandwidth)
    fc_init = fc
    print fc_init

    b1, a1 = signal.butter(3, [lower1, upper1], \
                           btype='bandpass')

    b_hpf, a_hpf = signal.butter(3, (2.0/fsamp)*high_pass, btype='high')

    notch_digital = (2.0 / fsamp) * (2.0 * f_rot)
    bn, an = signal.iirnotch(notch_digital, 10000)                          

    times = []

    center_freq = []
    center_freq_err = []

    all_freq = []
    all_freq_err = []

    all_phase = []
    all_phase_err = []

    all_time = []

    nfiles = len(files)
    #suffix = '%i / %i' % (pathind+1, npaths)
    suffix = ''

    amp = 0
    dfdt = 0
    spindown = False
    first = False
    for fileind, file in enumerate(files):
        bu.progress_bar(fileind, nfiles, suffix=suffix)

        fobj = hsDat(file)
        t = fobj.attribs["time"]

        vperp = fobj.dat[:,0]

        if not spindown:
            vperp_filt = signal.filtfilt(b1, a1, vperp)

            vperp_filt_fft = np.fft.rfft(vperp_filt)
            vperp_filt_asd = np.abs(vperp_filt_fft)

            # if plot_raw_dat:
            #     plt.loglog(freqs, vperp_filt_asd)
            #     plt.loglog(freqs, np.abs(np.fft.rfft(vperp)))
            #     plt.xlim(fc - bandwidth, fc + bandwidth)
            #     plt.show()

            p0 = [np.max(vperp_filt_asd), fc, 1, 0]
            popt, pcov = opti.curve_fit(lorentzian, freqs, vperp_filt_asd, p0=p0)
            if amp == 0:
                amp = popt[0]

            if (np.abs(fc - popt[1]) > 5.0) or (popt[0] < 0.5 * amp):
                print 'CHIRP STARTED' 
                spindown = True
                first = True
                #fc_old = fc
                fc_old = popt[1]
                fc_init = fc
                t_init = t
            else:
                amp = popt[0]
                fc = popt[1]
                continue


        if first:
            fc = fc_old #+ dfdt * (nsamp * (1.0 / fsamp))
            first = False
        else:
            delta_t = (t - t0)*1e-9
            fc = fc_old + dfdt * delta_t

        t0 = t

        # start = time.time()
        upper1 = (2.0 / fsamp) * (1.01 * fc_old)
        lower1 = (2.0 / fsamp) * (0.99 * fc_old)

        b1, a1 = signal.butter(3, [lower1, upper1], \
                                    btype='bandpass')
        vperp_filt = signal.filtfilt(b1, a1, vperp)    
        vperp_filt_2 = signal.lfilter(bn, an, vperp_filt) 

        vperp_filt_fft = np.fft.rfft(vperp_filt_2)
        vperp_filt_asd = np.abs(vperp_filt_fft)

        p0 = [np.max(vperp_filt_asd), fc, 1, 0]
        #popt, pcov = opti.curve_fit(lorentzian, freqs, vperp_filt_asd, p0=p0)

        freq_ind = np.argmin(np.abs(freqs - fc))
        fit_inds = np.abs(freqs - fc) < 1000.0

        start = time.time()
        def fit_fun(x, A, mu, sigma, c):
            return ngauss(x, A, mu, sigma, c, n=5)
        popt, pcov = opti.curve_fit(fit_fun, freqs[fit_inds], vperp_filt_asd[fit_inds], \
                                    p0=p0, maxfev=3000)
        stop = time.time()
        if debug:
            print "Init fit: ", stop-start

        fc = popt[1]

        upper1 = (2.0 / fsamp) * (1.0075 * fc)
        lower1 = (2.0 / fsamp) * (0.9925 * fc)

        start = time.time()
        b1, a1 = signal.butter(3, [lower1, upper1], \
                                    btype='bandpass')
        vperp_filt = signal.filtfilt(b1, a1, vperp)
        vperp_filt_2 = signal.lfilter(bn, an, vperp_filt) 

        vperp_filt_fft = np.fft.rfft(vperp_filt_2)
        vperp_filt_asd = np.abs(vperp_filt_fft)
        stop = time.time()
        if debug:
            print "Filtering time: ", stop - start

        fc_old = fc
        # stop = time.time()
        # print 'Filter time: ', stop - start

        if plot_raw_dat:
            plt.figure()
            plt.loglog(freqs, np.abs(np.fft.rfft(vperp)))
            plt.loglog(freqs, vperp_filt_asd)
            plt.xlim(fc - 10*bandwidth, fc + 10*bandwidth)

        
        window = signal.tukey(nsamp, alpha=1e-4)
        ht = signal.hilbert(vperp_filt_2)# * window)
        inst_phase = np.unwrap(np.angle(ht))

        inst_freq = (fsamp / (2 * np.pi)) * np.gradient(inst_phase)
        inst_freq = 0.5 * inst_freq

        start_ind = int(0.1 * nsamp)

        fit_inst_phase = inst_phase[start_ind:-1000]
        fit_inst_freq = inst_freq[start_ind:-1000]
        fit_time_vec = time_vec[start_ind:-1000]

        start = time.time()
        fit_time_vec_ds, fit_time_vec_err_ds = rebin( fit_time_vec, 1000 )
        fit_inst_freq_ds, fit_inst_freq_err_ds = rebin( fit_inst_freq, 1000 )
        fit_inst_phase_ds, fit_inst_phase_err_ds = rebin( fit_inst_phase, 1000 )
        stop = time.time()
        if debug:
            print "Rebinning time: ", stop - start

        # plt.plot(fit_time_vec, fit_inst_freq)
        # plt.plot(fit_time_vec_ds, fit_inst_freq_ds)
        # plt.plot(fit_time_vec_ds_2, fit_inst_freq_ds_2, '--')


        #start = time.time()
        popt_line, pcov_line = opti.curve_fit(line, fit_time_vec_ds, fit_inst_freq_ds)

        fc_old = 2.0 * line(np.mean(fit_time_vec_ds), *popt_line)
        dfdt = 2.0 * popt_line[0]

        if plot_raw_dat:
            plt.figure()
            plt.errorbar(fit_time_vec_ds, fit_inst_freq_ds, \
                            xerr=fit_time_vec_err_ds, yerr=fit_inst_freq_err_ds)
            plt.plot(fit_time_vec_ds, line(fit_time_vec_ds, *popt_line))
            plt.plot(fit_time_vec_ds, np.ones_like(fit_time_vec_ds)*0.5*fc_old)
            plt.show()

        all_freq.append(fit_inst_freq_ds)
        all_freq_err.append(fit_inst_freq_err_ds)

        all_phase.append(fit_inst_phase_ds)
        all_phase_err.append(fit_inst_phase_err_ds)

        all_time.append(fit_time_vec_ds + (t - t_init)*1e-9)

        # plt.plot(np.array(all_time).flatten(), np.array(all_freq).flatten())
        # plt.show()

        times.append(t)
        center_freq.append(line(np.median(time_vec), *popt_line))
        center_freq_err.append(np.std(fit_inst_freq_ds - line(fit_time_vec_ds, *popt_line)))
        #stop = time.time()
        #print 'Linear fit and array concatentation: ', stop - start
        #print
        #sys.stdout.flush()

        # stop = time.time()
        # print 'All time: ', stop - start

    times = (np.array(times) - times[0])*1e-9
    center_freq = np.array(center_freq)

    all_time = np.array(all_time)
    all_freq = np.array(all_freq)
    all_freq_err = np.array(all_freq_err)

    resdict = {'init_freq': 0.5 * fc_init, 't_init': t_init, \
                'times': times, 'center_freq': center_freq, \
                'all_time': all_time, 'all_freq': all_freq, 'all_freq_err': all_freq_err, \
                'all_phase': all_phase, 'all_phase_err': all_phase_err}

    if save:
        pickle.dump(resdict, open(out_f + '_all.p', 'wb'))

    #outdict[out_f] = resdict
    return resdict

outdict = {}
if not load:
    results = Parallel(n_jobs=len(paths))(delayed(proc_dir)(path) for path in tqdm(paths))
    for ind, path in enumerate(paths):
        strs = path.split('/')
        if len(strs[-1]) == 0:
            dirname = strs[-2]
        else:
            dirname = strs[-1]

        outdict[dirname] = results[ind]
if load:
    for path in paths:
        strs = path.split('/')
        if len(strs[-1]) == 0:
            dirname = strs[-2]
        else:
            dirname = strs[-1]
        out_f = save_base + dirname

        outdict[out_f] = pickle.load(open(out_f + '_all.p', 'rb'))

if no_fits:
    exit()


dirs = outdict.keys()
dirs.sort()
for dirname in dirs:

    print
    print dirname

    fig_lin, ax_lin = plt.subplots(1,1)
    fig_exp, ax_exp = plt.subplots(1,1)
    fig_tau, ax_tau = plt.subplots(1,1)
    #fig_ext, ax_ext = plt.subplots(1,1)

    f0 = outdict[dirname]['init_freq']

    times = outdict[dirname]['times']
    center_freq = outdict[dirname]['center_freq']

    all_time = outdict[dirname]['all_time']
    all_freq = outdict[dirname]['all_freq']
    all_phase = outdict[dirname]['all_phase']


    decay_time = []
    decay_time_2 = []
    decay_time_3 = []
    decay_time_4 = []

    for ind in range(len(times)):

        time_vec = all_time[ind]
        freq_vec = all_freq[ind]
        phase_vec = all_phase[ind]


        popt_para, pcov_para = opti.curve_fit(parabola, time_vec, phase_vec)
        tau = -2.0 * np.pi * f0 / popt_para[0]
        decay_time.append(tau)

        # def para_fit_fun(t, tau, Nopt_Ibead, phi0):
        #     slope = 2.0 * np.pi * f0
        #     quad = 0.5 * ((f0 / tau) - (Nopt_Ibead / (2.0 * np.pi)))
        #     return phi0 + slope * t + quad * t**2

        # p0 = [2000, 10, 0]
        # bounds = ([100, 0, -np.inf], [np.inf, np.inf, np.inf])
        # plt.plot(time_vec, phase_vec)
        # plt.plot(time_vec, para_fit_fun(time_vec, *p0))
        # plt.show()
        # popt_modpara, pcov_modpara = opti.curve_fit(para_fit_fun, time_vec, phase_vec, \
        #                                             p0=p0, bounds=bounds, maxfev=10000)
        # decay_time_2.append(popt_modpara[0])


        popt_line, pcov_line = opti.curve_fit(line, time_vec, freq_vec)
        tau = -1.0 * f0 / popt_line[0]
        decay_time_3.append(tau)

        def lin_fit_fun(t, tau, Nopt_Ibead):
            slope = (f0 / tau) - (Nopt_Ibead / (2.0 * np.pi))
            return f0 - slope * t

        p0 = [2000, 10]
        bounds = ([100, 0], [np.inf, np.inf])
        popt_modline, pcov_modline = opti.curve_fit(lin_fit_fun, time_vec, freq_vec, \
                                                    p0=p0, bounds=bounds)
        decay_time_4.append(popt_modline[0])


        # tau = -1.0 * (0.5 * fc_init) / popt_line[0]
        # tau_err = tau * (np.sqrt(pcov_line[0,0]) / popt_line[0])

        # decay_time.append(tau)
        # decay_time_2.append(tau_p)
        # decay_time_err.append(tau_err)


        # ax1.plot(time_vec[start_ind:-1000], inst_freq[start_ind:-1000])
        # ax1.plot(time_vec, line(time_vec, *popt_line), '--', lw=2, color='r', alpha=0.75)
        # plt.draw()
        # plt.pause(1)

    # decay_time = np.array(decay_time)
    # decay_time_err = np.array(decay_time_err)

    # 'decay_time': decay_time, 'decay_time_2': decay_time_2, 'decay_time_err': decay_time_err, \
    #             'decay_time_3': decay_time_3, 'decay_time_4': decay_time_4,








    all_time_flat = all_time.flatten()
    all_freq_flat = all_freq.flatten()

    inds = all_time_flat < lin_fit_seconds

    popt_line, pcov_line = opti.curve_fit(line, all_time_flat[inds], all_freq_flat[inds])

    tau_line = f0 / np.abs(popt_line[0])

    ax_lin.plot(all_time_flat[inds], all_freq_flat[inds], '.')
    ax_lin.plot(all_time_flat[inds], line(all_time_flat[inds], *popt_line), \
                    '--', lw=2, color='r', alpha=0.5, label='$\tau = ${:0.2f} s'.format(tau_line))

    def fit_fun(x, tau, toff):
        return exponential(x, f0, -1.0 / tau, toff, 0)

    p0 = [2000, 0]
    bounds = ([100, -np.inf], [np.inf, np.inf])

    inds = times < exp_fit_seconds

    tau_time = []
    tau_from_exp = []
    fit_time = all_time[0]
    fit_freq = all_freq[0]
    end_ind = np.argmin( np.abs(np.array(times) - exp_fit_seconds) )
    for i in range(end_ind+1):
        bu.progress_bar(i, end_ind+1, suffix='fitting exponentials')
        popt_exp, pcov_exp = opti.curve_fit(fit_fun, fit_time, fit_freq, p0=p0, bounds=bounds)
        tau_from_exp.append(popt_exp[0])
        tau_time.append(times[i])
        if i < len(times) - 1:
            fit_time = np.concatenate((fit_time, all_time[i+1]))
            fit_freq = np.concatenate((fit_freq, all_freq[i+1]))

    ax_tau.plot(tau_time, tau_from_exp, label='Successive Exp Fits')
    ax_tau.plot(times, decay_time, label='Successive Parabolic Fits to $\phi$')
    #ax_tau.plot(times, decay_time_2, label='Successive Mod-Linear Fits to $f$')
    ax_tau.plot(times, decay_time_3, label='Successive Linear Fits to $f$')
    ax_tau.plot(times, decay_time_4, label='Successive Mod-Linear Fits to $f$')
    ax_tau.legend(loc=0)
    ax_tau.set_ylim(0,3000)
    ax_tau.set_xlim(0,500)

    print 'From exponential fit: {:0.2f}'.format(popt_exp[0])
    print 'From initial estimate: {:0.2f}'.format(decay_time_3[0])
    print 'From first {:d} seconds: {:0.2f}'.format(lin_fit_seconds, tau_line)

    resid = center_freq - fit_fun(times, *popt_exp)

    ax_exp.plot(times, center_freq)
    ax_exp.plot(times, fit_fun(times, *popt_exp), '--', lw=2, color='r', alpha=0.5)
    # plt.draw()

    plt.show()

