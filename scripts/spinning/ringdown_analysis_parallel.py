import os, time, sys
import numpy as np
import matplotlib.pyplot as plt
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

progress_bar = False

high_pass = 10.0

#nbin = 1000
nbin = 100




# date = '20191017'
# base_path = '/data/old_trap/{:s}/bead1/spinning/ringdown/'.format(date)
# #base_path = '/data/old_trap/{:s}/bead1/spinning/ringdown_manual/'.format(date)

# save_base = '/data/old_trap_processed/spinning/ringdown/{:s}/'.format(date)
# #save_base = '/data/old_trap_processed/spinning/ringdown_manual/{:s}/'.format(date)

# #save_suffix = ''
# save_suffix = '_coarse'

# paths = [#base_path + '110kHz_start_1', \
#          base_path + '110kHz_start_2', \
#          base_path + '110kHz_start_3', \
#          #base_path + '110kHz_start_4', \
#          base_path + '110kHz_start_5', \
#          base_path + '110kHz_start_6', \
#          ]

# date = '20200322'
bead = 'gbead1'

date = '20200924'
bead = 'bead1'

base_path = '/data/old_trap/{:s}/{:s}/spinning/ringdown/'.format(date, bead)

save_base = '/data/old_trap_processed/spinning/ringdown/{:s}/'.format(date)

save_suffix = ''

paths = [base_path + '110kHz_start_1', \
         base_path + '110kHz_start_2', \
         base_path + '110kHz_start_3', \
         ]





sim_data = False
# base_path = '/data/old_trap_processed/spinsim_data/spindowns/sim_110kHz_real-noise'
# base_save_path = '/data/old_trap_processed/spinsim_data/spindowns_processed/sim_110kHz_real-noise'
# paths = []
# save_paths = []
# for subdir in next(os.walk(base_path))[1]:
#     paths.append(os.path.join(base_path, subdir))
#     save_paths.append(os.path.join(base_save_path, subdir))


npaths = len(paths)

ncore = npaths
#ncore = 1

save = True
load = False

lin_fit_seconds = 10
exp_fit_seconds = 500


############################

### Define some analytic functions that could be use

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




### Define the main function to process a ringdown. Unfortunately, this analysis
### has to be performed in a serial fashion, as identifying the ringdown feature 
### in the presence of all the other lines is difficult if you don't have some
### prior knowledge of where it is
def proc_dir(path):

    ### Simulated ringdowns just have the rotational frequency data, since
    ### there isn't a "readout", i.e. the cross-polarized light that appears
    ### at twice the rotation frequency
    if sim_data:
        fc = f_rot
        wc = 2.0*np.pi*fc
    else:
        fc = 2.0*f_rot
        wc = 2.0*np.pi*fc

    ### Find the final directory in which all the hdf5 files are stored in order
    ### to provide semi-intelligent naming
    strs = path.split('/')
    if len(strs[-1]) == 0:
        dirname = strs[-2]
    else:
        dirname = strs[-1]

    ### Derpy loop
    if sim_data:
        for thing in save_paths:
            if dirname in thing:
                out_f = thing
                break
    else:
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

    if sim_data:
        files, lengths = bu.find_all_fnames(path, ext='.npy', sort_time=True, verbose=False)

    else:
        files, lengths = bu.find_all_fnames(path, sort_time=True, verbose=False)

    #files = files[:1000]

    if sim_data:
        fobj = np.load(files[0])
        t0 = 0.0
        nsamp = fobj.shape[1]
        fsamp = 500000.0
    else:
        fobj = bu.hsDat(files[0], load=True)
        nsamp = fobj.nsamp
        fsamp = fobj.fsamp
        t0 = fobj.time

    time_vec = np.arange(nsamp) * (1.0 / fsamp)
    freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

    # upper1 = (2.0 / fsamp) * (fc + 0.5 * bandwidth)
    # lower1 = (2.0 / fsamp) * (fc - 0.5 * bandwidth)
    upper1 = (2.0 / fsamp) * (fc + 5 * bandwidth)
    lower1 = (2.0 / fsamp) * (fc - 5 * bandwidth)

    upper2 = (2.0 / fsamp) * (f_rot + 5 * bandwidth)
    lower2 = (2.0 / fsamp) * (f_rot - 5 * bandwidth)
    fc_init = fc
    #print fc_init

    b1, a1 = signal.butter(3, [lower1, upper1], \
                           btype='bandpass')

    b2, a2 = signal.butter(3, [lower2, upper2], \
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
        if progress_bar:
            bu.progress_bar(fileind, nfiles, suffix=suffix)

        if sim_data:
            fobj = np.load(file)
            px = fobj[1]
            # py = fobj[2]
            # vperp = np.arctan2(py, px)

            # plt.loglog(freqs, np.abs(np.fft.rfft(px)))
            # plt.loglog(freqs, np.abs(np.fft.rfft(py)))
            # plt.loglog(freqs, np.abs(np.fft.rfft(vperp)))
            # plt.show()

            vperp = px
            t = nsamp * (1.0 / fsamp) * (1.0 + fileind) * 1e9

        else:
            fobj = bu.hsDat(file, load=True)
            t = fobj.time

            vperp = fobj.dat[:,0]
            drive = fobj.dat[:,1]

        if not spindown:
            vperp_filt = signal.filtfilt(b1, a1, vperp)
            drive_filt = signal.filtfilt(b2, a2, drive)

            vperp_filt_fft = np.fft.rfft(vperp_filt)
            vperp_filt_asd = np.abs(vperp_filt_fft)

            drive_filt_fft = np.fft.rfft(drive_filt)
            drive_filt_asd = np.abs(drive_filt_fft)

            if plot_raw_dat:
                plt.loglog(freqs, np.abs(np.fft.rfft(vperp)))
                plt.loglog(freqs, vperp_filt_asd)
                plt.xlim(fc - bandwidth, fc + bandwidth)
                plt.show()

            p0 = [np.max(vperp_filt_asd), fc, 1, 0]
            popt, pcov = opti.curve_fit(lorentzian, freqs, vperp_filt_asd, p0=p0)

            p0_drive = [np.max(drive_filt_asd), f_rot, 1, 0]
            popt_drive, pcov_drive = opti.curve_fit(lorentzian, freqs, drive_filt_asd, p0=p0_drive)

            if amp == 0:
                amp = popt[0]
                amp_drive = popt_drive[0]
                continue

            # if (np.abs(fc - popt[1]) > 5.0) or (popt[0] < 0.5 * amp):
            if amp_drive > 10.0 * popt_drive[0]:
                print('CHIRP STARTED')
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
            print("Init fit: ", stop-start)

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
            print("Filtering time: ", stop - start)

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
        if not sim_data:
            inst_freq = 0.5 * inst_freq

        start_ind = int(0.1 * nsamp)

        fit_inst_phase = inst_phase[start_ind:-1000]
        fit_inst_freq = inst_freq[start_ind:-1000]
        fit_time_vec = time_vec[start_ind:-1000]

        start = time.time()
        fit_time_vec_ds, fit_time_vec_err_ds = bu.rebin_vectorized( fit_time_vec, nbin )
        fit_inst_phase_ds, fit_inst_phase_err_ds = bu.rebin_vectorized( fit_inst_phase, nbin )
        fit_inst_freq_ds, fit_inst_freq_err_ds = bu.rebin_vectorized( fit_inst_freq, nbin)#, model=line )
        stop = time.time()
        if debug:
            print("Rebinning time: ", stop - start)

        # plt.plot(fit_time_vec, fit_inst_freq)
        # plt.plot(fit_time_vec_ds, fit_inst_freq_ds)
        # plt.plot(fit_time_vec_ds_2, fit_inst_freq_ds_2, '--')


        #start = time.time()
        popt_line, pcov_line = opti.curve_fit(line, fit_time_vec_ds, fit_inst_freq_ds)

        if sim_data:
            fac = 1.0
        else:
            fac = 2.0

        fc_old = fac * line(np.mean(fit_time_vec_ds), *popt_line)
        dfdt = fac * popt_line[0]

        # resid = fit_inst_freq_ds - line(fit_time_vec_ds, *popt_line)
        # chisq = (1.0 / (len(fit_time_vec_ds) - len(popt_line)) ) * \
        #                 np.sum( resid**2 / fit_inst_freq_err_ds**2 )
        # print chisq
        # fit_inst_freq_err_ds *= np.sqrt(chisq)

        if plot_raw_dat:
            plt.figure()
            plt.errorbar(fit_time_vec_ds, fit_inst_freq_ds, \
                            xerr=fit_time_vec_err_ds, yerr=fit_inst_freq_err_ds)
            plt.plot(fit_time_vec_ds, line(fit_time_vec_ds, *popt_line))
            plt.plot(fit_time_vec_ds, np.ones_like(fit_time_vec_ds)*(1.0/fac)*fc_old)


            # freqs = np.fft.rfftfreq(len(fit_time_vec_ds), \
            #                         d=(fit_time_vec_ds[1] - fit_time_vec_ds[0]))
            # fft = np.fft.rfft(resid)
            # plt.figure()
            # plt.loglog(freqs, np.abs(fft))

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

    resdict = {'init_freq': (1.0/fac) * fc_init, 't_init': t_init, \
                'times': times, 'center_freq': center_freq, \
                'all_time': all_time, 'all_freq': all_freq, 'all_freq_err': all_freq_err, \
                'all_phase': all_phase, 'all_phase_err': all_phase_err}

    if save:
        pickle.dump(resdict, open(out_f + '{:s}_all.p'.format(save_suffix), 'wb'))

    #outdict[out_f] = resdict
    return resdict

outdict = {}
if not load:
    results = Parallel(n_jobs=ncore)(delayed(proc_dir)(path) for path in tqdm(paths))
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


