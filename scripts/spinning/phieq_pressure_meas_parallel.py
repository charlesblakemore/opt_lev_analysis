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
ncore = 30


plt.rcParams.update({'font.size': 14})

#f_rot = 210000.0
#f_rot = 100000.0
f_rot = 50000.0
filter_bandwidth = 50.0
bandwidth = 1

plot_fit = False
plot_raw_dat = False


high_pass = 10.0


# path = '/data/old_trap/0190626/bead1/spinning/ringdown/50kHz_ringdown'
# after_path = '/data/old_trap/20190626/bead1/spinning/ringdown/after_pramp'

# paths = [path, path+'2', after_path]

# save_base = '/data/old_trap_processed/spinning/ringdown/20190626/'

date = '20191017'
base_path = '/data/old_trap/{:s}/bead1/spinning/phieq_pressure_meas/'.format(date)
#base_path = '/data/old_trap/{:s}/bead1/spinning/ringdown_manual/'.format(date)

save_base = '/data/old_trap_processed/spinning/phieq_pressure_meas/{:s}/'.format(date)
#save_base = '/data/old_trap_processed/spinning/ringdown_manual/{:s}/'.format(date)

paths = [#base_path + '50kHz_1', \
         #base_path + '50kHz_2', \
         #base_path + '50kHz_3', \
         #base_path + '50kHz_4', \
         base_path + '50kHz_5', \
         ]


mbead, mbead_sterr, mbead_syserr = bu.get_mbead(date)

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

if plot_raw_dat or plot_fit:
    ncore = 1

############################

rbead, rbead_sterr, rbead_syserr = bu.get_rbead(mbead, mbead_sterr, mbead_syserr)
Ibead, Ibead_sterr, Ibead_syserr = bu.get_Ibead(mbead, mbead_sterr, mbead_syserr)

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

def sine(x, a, f, phi, c):
    return a * np.sin( 2.0 * np.pi * f * x + phi ) + c






for pathind, path in enumerate(paths):
    if load:
        continue
    files, lengths = bu.find_all_fnames(path, sort_time=True)

    fc = 2.0 * f_rot

    fobj = hsDat(files[0])
    nsamp = fobj.attribs["nsamp"]
    fsamp = fobj.attribs["fsamp"]

    time_vec = np.arange(nsamp) * (1.0 / fsamp)
    freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

    upper1 = (2.0 / fsamp) * (fc + 0.5 * filter_bandwidth)
    lower1 = (2.0 / fsamp) * (fc - 0.5 * filter_bandwidth)

    upper2 = (2.0 / fsamp) * (0.5*fc + 0.25 * filter_bandwidth)
    lower2 = (2.0 / fsamp) * (0.5*fc - 0.25 * filter_bandwidth)


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

        vperp_fft = np.fft.rfft(vperp_filt)
        elec3_fft = np.fft.rfft(elec3_filt)

        if plot_raw_dat:
            plt.plot(time_vec[:10000], vperp_filt[:10000])
            plt.figure()
            plt.loglog(freqs, np.abs(np.fft.rfft(vperp)))
            plt.loglog(freqs, np.abs(np.fft.rfft(vperp_filt)))
            plt.figure()
            plt.loglog(freqs, np.abs(np.fft.rfft(elec3)))
            plt.loglog(freqs, np.abs(np.fft.rfft(elec3_filt)))
            plt.show() 

        fc = freqs[np.argmax(np.abs(vperp_fft))]
        #print fc

        inds1 = np.abs(freqs - fc) < 0.5 * bandwidth
        inds2 = np.abs(freqs - 0.5*fc) < 0.25 * bandwidth

        dat_phase = np.angle(np.sum(vperp_fft[inds1]))
        drive_phase = np.angle(np.sum(elec3_fft[inds2]))

        drive_amp = np.abs(np.sum(elec3_fft[inds2])) * bu.fft_norm(nsamp, fsamp)

        p01 = [np.std(vperp_filt), fc, dat_phase, 0]
        popt1, pcov1 = opti.curve_fit(sine, time_vec, vperp_filt, p0=p01, maxfev=10000)
        #print popt1

        p02 = [np.std(elec3_filt), 0.5*fc, drive_phase, 0]
        popt2, pcov2 = opti.curve_fit(sine, time_vec, elec3_filt, p0=p02, maxfev=10000)
        #print popt2

        drive_amp2 = np.abs(popt2[0])
        drive_phase2 = popt2[2]
        dat_phase2 = popt1[2]

        print(drive_amp / drive_amp2)

        if plot_raw_dat:
            plot_x = time_vec[:10000]
            plt.plot(plot_x, vperp_filt[:10000])
            plt.figure()
            plt.loglog(freqs, np.abs(np.fft.rfft(vperp)))
            plt.loglog(freqs, np.abs(np.fft.rfft(vperp_filt)))
            plt.figure()
            plt.loglog(freqs, np.abs(np.fft.rfft(elec3)))
            plt.loglog(freqs, np.abs(np.fft.rfft(elec3_filt)))
            plt.show() 


        if plot_fit:
            plot_x = time_vec[:10000]
            plt.figure()
            plt.plot(plot_x, vperp_filt[:10000])
            plt.plot(plot_x, sine(plot_x, *p01))
            plt.plot(plot_x, sine(plot_x, *popt1))
            plt.figure()
            plt.plot(plot_x, elec3_filt[:10000])
            plt.plot(plot_x, sine(plot_x, *p02))
            plt.plot(plot_x, sine(plot_x, *popt2))
            plt.show()

        return [drive_amp, drive_phase, dat_phase, drive_amp2, drive_phase2, dat_phase2]


    results = Parallel(n_jobs=ncore)(delayed(proc_file)(file) for file in tqdm(files))

    results = np.array(results)

    sorter = np.argsort(results[:,0])[::-1]
    sorter2 = np.argsort(results[:,3])[::-1]

    amps = results[:,0][sorter]
    dphases = results[:,1][sorter]
    phases = results[:,2][sorter]

    amps2 = results[:,3][sorter2]
    dphases2 = results[:,4][sorter2]
    phases2 = results[:,5][sorter2]

    dphases[dphases<0.] += np.pi
    dphases2[dphases2<0.] += np.pi

    # Since sin(theta)^2 = 0.5*(1 - cos(2*theta)), there is a factor
    # of two between the phase of drive and response.
    delta_phi = phases - 2.*dphases
    delta_phi2 = phases2 - 2.*dphases2

    # Put all the phase between pi and -pi for better unwrapping 
    # later when we do more analysis
    delta_phi[delta_phi > np.pi] -= 2.*np.pi
    delta_phi[delta_phi < -np.pi] += 2.*np.pi

    uphases = np.unwrap(delta_phi) * 0.5


    delta_phi2[delta_phi2 > np.pi] -= 2.*np.pi
    delta_phi2[delta_phi2 < -np.pi] += 2.*np.pi

    uphases2 = np.unwrap(delta_phi2) * 0.5


    plt.plot(amps, uphases)
    plt.plot(amps2, uphases2)
    plt.show()