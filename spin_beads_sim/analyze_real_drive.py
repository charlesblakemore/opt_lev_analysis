import sys, time, os
import numpy as np
import dill as pickle

import scipy.interpolate as interp
import scipy.signal as signal
import scipy.constants as constants
import scipy.integrate as integrate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torsion_noise as tn
import bead_util as bu
import hs_digitizer as hs
import configuration as config

import peakdetect as pdet

from numba import jit
from joblib import Parallel, delayed



nfiles = 5

datadir = '/data/old_trap/20191017/bead1/spinning/ringdown/110kHz_start_6/'

bw = 10.0

mon_fac = 100.0
volt_to_efield = np.abs( bu.trap_efield([0,0,0,1,0,0,0,0], nsamp=1)[0] )

files, lengths = bu.find_all_fnames(datadir, ext='.h5', sort_time=True)
files = files[:nfiles]

real_drive = {}
for fileind, file in enumerate(files):
    obj = hs.hsDat(file)

    fsamp = obj.attribs['fsamp']
    nsamp = obj.attribs['nsamp']

    freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)

    sig_filt = obj.dat[:,1]
    for i in range(10):
        notch_freq = 60.0 * (2.0*i + 1)
        Q = 0.05 * notch_freq
        if notch_freq == 60.0:
            Q = 0.02 * notch_freq

        #start_filter_build = time.time()
        notch_digital = (2.0 / fsamp) * (notch_freq)
        bn, an = signal.iirnotch(notch_digital, Q)
        sig_filt = signal.lfilter(bn, an, sig_filt)

    asd = np.abs(np.fft.rfft(obj.dat[:,1]))
    asd_filt = np.abs(np.fft.rfft(sig_filt))

    maxind = np.argmax(asd_filt)
    driveband = np.abs(freqs - freqs[maxind]) < bw

    #plt.loglog(freqs, asd)

    tarr = np.arange(nsamp) * (1.0 / fsamp)
    sig_recon = np.zeros_like(tarr)
    #sig_recon += np.sqrt(np.mean(sig_filt**2)) * np.random.randn(nsamp)

    fft = np.fft.rfft(sig_filt)
    #drivephase = np.angle(fft[driveband])
    drivephase = np.angle(fft[maxind])

    fac = bu.fft_norm(nsamp, fsamp)

    peaks = pdet.peakdetect(asd_filt, lookahead=5, delta=20)
    max_peaks = peaks[0]
    for peak in max_peaks:
        loc, val = peak
        #plt.scatter([freqs[loc]], [val], s=50, marker='x', color='r')

        amp = np.sqrt(0.5 * nsamp / fsamp) * fac * np.abs(fft[loc])
        phase = np.angle(fft[loc]) - drivephase
        omega = 2.0 * np.pi * freqs[loc]

        if freqs[loc] in list(real_drive.keys()):
            amp_avg, phase_avg, Ndat = real_drive[freqs[loc]]
            real_drive[freqs[loc]]  = ((amp_avg * Ndat + amp) / (Ndat + 1.0), \
                                       (phase_avg * Ndat + phase) / (Ndat + 1.0), \
                                       Ndat + 1.0)
        else:
            real_drive[freqs[loc]] = (amp, phase, 1.0)

        # band = np.abs(freqs - freqs[loc]) < bw
        # dumb_fft = fft * band

        # phase_diff = drivephase - np.angle(fft[band])
        # plt.semilogx(freqs[band], phase_diff, 'o', color='k')

        # sig_recon += np.fft.irfft(dumb_fft)
        sig_recon += amp * np.cos(omega * tarr + phase)

    diff = sig_filt - sig_recon
    #diff_asd = np.abs()
    sig_recon += 5.0e-3 * np.random.randn(nsamp)
    real_drive[0.0] = (5.0e-3, 0, 1)

    # plt.figure()
    # plt.loglog(freqs, asd_filt, alpha=0.5)
    # plt.loglog(freqs, np.abs(np.fft.rfft(sig_recon)), alpha=0.5)
    # #plt.loglog(freqs, np.abs(np.fft.rfft(diff)))


    # plt.show()

    # sort_inds = np.argsort(asd)[::-1]

    # plt.loglog(freqs[sort_inds], asd[sort_inds])
    # plt.show()

keys = list(real_drive.keys())
for key in keys:
    amp, phase, Ndat = real_drive[key]
    if (Ndat < 3.0) and (key != 0.0):
        del real_drive[key]
    else:
        real_drive[key] = (amp * mon_fac * volt_to_efield, phase, Ndat)

pickle.dump(real_drive, open('./real_drive_dat.p', 'wb'))