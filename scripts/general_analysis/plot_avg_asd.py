import os, fnmatch, sys, time

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import calib_util as cu
import configuration as config

import time


#dirname = '/data/20180904/bead1/discharge/fine3/'
dirname = '/data/20180904/bead1/recharged_20180909/cant_force/acgrid_3freqs_10s'
#dirname = '/data/20180827/bead2/500e_data/dipole_v_height_no_v_ysweep'
#dirname = '/data/20180827/bead2/500e_data/dipole_v_height_no_v_xsweep'
#dirname = '/data/20180827/bead2/500e_data/shield/dipole_v_height_ydrive_no_bias'


maxfile = 200

files = bu.find_all_fnames(dirname)
files = bu.sort_files_by_timestamp(files)
nfiles = len(files)

avg_asd = [[], [], []]
avg_diag_asd = [[], [], []]
N = 0

for fileind, filname in enumerate(files[:maxfile]):
    bu.progress_bar(fileind, nfiles)

    df = bu.DataFile()
    try:
        df.load(filname)
    except:
        continue

    #df.high_pass_filter(order=1, fc=30.0)
    #df.detrend_poly(order=10, plot=True)

    df.diagonalize(plot=False)

    drive = df.electrode_data[0]
    resp = df.pos_data
    diag_resp = df.diag_pos_data

    normfac = bu.fft_norm(df.nsamp, df.fsamp)

    #if len(resp) != len(drive):
    #    continue

    freqs = np.fft.rfftfreq(len(resp[0]), d=1.0/df.fsamp)
    fft = np.fft.rfft(resp)
    diag_fft = np.fft.rfft(diag_resp)
    dfft = np.fft.rfft(drive)

    #plt.figure()
    #plt.loglog(freqs, np.abs(dfft))
    #plt.loglog(freqs, np.abs(fft))
    #plt.show()

    amp = np.abs(fft)
    diag_amp = np.abs(diag_fft)
    phase = np.angle(fft)

    #if (fileind >= 143) and (fileind <= 160):
    #if True:
    if N == 0:
        for ax in [0,1,2]:
            avg_asd[ax] = amp[ax] * df.conv_facs[ax] * normfac
            avg_diag_asd[ax] = diag_amp[ax] * normfac
        N += 1
    else:
        for ax in [0,1,2]:
            avg_asd[ax] += amp[ax] * df.conv_facs[ax] * normfac
            avg_diag_asd[ax] += diag_amp[ax] * normfac
        N += 1

    damp = np.abs(dfft)
    dphase = np.angle(dfft)

    ind = np.argmax(damp[1:]) + 1

    drive_freq = freqs[ind]
    #plt.loglog(drive_freq, amp[ind], '.', ms=10)
    #plt.show()


plt.figure(dpi=200)
plt.loglog(freqs, avg_asd[0] / N)
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('ASD', fontsize=16)

#plt.figure()
#plt.loglog(freqs, avg_diag_asd[0] / N)

plt.show()
