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


dirname = '/data/20180904/bead1/discharge/fine3/'
live = False

elec_ind = 3
pos_ind = 0  # {0: x, 1: y, 2: z}

ts = 0.5


########

max_corr = []
inphase_corr = []

#plt.ion()

#fig, ax = plt.subplots(1,1)
#ax.plot(max_corr)
#ax.plot(inphase_corr)

old_mrf = ''


if live:
    while True:

        files = bu.find_all_fnames(dirname)
        files = bu.sort_files_by_timestamp(files)

        try:
            mrf = files[-2]
        except:
            mrf = ''

        if mrf != old_mrf:

            df = bu.DataFile()
            df.load(mrf)

            drive = df.electrode_data[elec_ind]
            resp = df.pos_data[pos_ind]

            freqs = np.fft.rfftfreq(len(resp), d=1.0/df.fsamp)
            fft = np.fft.rfft(resp)
            dfft = np.fft.rfft(drive)

            amp = np.abs(fft)
            phase = np.angle(fft)

            damp = np.abs(dfft)
            dphase = np.angle(dfft)

            ind = np.argmax(amp[1:]) + 1

            drive_freq = freqs[ind]

            corr = amp[ind] / damp[ind]
            max_corr.append(corr)
            inphase_corr.append( (corr * np.exp( 1.0j * (phase[ind] - dphase[ind]) )).real )

            ax.clear()

            ax.plot(max_corr)
            plt.pause(0.001)
            ax.plot(inphase_corr)
            plt.pause(0.001)
            plt.draw()
    
            old_mrf = mrf

        time.sleep(ts)

else:

    files = bu.find_all_fnames(dirname)
    files = bu.sort_files_by_timestamp(files)
    nfiles = len(files)

    avg_asd = []

    for fileind, filname in enumerate(files):
        bu.progress_bar(fileind, nfiles)

        df = bu.DataFile()
        df.load(filname)

        df.diagonalize(plot=False)

        drive = df.electrode_data[elec_ind]
        resp = df.pos_data[pos_ind]
        diag_resp = df.diag_pos_data[pos_ind]

        normfac = bu.fft_norm(df.nsamp, df.fsamp)

        if len(resp) != len(drive):
            continue

        freqs = np.fft.rfftfreq(len(resp), d=1.0/df.fsamp)
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
        
        if (fileind >= 143) and (fileind <= 160):
            if not len(avg_asd):
                avg_asd = amp * df.conv_facs[0] * normfac
                avg_diag_asd = diag_amp * normfac
                N = 1
            else:
                avg_asd += amp * df.conv_facs[0] * normfac
                avg_diag_asd += diag_amp * normfac
                N += 1

        damp = np.abs(dfft)
        dphase = np.angle(dfft)

        ind = np.argmax(damp[1:]) + 1

        drive_freq = freqs[ind]
        #plt.loglog(drive_freq, amp[ind], '.', ms=10)
        #plt.show()

        corr = amp[ind] / damp[ind]
        max_corr.append(corr)
        inphase_corr.append( (corr * np.exp( 1.0j * (phase[ind] - dphase[ind]) )).real )
    
        #ax.clear()
    
        #ax.plot(max_corr)
        #plt.pause(0.001)
        #ax.plot(inphase_corr)
        #plt.pause(0.001)
        #plt.draw()

        #time.sleep(ts)


    plt.figure()
    plt.loglog(freqs, avg_asd / N)

    plt.figure()
    plt.loglog(freqs, avg_diag_asd / N)


    plt.figure()
    plt.plot(max_corr)
    plt.plot(inphase_corr)
    plt.show()
        
        

    
