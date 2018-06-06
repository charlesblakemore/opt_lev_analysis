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


dirname = '/data/20180605/bead1/discharge/coarse3/'
live = False

elec_ind = 3
pos_ind = 0  # {0: x, 1: y, 2: z}

ts = 5


########

max_corr = []
inphase_corr = []

plt.ion()

fig, ax = plt.subplots(1,1)
ax.plot(max_corr)
ax.plot(inphase_corr)

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

    for filname in files:
        
        df = bu.DataFile()
        df.load(filname)

        drive = df.electrode_data[elec_ind]
        resp = df.pos_data[pos_ind]

        if len(resp) != len(drive):
            continue

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

        time.sleep(ts)
        
        

    
