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


dirname = '/data/20180612/new_sync2'

elec_ind = 3
pos_ind = 0  # {0: x, 1: y, 2: z}


files = bu.find_all_fnames(dirname)
files = bu.sort_files_by_timestamp(files)

#files = ['/data/20180611/bead4/1_4mbar_xyzcool.h5']


for filname in files:
    df = bu.DataFile()
    df.load(filname, plot_sync=True)

    freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)
    fft_x = np.fft.rfft(df.pos_data[0])
    fft_elec = np.fft.rfft(df.electrode_data[3])
    plt.loglog(freqs, np.abs(fft_x))
    plt.loglog(freqs, np.abs(fft_elec))
    plt.grid(alpha=0.6)

    plt.show()
