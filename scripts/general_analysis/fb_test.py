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


dirname = '/data/20180609/sync_test/basic_sync'

elec_ind = 3
pos_ind = 0  # {0: x, 1: y, 2: z}


files = bu.find_all_fnames(dirname)
files = bu.sort_files_by_timestamp(files)

files = ['/data/20180611/bead4/1_4mbar_xyzcool.h5']


for filname in files:
    df = bu.DataFile()
    df.load(filname)

    print df.nsamp
    print df.fsamp
    print df.pos_data.shape
    print df.pos_fb.shape
    raw_input()

    freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)
    fft_fbx = np.fft.rfft(df.pos_fb[0])
    fft_fby = np.fft.rfft(df.pos_fb[1])
    plt.loglog(freqs, np.abs(fft_fbx))
    plt.loglog(freqs, np.abs(fft_fby))
    plt.grid(alpha=0.6)

plt.show()
