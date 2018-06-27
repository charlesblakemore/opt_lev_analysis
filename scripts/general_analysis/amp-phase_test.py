import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config


dir1 = '/data/20180618/bead1/tf_20180618/freq_comb_elec5_10V'
dir1 = '/data/20180618/bead1/discharge/fine3/'

dir1 = '/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz'


files = bu.find_all_fnames(dir1)

for file in files:
    df = bu.DataFile()
    df.load(file)

    freqs = np.fft.rfftfreq(len(df.amp[0]), d=1.0/df.fsamp)

    plt.figure(1)
    plt.loglog(freqs, np.abs(np.fft.rfft(df.pos_data_3[0])) * 1000, label='offline X')
    plt.loglog(freqs, np.abs(np.fft.rfft(df.pos_data_2[0])) * 1000, label='FPGA X2')
    plt.loglog(freqs, np.abs(np.fft.rfft(df.pos_data[0])), label='FPGA X1')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('sqrt(PSD) [arb]')
    plt.legend()

    plt.figure(4)
    plt.loglog(freqs, np.abs(np.fft.rfft(df.cant_data[1])))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('sqrt(PSD) [arb]')

    for quad in [0,1,2,3]:
        print type(df.amp[quad])
        fft = np.fft.rfft(df.amp[quad])

        plt.figure(2)
        plt.plot(df.amp[quad])
        plt.figure(3)
        plt.loglog(freqs, np.abs(fft), label='Quadrant ' + str(quad))

    plt.legend()

    plt.figure(3)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('sqrt(PSD) [arb]')

    plt.show()
