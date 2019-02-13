import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config
import transfer_func_util as tf



dirs = ['/data/20181024/backscatter_test/test1' \
       ]

#dirs = ['/data/20181025/backscatter_test/test_wfb' \
#       ]

#dirs = ['/data/20181025/backscatter_test/power_mod']


maxfiles = 1000 # Many more than necessary
lpf = 2500   # Hz

file_inds = (0, 500)

userNFFT = 2**12


###########################################################



def check_backscatter(files, colormap='jet', sort='time', file_inds=(0,10000)):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then plots the amplitude spectral density of any number of data
       or cantilever/electrode drive signals

       INPUTS: files, list of files names to extract data

       OUTPUTS: none, plots stuff
    '''

    files = [(os.stat(path), path) for path in files]
    files = [(stat.st_ctime, path) for stat, path in files]
    files.sort(key = lambda x: (x[0]))
    files = [obj[1] for obj in files]

    files = files[file_inds[0]:file_inds[1]]
    #files = files[::10]

    date = files[0].split('/')[2]

    nfiles = len(files)

    amps = []

    print "Processing %i files..." % nfiles
    for fil_ind, fil in enumerate(files):

        bu.progress_bar(fil_ind, nfiles)

        # Load data
        df = bu.DataFile()
        try:
            df.load(fil)
        except:
            continue

        df.calibrate_stage_position()
        df.calibrate_phase()

        dz_dphi = (1064e-9 / 2.0) /  (2.0 * np.pi)

        dat1 = df.zcal * dz_dphi * 1e6
        dat2 = df.cant_data[2]

    
        freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)
        fft = np.fft.rfft(dat1)
        fft_fb = np.fft.rfft(df.pos_fb[2])

        #plt.loglog(freqs, np.abs(fft))
        #plt.loglog(freqs, np.abs(fft_fb))
        #plt.show()


        times = (df.daqmx_time - df.daqmx_time[0])*1e-9

        plt.plot(times, dat1 - np.mean(dat1), label='Phase Measurement, Naive Calibration')
        plt.plot(times, dat2 - np.mean(dat2), label='Cantilever Monitor',  ls='--')
        plt.xlabel('Time [s]', fontsize=14)
        plt.ylabel('Amplitude [$\mu$m]', fontsize=14)
        plt.legend(loc=1)
        plt.tight_layout()
        plt.show()






allfiles = []
for dir in dirs:
    allfiles += bu.find_all_fnames(dir)


check_backscatter(allfiles)
