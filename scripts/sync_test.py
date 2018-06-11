import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

import time


dirname = '/data/20180605/bead1/discharge/coarse2/'

files = bu.find_all_fnames(dirname, exclude_fpga=True)

for filname in files[:1000]:
    df = bu.DataFile()
    df.load(filname)

    print df.quad_time[0]
    print df.pos_time[0]
    print np.uint64(df.daqmx_time[0])

    diff1 = np.abs(float(df.quad_time[0]) - df.daqmx_time[0])
    diff2 = np.abs(float(df.pos_time[0]) - df.daqmx_time[0])

    print 'diffs', diff1*(10**(-9)), diff2*(10**(-9))

    posdat_range = np.max(df.pos_data[0]) - np.min(df.pos_data[0])
    cantdat_range = np.max(df.electrode_data[3]) - np.min(df.electrode_data[3])
    fac = cantdat_range / posdat_range


    #for point in df.pos_data[2][:100]:
    #    print np.binary_repr(point.astype(np.int32))
    #    time.sleep(0.5)

    fig, ax = plt.subplots(1,1)
    #ax.plot((df.pos_data[2]-np.mean(df.pos_data[2])) * fac)
    ax.plot(df.pos_time, (df.pos_data[0] - np.mean(df.pos_data[0])) * fac, '-', \
            lw=3, label='X')
    ax.plot(df.daqmx_time, df.electrode_data[3], label='Elec')

    #fig2, ax2 = plt.subplots(1,1)
    #ax2.plot(df.pos_data[0])

    ax.legend()

    plt.show()
    
