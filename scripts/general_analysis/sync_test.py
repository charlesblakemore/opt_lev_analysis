import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config


dir1 = '/data/20180618/bead1/discharge/coarse2'

files = bu.find_all_fnames(dir1, ext='.h5')

for file in files:
    df = bu.DataFile()
    df.load(file, plot_sync=True)
    plt.plot(df.electrode_data[3])
    plt.figure()
    plt.plot(df.pos_data[0])
    plt.show()
