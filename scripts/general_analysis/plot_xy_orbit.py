import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

plt.rcParams.update({'font.size': 16})


dirnames = ['/daq2/20190619/bead1/orbit_test/test1', \
            '/daq2/20190619/bead1/orbit_test/test2']


def plot_xy_orbit(dirname, allfiles=True, user_filind=0, filter=True, fdrive=41.0):

    print 'Analyzing: ', dirname, '  ...'

    files, lengths = bu.find_all_fnames(dirname)
    nfiles = len(files)

    for filind, fil in enumerate(files):
        if not allfiles:
            if filind != user_filind:
                continue

        bu.progress_bar(filind, nfiles)

        df = bu.DataFile()
        df.load(fil)

        freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)

        plt.loglog(freqs, np.abs(np.fft.rfft(df.pos_data[0])))
        plt.loglog(freqs, np.abs(np.fft.rfft(df.pos_data[1])))

        plt.figure()
        plt.scatter(df.pos_data[0], df.pos_data[1])
        plt.show()



for dirname in dirnames:
    plot_xy_orbit(dirname)