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


dirname = '/data/old_trap/20201202/power/init'

files, _ = bu.find_all_fnames(dirname, sort_time=True)


fb_set = []
power = []

for filname in files:
    df = bu.DataFile()
    df.load(filname)

    fb_set.append(np.mean(df.pos_fb[2]))
    power.append(np.abs(np.mean(df.power)))


plt.plot(fb_set, power)
plt.show()
