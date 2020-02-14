import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti
import scipy.constants as constants

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config
import transfer_func_util as tf

import iminuit


filename = '/data/old_trap/measure_power_2.h5'

df = bu.DataFile()

df.load(filename, load_other=True)

trans_gain = 100e3#TIA amplifier gain
pd_gain = 0.25 #A/W from pd manual at 1064nm

line_filter_trans = 0.45 #factor on the filter in the power pd
bs_fac = 0.01 #beam splitter factor. 1% of power is digitized

power = df.power

current = 1e-6 * df.power / trans_gain

power = current / pd_gain
power = power / line_filter_trans
power = power / bs_fac


print(np.mean(power))
