import sys, time, itertools

import dill as pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opti
import scipy.linalg as linalg

import bead_util as bu
import grav_util as gu
import calib_util as cal
import transfer_func_util as tf
import configuration as config

import warnings
import build_yukfuncs as yf
import scipy.signal as ss
import matplotlib.mlab as ml
warnings.filterwarnings("ignore")


p0 = [20., 0., 25.]


#############  Data Directories and Save/Load params  ############

theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'

#data_dir = '/data/20180613/bead1/grav_data/no_shield/X60-80um_Z20-30um'
data_dir = '/data/20180625/bead1/grav_data/no_shield/X60-80um_Z15-25um_17Hz'
files = bu.find_all_fnames(data_dir)

df = bu.DataFile()
df.load(files[0])

df.diagonalize()
cf = df.conv_facs
ind_lam_25um = np.argmin((yf.lambdas - 25E-6)**2)
psd_before, freqs = ml.psd(df.pos_data[0]*cf[0], Fs = 5000., NFFT = len(df.pos_data[0]))


plt.plot(ss.detrend(df.pos_data[0]*cf[0]), label = "Fx before injection")


df.inject_fake_signal(yf.yukfuncs[:, ind_lam_25um], p0, fake_alpha = 1E11)

psd_after, freqs = ml.psd(df.pos_data[0]*cf[0], Fs = 5000., NFFT = len(df.pos_data[0]))

plt.plot(ss.detrend(df.pos_data[0]*cf[0]), alpha = 0.5, label = "Fx after injection alpha 1E11")
plt.legend()
plt.xlabel("sample number")
plt.ylabel("Fx[N]")
plt.show()

plt.loglog(freqs, np.sqrt(psd_before), label = "before injection")
plt.loglog(freqs, np.sqrt(psd_after), alpha = 0.8, label = "after injection alpha = 1E11")
plt.xlabel("frequency[Hz]")
plt.ylabel("ASD [N/rt(Hz)]")
plt.legend()
plt.show()


