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

plt.rcParams.update({'font.size': 14})


dir = '/data/20181119/power_v_bits/init_meas'
#dir = '/data/20181119/power_v_bits/no-turbo_down_fine'
load_dir = False

save_ext = '20181119_init'


meas_to_plot = [('/power_v_bits/20181023_init.txt', 'historic'), \
                ('/power_v_bits/20181119_init.txt', 'init'), \
                ('/power_v_bits/20181119_init2.txt', 'up1'), \
                ('/power_v_bits/20181119_up2.txt', 'up2'), \
                ('/power_v_bits/20181119_down1.txt', 'down1'), \
                ('/power_v_bits/20181119_down2.txt', 'down2')]


trans_gain = 100e3  # V/A
pd_gain = 0.25      # A/W

line_filter_trans = 0.45 


maxfiles = 1000 # Many more than necessary
lpf = 2500   # Hz

file_inds = (0, 500)

userNFFT = 2**12
diag = False


fullNFFT = False

###########################################################




if load_dir:

    files = bu.find_all_fnames(dir, sort_time=True)
    nfiles = len(files)

    bits = []
    pows = []

    for fil_ind, fil in enumerate(files):#files[56*(i):56*(i+1)]):
        bu.progress_bar(fil_ind, nfiles)

        # Load data
        df = bu.DataFile()
        try:
            df.load(fil, load_other=True, skip_mon=True)
        except:
            print 'bad'
            continue

        mean_fb = np.mean(df.pos_fb[2])
    
        if (mean_fb > 1000) or (mean_fb < -33000):
            continue

        current = np.abs(np.mean(df.other_data[4])) / trans_gain
        power = 99.0 * current / pd_gain
        power = power / line_filter_trans

        bits.append(mean_fb)
        pows.append(power)

    plt.figure(figsize=(8.4,4.8))
    plt.plot(bits, np.array(pows) * 1e3, 'o', label='init')
    plt.xlabel('Mean Axial Feedback [bits]')
    plt.ylabel('Measured Power [mW]')
    plt.tight_layout()
    plt.show()


    outarr = np.array([bits, pows])
    np.savetxt('/power_v_bits/' + save_ext + '.txt', \
               outarr, delimiter=',')



else:

    plt.figure(figsize=(8.4,4.8))
    
    for entry in meas_to_plot:
        meas, label = entry
        dat = np.loadtxt(meas, delimiter=',')
        print dat.shape
        bits = dat[0]
        pows = dat[1]
        inds = (bits < 10000) * (bits > -33000)
        inds2 = (pows > -0.001) * (pows < 0.02)
        plt.plot(bits[inds*inds2], pows[inds*inds2] * 1e3, \
                 'o', label=label)

    plt.title('Bit to Power Calibration Drift - Approx. 1 day')
    plt.xlabel('Mean Axial Feedback [bits]')
    plt.ylabel('Measured Power [mW]')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
