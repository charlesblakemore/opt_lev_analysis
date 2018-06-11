import dill as pickle

import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config



#filepath = '/data/20170903/bead1/grav_data/manysep_h0-20um/'
#filname = 'turbombar_xyzcool_discharged_stage-X7um-Y40um-Z0um_Ydrive40umAC-13Hz_0.h5'

filepath = '/data/20170903/bead1/discharge_fine/'
filname = 'turbombar_xyzcool_elec3_10000mV41Hz0mVdc_0.h5'

plot_title = ''


df = bu.DataFile()
df.load(filepath + filname)


df.calibrate_stage_position()

df.diagonalize(maxfreq=100)

df.get_force_v_pos(verbose=True, cantilever_drive=False, electrode_drive=True)


fig, axarr = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(6,10), dpi=150)

for chan in [0,1,2]:
    axarr[chan,0].plot(df.binned_data[chan][0], df.binned_data[chan][1]*df.conv_facs[chan])
    axarr[chan,1].plot(df.diag_binned_data[chan][0], df.diag_binned_data[chan][1])

plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0.5)

if plot_title:
    plt.suptitle(plot_title, fontsize=20)
    plt.subplots_adjust(top=0.9)

plt.show()
