import os, fnmatch

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config



tf_cal_dir = '/data/20180625/bead1/tf_20180625/'


tf_cal_files = []
for root, dirnames, filenames in os.walk(tf_cal_dir):
    for filename in fnmatch.filter(filenames, '*' + config.extensions['data']):
        if '_fpga.h5' in filename:
            continue
        tf_cal_files.append(os.path.join(root, filename))


tf_file_objs = []
for fil_ind, filname in enumerate(tf_cal_files):
    bu.progress_bar(fil_ind, len(tf_cal_files), suffix='opening files')
    df = bu.DataFile()
    df.load(filname)
    #plt.loglog(np.abs(np.fft.rfft(df.electrode_data[3])))
    #plt.loglog(np.abs(np.fft.rfft(df.pos_data[0])))
    #plt.loglog(np.abs(np.fft.rfft(df.amp[0])))
    #plt.show()
    tf_file_objs.append(df)


allH = tf.build_uncalibrated_H(tf_file_objs, plot_qpd_response=True)

Hout = allH['Hout']
Hamp = allH['Hout_amp']
Hphase = allH['Hout_phase']
freqs = Hamp.keys()
freqs.sort()

ampfig, ampax = plt.subplots(5,3,sharex=True,sharey=True)
phasefig, phaseax = plt.subplots(5,3,sharex=True,sharey=False)

Hamp_arr = []
Hphase_arr = []
for freq in freqs:
    Hamp_arr.append(Hout[freq])
    Hphase_arr.append(Hphase[freq])
Hamp_arr = np.array(Hamp_arr)
Hphase_arr = np.array(Hphase_arr)


for quad in [0,1,2,3,4]:
    for drive in [0,1,2]:
        if quad not in [3,4]:
            ampax[quad,drive].loglog(freqs, np.abs(Hamp_arr[:,quad,drive]))
        phaseax[quad,drive].loglog(freqs, np.abs(Hphase_arr[:,quad,drive]))
plt.show()



