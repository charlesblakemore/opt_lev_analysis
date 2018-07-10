import os, fnmatch

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config

#######################################################
# Script to generate step calibrations and transfer
# functions making use of the calib_util and 
# transfer_func_util libraries.
#######################################################


#### PREAMBLE
####   include paths and saving options

step_cal_dir = '/data/20180704/bead1/discharge/fine2'
max_file = 500


fake_step_cal = False
vpn = 1.0e14

tf_cal_dir = '/data/20180704/bead1/tf_20180704/'

date = tf_cal_dir.split('/')[2]

plot_Hfunc = True
interpolate = False 
save = True 


# Doesn't use this but might later
thermal_path = '/data/20170903/bead1/1_5mbar_nocool.h5'

#######################################################

ext = config.extensions['trans_fun']

# Generate automatic path for saving
if interpolate:
    savepath = '/calibrations/transfer_funcs/' + date + '_interp' + ext
else:
    savepath = '/calibrations/transfer_funcs/' + date + ext





# Find all the relevant files
step_cal_files = []
for root, dirnames, filenames in os.walk(step_cal_dir):
    for filename in fnmatch.filter(filenames, '*' + config.extensions['data']):
        if '_fpga.h5' in filename:
            continue
        step_cal_files.append(os.path.join(root, filename))
# Sort files based on final index
step_cal_files.sort(key = bu.find_str)


tf_cal_files = []
for root, dirnames, filenames in os.walk(tf_cal_dir):
    for filename in fnmatch.filter(filenames, '*' + config.extensions['data']):
        if '_fpga.h5' in filename:
            continue
        tf_cal_files.append(os.path.join(root, filename))



#### BODY OF CALIBRATION



# Do the step calibration
if not fake_step_cal:

    step_file_objs = []
    for filname in step_cal_files[:max_file]:
        df = bu.DataFile()
        df.load(filname)
        step_file_objs.append(df)

    vpn, off, err = cal.step_cal(step_file_objs)





tf_file_objs = []
for fil_ind, filname in enumerate(tf_cal_files):
    bu.progress_bar(fil_ind, len(tf_cal_files), suffix='opening files')
    df = bu.DataFile()
    df.load(filname)
    tf_file_objs.append(df)

# Build the uncalibrated TF: Vresp / Vdrive
allH = tf.build_uncalibrated_H(tf_file_objs, plot_qpd_response=False)

Hout = allH['Hout']
Hnoise = all['Hnoise']

# Calibrate the transfer function to Vresp / Newton_drive
# for a particular charge step calibration
Hcal = tf.calibrate_H(Hout, vpn)

# Build the Hfunc object
if not interpolate:
    Hfunc = tf.build_Hfuncs(Hcal, fpeaks=[400, 400, 200], weight_peak=False, \
                            weight_lowf=True, plot_fits=plot_Hfunc, \
                            plot_inits=False, weight_phase=True, grid=True,\
                            deweight_peak=True, lowf_weight_fac=0.01)
if interpolate:
    Hfunc = tf.build_Hfuncs(Hcal, interpolate=True, plot_fits=plot_Hfunc, \
                             max_freq=600)

# Save the Hfunc object
if save:
    pickle.dump(Hfunc, open(savepath, 'wb'))

