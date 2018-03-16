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

step_cal_dir = '/data/20180314/bead1/discharge_fine4'
#step_cal_dir = '/data/20171106/bead1/discharge_fine3'

fake_step_cal = False
vpn = 1.0e14

tf_cal_dir = '/data/20180314/bead1/tf_20180314/'

date = tf_cal_dir.split('/')[2]

plot_Hfunc = True
interpolate = True 
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
        step_cal_files.append(os.path.join(root, filename))
# Sort files based on final index
step_cal_files.sort(key = bu.find_str)


tf_cal_files = []
for root, dirnames, filenames in os.walk(tf_cal_dir):
    for filename in fnmatch.filter(filenames, '*' + config.extensions['data']):
        tf_cal_files.append(os.path.join(root, filename))


#### BODY OF CALIBRATION

step_file_objs = []
for filname in step_cal_files:
    df = bu.DataFile()
    df.load(filname)
    step_file_objs.append(df)

# Do the step calibration
if not fake_step_cal:
    vpn, off, err = cal.step_cal(step_file_objs)





tf_file_objs = []
for filname in tf_cal_files:
    df = bu.DataFile()
    df.load(filname)
    tf_file_objs.append(df)

# Build the uncalibrated TF: Vresp / Vdrive
Hout, Hnoise = tf.build_uncalibrated_H(tf_file_objs, fix_HF = True)

# Calibrate the transfer function to Vresp / Newton_drive
# for a particular charge step calibration
Hcal = tf.calibrate_H(Hout, vpn)

# Build the Hfunc object
if not interpolate:
    Hfunc = tf.build_Hfuncs(Hcal, fpeaks=[400, 400, 50], weight_peak=False, \
                            weight_lowf=True, plot_fits=plot_Hfunc, \
                            plot_inits=False, weight_phase=True, grid=True,\
                            deweight_peak=True)
if interpolate:
    Hfunc = tf.build_Hfuncs(Hcal, interpolate=True, plot_fits=plot_Hfunc, \
                             max_freq=600)

# Save the Hfunc object
if save:
    pickle.dump(Hfunc, open(savepath, 'wb'))

