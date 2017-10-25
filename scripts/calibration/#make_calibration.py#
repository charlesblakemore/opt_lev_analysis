import os, fnmatch

import numpy as np

import bead_util as bu
import calib_util as cal
import configuration as config

#######################################################
# Script to generate step calibrations and transfer
# functions making use of the calib_util library.
#######################################################


#### PREAMBLE
####   include paths and saving options

step_cal_dir = '/data/20170903/bead1/discharge_fine'

tf_cal_dir = '/data/20170903/bead1/tf_20170903/'

thermal_path = '/data/20170903/bead1/1_5mbar_nocool.h5'

date = '20170903'
save = True

maxfiles = 141


# Find all the relevant files

step_cal_files = []
for root, dirnames, filenames in os.walk(step_cal_dir):
    for filename in fnmatch.filter(filenames, '*' + config.extensions['data']):
        step_cal_files.append(os.path.join(root, filename))

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

#vpn, off, err = cal.step_cal(step_file_objs)
#print vpn, off, err


tf_file_objs = []
for filname in tf_cal_files:
    df = bu.DataFile()
    df.load(filname)
    tf_file_objs.append(df)

