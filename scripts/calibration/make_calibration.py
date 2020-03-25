import os, fnmatch, traceback, re

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


step_cal_dir = ['/data/old_trap/20200322/gbead1/discharge/fine']
first_file = 0
last_file = -1

using_tabor = True
tabor_ind = 3



# step_cal_dir = ['/data/new_trap/20200306/Bead1/Discharge/']
# first_file = 150
# last_file = -1

# step_cal_dir = ['/data/new_trap/20200311/Bead1/Discharge/']
# first_file = 100
# last_file = -1

# step_cal_drive_freq = 151.0
step_cal_drive_freq = 41.0


elec_channel_select = 3
# pcol = -1
# pcol = 2
pcol = 0

# auto_try = 0.25     ### for Z direction in new trap
# auto_try = 1.5e-8   ### for Y direction in new trap
# auto_try = 0.09
auto_try = 0.0


# new_trap = True
new_trap = False


recharge = False
if type(step_cal_dir) == str:
    step_date = re.search(r"\d{8,}", step_cal_dir)[0]
    if 'recharge' in step_cal_dir:
        recharge = True
    else:
        recharge = False
else:
    step_date = re.search(r"\d{8,}", step_cal_dir[0])[0]
    for dir in step_cal_dir:
        if 'recharge' in dir:
            recharge = True

max_file = 145
decimate = False
dec_fac = 2

fake_step_cal = False
## OLD TRAP
vpn = 7.264e16
## NEW TRAP
# vpn = 7.1126e1



tf_cal_dir = '/data/old_trap/20200307/gbead1/tf_20200311/'

# tf_cal_dir = '/data/new_trap/20200306/Bead1/TransFunc/'
# tf_cal_dir = '/data/new_trap/20200311/Bead1/TransFunc/'

tf_substr = ''
# tf_substr = '_0.h5'




tf_date = re.search(r"\d{8,}", tf_cal_dir)[0]

tf_date = step_date
plot_Hfunc = True
plot_without_fits = False
interpolate = True
save = True
save_charge = True

# Doesn't use this but might later
thermal_path = '/data/20170903/bead1/1_5mbar_nocool.h5'








#######################################################

ext = config.extensions['trans_fun']

# Generate automatic paths for saving
if interpolate:
    savepath = '/data/old_trap_processed/calibrations/transfer_funcs/' + tf_date + '_interp' + ext
else:
    savepath = '/data/old_trap_processed/calibrations/transfer_funcs/' + tf_date + ext

if save_charge:
    prefix = '/data/old_trap_processed/calibrations/charges/'
    if recharge:
        charge_path = prefix + step_date + '_recharge.charge'
    else:
        charge_path = prefix + step_date + '.charge'

    print(charge_path)

    if new_trap:
        charge_path = charge_path.replace('old_trap', 'new_trap')
    bu.make_all_pardirs(charge_path)

if new_trap:
    savepath = savepath.replace('old_trap', 'new_trap')
bu.make_all_pardirs(savepath)



# Find all the relevant files
step_cal_files, lengths = bu.find_all_fnames(step_cal_dir, sort_time=True)

step_cal_files.pop(24)


#print len(step_cal_files)

# for 20180827, uncomment this
#step_cal_files.pop(53)
#step_cal_files.pop(72)

if recharge:
    step_cal_files = step_cal_files[::-1]

# ## for 20181119 recharge charge AND discharge
# step_cal_files.pop(17)
# step_cal_files.pop(17)
# step_cal_files.pop(17)
# step_cal_files.pop(17)
# # step_cal_files.pop(212)



tf_cal_files, lengths = bu.find_all_fnames(tf_cal_dir, substr=tf_substr)

# tf_cal_files_2 = []
# for file in tf_cal_files:
#     if '_1.h5' in file:
#         continue
#     tf_cal_files_2.append(file)
# tf_cal_files = tf_cal_files_2





if decimate:
    step_cal_files = step_cal_files[::dec_fac]









#### BODY OF CALIBRATION
if last_file == -1:
    last_file = len(step_cal_files)
step_cal_files = step_cal_files[first_file:last_file]

nstep_files = len(step_cal_files)

# nstep_files = np.min([max_file, len(step_cal_files)])
# Do the step calibration
if not fake_step_cal:
    step_file_objs = []
    step_cal_vec = []
    pow_vec = []
    zpos_vec = []
    #for fileind, filname in enumerate(step_cal_files[:max_file]):
    for fileind, filname in enumerate(step_cal_files):
        bu.progress_bar(fileind, nstep_files)
        df = bu.DataFile()
        try:
            if new_trap:
                df.load_new(filname)
                if not df.electrode_settings['driven'][elec_channel_select]:
                    continue
            else:
                df.load(filname)
        except Exception:
            traceback.print_exc()
            continue

        if using_tabor and not new_trap:
            df.load_other_data()

        step_resp, step_resp_nonorm, power, zpos = \
            cal.find_step_cal_response(df, bandwidth=20.0, tabor_ind=tabor_ind,\
                                       using_tabor=using_tabor, pcol=pcol, \
                                       new_trap=new_trap, plot=False)

        step_cal_vec.append(step_resp)
        # step_cal_vec.append(step_resp_nonorm)


        pow_vec.append(power)
        zpos_vec.append(zpos)

    vpn, off, err, q0 = cal.step_cal(step_cal_vec, new_trap=new_trap, \
                                     first_file=first_file, auto_try=auto_try)
    print(vpn)

if save_charge:
    if recharge:
        np.save(open(charge_path, 'wb'), [q0])
    else:
        np.save(open(charge_path, 'wb'), [-1.0 * q0])





tf_file_objs = []
for fil_ind, filname in enumerate(tf_cal_files):
    bu.progress_bar(fil_ind, len(tf_cal_files), suffix='opening files')
    df = bu.DataFile()
    if new_trap:
        df.load_new(filname)
        if df.nsamp < 1.0e5:
            continue
    else:
        df.load(filname)

    tf_file_objs.append(df)

# Build the uncalibrated TF: Vresp / Vdrive
allH = tf.build_uncalibrated_H(tf_file_objs, plot_qpd_response=False, new_trap=new_trap)

Hout = allH['Hout']
Hnoise = allH['Hout_noise']

# Calibrate the transfer function to Vresp / Newton_drive
# for a particular charge step calibration
keys = np.array(list(Hout.keys()))
keys.sort()
close_freq = keys[ np.argmin(np.abs(keys - 151.0)) ]
# print(vpn, pcol, Hout[close_freq][2,2])

Hcal, q = tf.calibrate_H(Hout, vpn, step_cal_drive_channel=pcol, drive_freq=step_cal_drive_freq)

# Build the Hfunc object
if not interpolate:
    Hfunc = tf.build_Hfuncs(Hcal, fpeaks=[400, 400, 200], weight_peak=False, \
                            weight_lowf=True, plot_fits=plot_Hfunc, \
                            plot_without_fits=plot_without_fits, \
                            plot_inits=False, weight_phase=True, grid=True,\
                            deweight_peak=True, lowf_weight_fac=0.01)
if interpolate:
    Hfunc = tf.build_Hfuncs(Hcal, interpolate=True, plot_fits=plot_Hfunc, \
                             max_freq=700)

# Save the Hfunc object
if save:
    pickle.dump(Hfunc, open(savepath, 'wb'))

