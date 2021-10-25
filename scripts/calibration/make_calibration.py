import os, fnmatch, traceback, re, time

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interp

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config


plt.rcParams.update({'font.size': 14})

#######################################################
# Script to generate step calibrations and transfer
# functions making use of the calib_util and 
# transfer_func_util libraries.
#######################################################



#####################################
### Settings for discharge
#####################################

plot_raw_dat = False

# step_cal_dir = ['/data/old_trap/20180613/bead1/discharge/fine2/']
# first_file = 0
# last_file = -1

# step_cal_dir = ['/data/old_trap/20181119/bead1/discharge/fine/']
# first_file = 0
# last_file = -1

# step_cal_dir = ['/data/old_trap/20190408/bead1/discharge/fine/']
# first_file = 0
# last_file = -1

using_tabor = True
tabor_ind = 3

# step_cal_dir = ['/data/old_trap/20200727/bead1/discharge/fine/']
# first_file = 0
# last_file = -1
# files_to_pop = []

# step_cal_dir = ['/data/old_trap/20200924/bead1/discharge/fine/']
# first_file = 1
# last_file = -1
# files_to_pop = [120, 125, 126, 133, 147, 155, 161, 162, 168, \
#                 174, 175, 185, 186, 194, 204, 228, 229, 230, \
#                 231, 235, 279, 282]

step_cal_dir = ['/data/old_trap/20201030/bead1/discharge/fine/']
first_file = 0
last_file = 550
files_to_pop = [46, 133, 209]

# step_cal_dir = ['/data/old_trap/20201222/gbead1/discharge/fine/']
# first_file = 0
# last_file = -1

# using_tabor = True
# tabor_ind = 3
# tabor_ind = 4


# new_trap = True
new_trap = False

# step_cal_dir = ['/data/new_trap/20200113/Bead1/Discharge/']
# first_file = 64
# last_file = -1

# step_cal_dir = ['/data/new_trap/20200320/Bead1/Discharge/Discharge_after_Mass_20200402/']
# # first_file = 300
# first_file = 0
# last_file = -1

# step_cal_dir = ['/data/new_trap/20200525/Bead2/Discharge/Discharge0526/']
# first_file = 0
# last_file = -1

sort_by_index = False
sort_time = True

# files_to_pop = []
# files_to_pop = [229, 230, 231, 232, 233, 234]



skip_subdirectories = False

elec_channel_select = 1
pcol = -1
# pcol = 2
# pcol = 2


correlation_phase = 0.0
# correlation_phase = -1.0 * np.pi / 4.0
# correlation_phase = np.pi / 6.0
plot_correlations = False


# auto_try = 0.25     ### for Z direction in new trap
# auto_try = 1.5e-8   ### for Y direction in new trap
# auto_try = 0.1
# auto_try = 0.028
# auto_try = 0.011
# auto_try = 100
auto_try = 0.0

decimate = False
dec_fac = 2


fake_step_cal = False

# ## OLD TRAP
# vpn = 7.264e16
vpn = 8.48e16
drive_freq = 41.0

## NEW TRAP
# vpn = 1.796986e17
# drive_freq = 71.0

save_discharge_plot = True
plot_residual_histograms = True

save_charge = True
# save_charge = False


#####################################
### Settings for transfer function
#####################################


# tf_cal_dir = '/data/old_trap/20180613/bead1/tf_20180613_2/'
# tf_cal_dir = '/data/old_trap/20181119/bead1/tf_20181119/'
tf_cal_dir = '/data/old_trap/20190408/bead1/tf_20190408/'
# tf_cal_dir = '/data/old_trap/20200307/gbead1/tf_20200311/'

# tf_cal_dir = '/data/new_trap/20191204/Bead1/TransFunc/'
# tf_cal_dir = '/data/new_trap/20200110/Bead2/TransFunc/'
# tf_cal_dir = '/data/new_trap/20200113/Bead1/TransFunc/'
# tf_cal_dir = '/data/new_trap/20200320/Bead1/TransFunc/'

tf_substr = ''
# tf_substr = 'm300k_250s_1hz'
# tf_substr = 'm300k_50s'
# tf_substr = '_0.h5'

plot_response = False

zero_drive_phase = True
skip_qpd = True
lines_to_remove = []
# lines_to_remove = [7.0, 35.0, 60.0, 98, 420.0, 490.0]
fit_freqs = [1.0, 600.0]

amp_xlim = (4, 750)
# amp_ylim = ()
amp_ylim = [(9e18, 6e22), (9e18, 6e22), (1e15, 3e18)]
phase_xlim = (4, 750)
phase_ylim = (-1.2*np.pi, 1.2*np.pi)

plot_tf = True
plot_tf_fits = True
plot_off_diagonal = False

### Matrix specifying which elements of the transfer function
### to interpolate. If not interpolated, fits a simple harmonic
### oscillator to the response
interps = [[0,1,1], \
           [1,0,1], \
           [1,1,0]]

### Matrices specifying the phase processing behavior, i.e. do
### we try a real unwrap like np.unwrap(), or do some stupid shit
### that works sometimes (empirical evidence)
derpy_unwrap = [[0,0,0], \
                [0,0,0], \
                [0,0,0]]
real_unwrap = [[1,0,0], \
               [0,1,0], \
               [0,0,1]]

smoothing = 1

plot_inverse_tf = False
suppress_off_diag = True


#####################################
###  Shared settings
#####################################

save_tf = True
# save_tf = False






recharge = False
if type(step_cal_dir) == str:
    step_date = re.search(r"\d{8,}", step_cal_dir)[0]
else:
    step_date = re.search(r"\d{8,}", step_cal_dir[0])[0]

tf_date = re.search(r"\d{8,}", tf_cal_dir)[0]

# tf_date = step_date







#######################################################

ext = config.extensions['trans_fun']

# Generate automatic paths for saving
savepath = '/data/old_trap_processed/calibrations/transfer_funcs/' + tf_date + ext

if save_charge:
    prefix = '/data/old_trap_processed/calibrations/charges/'
    if recharge:
        charge_path = prefix + step_date + '_recharge.charge'
    else:
        charge_path = prefix + step_date + '.charge'

    if new_trap:
        charge_path = charge_path.replace('old_trap', 'new_trap')
    bu.make_all_pardirs(charge_path)

if new_trap:
    savepath = savepath.replace('old_trap', 'new_trap')
bu.make_all_pardirs(savepath)

use_origin_timestamp = False
# if new_trap:
#     use_origin_timestamp = True


# Find all the relevant files
step_cal_files, lengths = bu.find_all_fnames(step_cal_dir, sort_by_index=sort_by_index, \
                                             sort_time=sort_time, \
                                             use_origin_timestamp=use_origin_timestamp, \
                                             skip_subdirectories=skip_subdirectories)
# for name in step_cal_files:
#     print(name)
# input()

for filind in files_to_pop[::-1]:
    step_cal_files.pop(filind)

# for i in range(5):
#     step_cal_files.pop(559)


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



tf_cal_files, lengths = bu.find_all_fnames(tf_cal_dir, substr=tf_substr,\
                                           sort_by_index=sort_by_index, \
                                           sort_time=sort_time, \
                                           use_origin_timestamp=use_origin_timestamp, \
                                           skip_subdirectories=skip_subdirectories)
# tf_cal_files = tf_cal_files[:3]
# print(tf_cal_files)
# input()

# tf_cal_files = ['/data/new_trap/20200525/Bead2/TransFunc/TransFunc_X_7.h5', \
#                 '/data/new_trap/20200525/Bead2/TransFunc/TransFunc_Y_7.h5', \
#                 '/data/new_trap/20200525/Bead2/TransFunc/TransFunc_Z_7.h5']

# tf_cal_files = ['/data/new_trap/20200525/Bead2/TransFunc/TransFunc_X_6.h5', \
#                 '/data/new_trap/20200525/Bead2/TransFunc/TransFunc_Y_6.h5', \
#                 '/data/new_trap/20200525/Bead2/TransFunc/TransFunc_Z_6.h5']

# tf_cal_files_2 = []
# for file in tf_cal_files:
#     if '_1.h5' in file:
#         continue
#     tf_cal_files_2.append(file)
# tf_cal_files = tf_cal_files_2





#####################################
#### BODY OF CALIBRATION
#####################################

if decimate:
    step_cal_files = step_cal_files[::dec_fac]

if last_file == -1:
    last_file = len(step_cal_files)
step_cal_files = step_cal_files[first_file:last_file]

nstep_files = len(step_cal_files)

# nstep_files = np.min([max_file, len(step_cal_files)])
# Do the step calibration
if not fake_step_cal:
    step_file_objs = []
    step_cal_vec_inphase = []
    step_cal_vec_max = []
    step_cal_vec_userphase = []
    pow_vec = []
    zpos_vec = []
    time_vec = []
    #for fileind, filname in enumerate(step_cal_files[:max_file]):
    print('Processing discharge files...')
    for fileind, filname in enumerate(step_cal_files):
        bu.progress_bar(fileind, nstep_files)
        df = bu.DataFile()
        try:
            if new_trap:
                df.load_new(filname)
                if not df.electrode_settings['driven'][elec_channel_select]:
                    continue
            else:
                df.load(filname, plot_raw_dat=plot_raw_dat)
        except Exception:
            traceback.print_exc()
            continue

        if using_tabor and not new_trap:
            df.load_other_data()

        time_vec.append(df.time * 1e-9) ### ns to seconds

        step_resp_dict = \
            cal.find_step_cal_response(df, bandwidth=20.0, tabor_ind=tabor_ind,\
                                       using_tabor=using_tabor, pcol=pcol, \
                                       new_trap=new_trap, plot=False, \
                                       userphase=correlation_phase)
        if fileind == 0:
            print('Drive freq [Hz]: {:0.1f}'.format(step_resp_dict['drive_freq']))
            pcol_actual = step_resp_dict['pcol']

        step_cal_vec_inphase.append(step_resp_dict['inphase'])
        step_cal_vec_max.append(step_resp_dict['max'])
        step_cal_vec_userphase.append(step_resp_dict['userphase'])

        drive_freq = step_resp_dict['drive_freq']

    if np.mean(step_cal_vec_inphase[:5]) > 0:
        fac = 1.0
    else:
        fac = -1.0

    time_vec = np.array(time_vec) - time_vec[0]
    step_cal_vec_inphase = np.array(step_cal_vec_inphase)
    step_cal_vec_max = np.array(step_cal_vec_max)
    step_cal_vec_userphase = np.array(step_cal_vec_userphase)

    step_cal_vec_inphase[-5:] = 0
    step_cal_vec_max[-5:] = 0
    step_cal_vec_userphase[-5:] = 0

    if plot_correlations:
        plt.rcParams.update({'font.size': 16})
        # tvec = np.arange(len(step_cal_vec_inphase)) * 10
        plt.figure(figsize=(10,4))
        plt.plot(time_vec, -1.0*step_cal_vec_inphase, 'o', \
                        label='In-Phase Correlation', zorder=2)
        plt.plot(time_vec, np.abs(step_cal_vec_max), 'o', \
                        label='Max Correlation', zorder=3)
        plt.plot(time_vec, -1.0*step_cal_vec_userphase, 'o', \
                        label='User-phase Correlation', zorder=4)
        plt.axhline(0,ls='--',alpha=0.5,color='k', zorder=1)
        plt.ylabel('Response [Arb/(V/m)]')
        plt.xlabel('Time [s]')
        plt.legend(loc=0, fontsize=10)
        plt.tight_layout()
        plt.show()

        input()
        # time.sleep(5)

    nsec = df.nsamp * (1.0 / df.fsamp)

    vpn, off, err, q0 = cal.step_cal(step_cal_vec_userphase, nsec=nsec, \
                                     new_trap=new_trap, auto_try=auto_try, \
                                     plot_residual_histograms=plot_residual_histograms, \
                                     save_discharge_plot=save_discharge_plot, \
                                     date=step_date)
    print(vpn)

if save_charge:
    print('Saving charge file to:\n     {:s}'.format(charge_path))
    if recharge:
        np.save(open(charge_path, 'wb'), [q0])
    else:
        np.save(open(charge_path, 'wb'), [-1.0 * q0])


print()
input('Proceed with transfer function? (ENTER) ')


tf_file_objs = []
print('Processing transfer function files...')
for fil_ind, filname in enumerate(tf_cal_files):
    bu.progress_bar(fil_ind, len(tf_cal_files), suffix='opening files')
    df = bu.DataFile()
    if new_trap:
        df.load_new(filname)
        # if df.nsamp < 1.0e5:
        #     continue
    else:
        df.load(filname)

    tf_file_objs.append(df)

# Build the uncalibrated TF: Vresp / Vdrive
allH = tf.build_uncalibrated_H(tf_file_objs, plot_response=plot_response, \
                                new_trap=new_trap, \
                                lines_to_remove=lines_to_remove, \
                                zero_drive_phase=zero_drive_phase, \
                                skip_qpd=skip_qpd)

Hout = allH['Hout']
Hnoise = allH['Hout_noise']

# Calibrate the transfer function to Vresp / Newton_drive
# for a particular charge step calibration
keys = np.array(list(Hout.keys()))
keys.sort()
close_freq = keys[ np.argmin(np.abs(keys - drive_freq)) ]
# print(vpn, pcol, Hout[close_freq][2,2])

Hcal, q = tf.calibrate_H(Hout, vpn, step_cal_drive_channel=pcol_actual, \
                            drive_freq=drive_freq, verbose=True)

### Build the Hfunc object
### Hfunc
Hfunc = tf.build_Hfuncs(Hcal, fpeaks=[400, 400, 200], \
                        weight_lowf=False, lowf_weight_fac=0.01, \
                        weight_phase=False, weight_peak=False, \
                        deweight_peak=False, linearize=False, \
                        real_unwrap=real_unwrap, derpy_unwrap=derpy_unwrap, \
                        fit_freqs=fit_freqs, interps=interps, \
                        plot=plot_tf, plot_fits=plot_tf_fits, plot_inits=False, \
                        plot_off_diagonal=plot_off_diagonal, grid=True,\
                        smoothing=smoothing, amp_xlim=amp_xlim, amp_ylim=amp_ylim, \
                        phase_xlim=phase_xlim, phase_ylim=phase_ylim)

if plot_inverse_tf:
    freqs = np.linspace(1, 1000, 2500)
    Harr = tf.make_tf_array(freqs, Hfunc, suppress_off_diag=suppress_off_diag)

    tf.plot_tf_array(freqs, Harr)

### Save the Hfunc object
if save_tf:
    print('Saving transfer function:\n     {:s}'.format(savepath))
    pickle.dump( Hfunc, open(savepath, 'wb') )

