import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import tkinter
import tkinter.filedialog
import os, sys
from scipy.optimize import curve_fit
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize
import pickle as pickle
import time

####################################################
####### Input parameters for data processing #######

TESTING = True

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
#print ddict

respdir = 'Y'
resp_axis = 1           # imaging response direction
cant_axis = 1           # stage control axis
straighten_axis = 2     # axis with coherent drive to straighten
bin_size = 5            # um of cantilever travel

load_charge_cal = True
maxfiles = 1000

plot_forward_backward = False #True
#subtract_background = True

drivefreq = 18.0
cant_volts_to_um = 8.0    # 80 um / 10 V

#fig_title = ('Force vs. Cantilever Position: %s Hz, %s - %s, ' + bead) % (drivefreq, gas, num)

#dirs = [530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543]   # 0 um sep 
dirs = [544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557]   # 10 um sep 
#dirs = [558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571]   # 20 um sep 


tf_path = './trans_funcs/Hout_20160808.p'

step_cal_path = './calibrations/step_cal_20160808.p'

thermal_cal_file_path = '/data/20160808/bead1/1_5mbar_zcool_final.h5'


fcurve_path = '/home/charles/gravity/data/force_curves.p'
force_curve_dic = pickle.load( open(fcurve_path, 'rb') )

# Identify Sep and Rbead




def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.diag_loader, maxfiles = maxfiles)

    dir_obj.load_H(tf_path)
    
    if load_charge_cal:
        dir_obj.load_step_cal(step_cal_path)
    else:
        dir_obj.charge_step_calibration = step_calibration

    dir_obj.gravity_signals = force_curve_dic

    dir_obj.calibrate_H()

    dir_obj.diagonalize_files(reconstruct_lowf=True, lowf_thresh=200.,  #plot_Happ=True, \
                             build_conv_facs=True, drive_freq=18.)

    amps = []
    for fil_obj in dir_obj.fobjs:
        stagestuff = fil_obj.get_stage_settings(axis=straighten_axis)
        amp = stagestuff[2] * cant_volts_to_um
        amps.append(amp)
    uamps = np.unique(amps)
    if len(uamps) > 1:
        print('STUPIDITYERROR: Multiple dirve amplitudes in directory')
        
    newlist = []
    for i in [0,1,2]:
        if i == straighten_axis:
            newlist.append(uamps[0])
        else:
            newlist.append(0.0)
    dir_obj.drive_amplitude = newlist

    return dir_obj


dir_objs = list(map(proc_dir, dirs))

colors_yeay = bu.get_color_map( len(dir_objs) )
f, axarr = plt.subplots(3,2,sharey='all',sharex='all',figsize=(10,12),dpi=100)

for ind, obj in enumerate(dir_objs):
    col = colors_yeay[ind]
    cal_facs = obj.conv_facs

    obj.get_avg_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    obj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    keys = list(obj.avg_force_v_pos.keys())
    for key in keys:
        amp = obj.drive_amplitude[straighten_axis]
        if straighten_axis == 0:
            lab = 'X: '
        elif straighten_axis == 1:
            lab = 'Y: '
        elif straighten_axis == 2:
            lab = 'Z: '
        lab = lab + str(amp) + ' um'

        for resp_axis in [0,1,2]:

            xdat = obj.avg_force_v_pos[key][resp_axis,0][0]
            ydat = (obj.avg_force_v_pos[key][resp_axis,0][1]) * cal_facs[resp_axis]
            errs = (obj.avg_force_v_pos[key][resp_axis,0][2]) * cal_facs[resp_axis]

            xdat_d = obj.avg_diag_force_v_pos[key][resp_axis,0][0]
            ydat_d = obj.avg_diag_force_v_pos[key][resp_axis,0][1]
            errs_d = obj.avg_diag_force_v_pos[key][resp_axis,0][2]

            xdatf = obj.avg_force_v_pos[key][resp_axis,1][0]
            xdatb = obj.avg_force_v_pos[key][resp_axis,-1][0]
            ydatf = (obj.avg_force_v_pos[key][resp_axis,1][1]) * cal_facs[resp_axis]
            ydatb = (obj.avg_force_v_pos[key][resp_axis,-1][1]) * cal_facs[resp_axis]
            errsf = (obj.avg_force_v_pos[key][resp_axis,1][2]) * cal_facs[resp_axis]
            errsb = (obj.avg_force_v_pos[key][resp_axis,-1][2]) * cal_facs[resp_axis]

            xdatf_d = obj.avg_diag_force_v_pos[key][resp_axis,1][0]
            xdatb_d = obj.avg_diag_force_v_pos[key][resp_axis,-1][0]
            ydatf_d = obj.avg_diag_force_v_pos[key][resp_axis,1][1]
            ydatb_d = obj.avg_diag_force_v_pos[key][resp_axis,-1][1]
            errsf_d = obj.avg_diag_force_v_pos[key][resp_axis,1][2]
            errsb_d = obj.avg_diag_force_v_pos[key][resp_axis,-1][2]

            offsetf = 0.0
            offsetf_d = 0.0
            offsetb = 0.0
            offsetb_d = 0.0
            offset = 0.0
            offset_d = 0.0

            if plot_forward_backward:
                axarr[resp_axis,0].errorbar(xdatf, (ydatf+offsetf)*1e15, errsf*1e15, \
                                            label = lab, fmt='<-', ms=5, color = col, mew=0.0)
                axarr[resp_axis,1].errorbar(xdatf_d, (ydatf_d+offsetf_d)*1e15, errsf_d*1e15, \
                                            label = lab, fmt='<-', ms=5, color = col, mew=0.0)
                axarr[resp_axis,0].errorbar(xdatb, (ydatb+offsetb)*1e15, errsb*1e15, \
                                            fmt='>-', ms=5, color = col, mew=0.0)
                axarr[resp_axis,1].errorbar(xdatb_d, (ydatb_d+offsetb_d)*1e15, errsb_d*1e15, \
                                            fmt='>-', ms=5, color = col, mew=0.0)
            else:
                axarr[resp_axis,0].errorbar(xdat, (ydat+offset)*1e15, errs*1e15, \
                                            label = lab, fmt='.-', ms=10, color = col)
                axarr[resp_axis,1].errorbar(xdat_d, (ydat_d+offset_d)*1e15, errs_d*1e15, \
                                            label = lab, fmt='.-', ms=10, color = col)

arrs = [axarr,]

for arr in arrs:
    arr[0,0].set_title('Raw Imaging Response')
    arr[0,1].set_title('Diagonalized Forces')

    for col in [0,1]:
        arr[2,col].set_xlabel('Distance from Cantilever [um]')

    arr[0,0].set_ylabel('X-direction Force [fN]')
    arr[1,0].set_ylabel('Y-direction Force [fN]')
    arr[2,0].set_ylabel('Z-direction Force [fN]')

    arr[0,0].legend(loc=0, numpoints=1, ncol=2, fontsize=9)

plt.show()
