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


ddict = bu.load_dir_file( "/dirfiles/dir_file_july2017.txt" )
#print ddict

respdir = 'X'
resp_axis = 0           # imaging response direction
cant_axis = 2           # stage control axis
step_axis = 1
straighten_axis = 2     # axis with coherent drive to straighten

bin_size = 2            # um of cantilever travel
lpf_freq = 150.

load_charge_cal = True
step_cal_drive_freq = 41.
maxfiles = 1000

plot_forward_backward = False #True
drivefreq = 18.0
cant_volts_to_um = 8.0    # 80 um / 10 V

dirs = [1,]


tf_path = '/calibrations/transfer_funcs/Hout_20170707.p'
step_cal_path = '/calibrations/step_cals/step_cal_20170707.p'
thermal_cal_file_path = '/data/20170707/bead5/1_5mbar_nocool.h5'


fcurve_path = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/7_5um_sep_200um_throw_force_curves.p'
force_curve_dic = pickle.load( open(fcurve_path, 'rb') )

limitdata_path = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160928_datathief_nodecca2.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160914_datathief.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'

figtitle = 'Sensitivity with 10um Beads and Current Background'

# Identify Sep and Rbead
rbead = 5.0e-06
sep = 7.5e-06
offset = 0.0

least_squares = True
opt_filt = False

rebin = False
average_first = True

diag = False
scale = 1.0e15

'''

def proc_dir(d):
    dv = ddict[d]

    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)


pos_dict = {}
for obj in dir_objs:
    dirlabel = obj.label
    for fobj in obj.fobjs:
        cpos = fobj.get_stage_settings(axis=step_axis)[0]
        cpos = cpos * 80. / 10.   # 80um travel per 10V control
        if cpos not in pos_dict:
            pos_dict[cpos] = []
        pos_dict[cpos].append(fobj.fname)


'''


def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,-40,dv[-1]], dv[1])
    dir_obj.load_dir(cu.diag_loader, maxfiles = maxfiles)

    dir_obj.load_H(tf_path)
    
    if load_charge_cal:
        dir_obj.load_step_cal(step_cal_path)
    else:
        dir_obj.charge_step_calibration = step_calibration

    dir_obj.gravity_signals = force_curve_dic

    dir_obj.calibrate_H()

    dir_obj.diagonalize_files(reconstruct_lowf=True, lowf_thresh=lpf_freq,  #plot_Happ=True, \
                              build_conv_facs=True, drive_freq=step_cal_drive_freq)

    return dir_obj


dir_objs = list(map(proc_dir, dirs))

colors_yeay = bu.get_color_map( len(dir_objs) )
f, axarr = plt.subplots(3,2,sharey='all',sharex='all',figsize=(10,12),dpi=100)


alpha_vecs = []
lambda_vecs = []

for ind, obj in enumerate(dir_objs):
    col = colors_yeay[ind]
    cal_facs = obj.conv_facs

    obj.get_avg_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    obj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    lambdas, alphas = obj.generate_alpha_lambda_limit(rbead=rbead, sep=sep, offset=offset, \
                                                      least_squares=least_squares, opt_filt=opt_filt, \
                                                      resp_axis=resp_axis, cant_axis=cant_axis, \
                                                      rebin=rebin, bin_size=bin_size, diag=diag, \
                                                      scale=scale)

    alpha_vecs.append(alphas)
    lambda_vecs.append(lambdas)

    keys = list(obj.avg_force_v_pos.keys())
    for key in keys:

        lab = obj.label

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




flim, axlim = plt.subplots(figsize=(10,8), dpi=100)

for ind in range(len(lambda_vecs)):
    col = colors_yeay[ind]
    plt.loglog(lambda_vecs[ind], alpha_vecs[ind], color=col, label='Sensitivity', linewidth=2)
    plt.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
    if limitlab2:
        plt.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
    plt.legend(loc=0, numpoints=1, fontsize=14)
    plt.xlabel('Lambda [um]')
    plt.ylabel('Alpha')
    plt.ylim(1e2, 1e16)
    plt.xlim(1e-8, 2e-4)
    plt.grid()
    plt.title(figtitle)

plt.show()
