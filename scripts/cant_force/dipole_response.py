import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import Tkinter
import tkFileDialog
import os, sys
from scipy.optimize import curve_fit
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize
import cPickle as pickle
import time

dirs = [387,]

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
#print ddict

load_from_file = False
show_each_file = False
show_avg_force = False
fft = False
calibrate = True

respdir = 'Y'
cant_axis = 2

load_charge_cal = True
maxfiles = 1000

fig_title = 'Force vs. Cantilever Position: Dipole Response'

tf_path = './trans_funcs/Hout_20160805.p'
step_cal_path = './calibrations/step_cal_20160805.p'



def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.diag_loader, maxfiles = maxfiles)

    dir_obj.load_H(tf_path)
    
    if load_charge_cal:
        dir_obj.load_step_cal(step_cal_path)
    else:
        dir_obj.charge_step_calibration = step_calibration

    dir_obj.calibrate_H()
    dir_obj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=200.,# plot_Happ=True, \
                              build_conv_facs=True, drive_freq=18.)

    #dir_obj.plot_H(cal=True)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)

thermal_cal_file_path = '/data/20160714/bead1/1_5mbar_zcool_final2.h5'


#f, axarr = plt.subplots(3,2,sharey='all',sharex='all',figsize=(10,12),dpi=100)
for i, obj in enumerate(dir_objs):
    f, axarr = plt.subplots(3,2,sharey='all',sharex='all',figsize=(10,12),dpi=100)
    if calibrate:
        cal_facs = obj.conv_facs
    else:
        cal_facs = [1.,1.,1.]

    obj.get_avg_force_v_pos(cant_axis = cant_axis, bin_size = 4, bias=True)

    obj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = 4, bias=True)

    keys = obj.avg_force_v_pos.keys()
    keys.sort()

    colors_yeay = bu.get_color_map( len(keys) )

    for keyind, key in enumerate(keys):
        lab = str(key) + ' V'
        col = colors_yeay[keyind]

        for resp_axis in [0,1,2]:
            #offset = 0
            #offset_d = 0
            offset = -1.0 * obj.avg_force_v_pos[key][resp_axis,0][1][-1]
            offset_d = -1.0 * obj.avg_diag_force_v_pos[key][resp_axis,0][1][-1]

            xdat = obj.avg_force_v_pos[key][resp_axis,0][0]
            ydat = (obj.avg_force_v_pos[key][resp_axis,0][1] + offset) * cal_facs[resp_axis]
            errs = (obj.avg_force_v_pos[key][resp_axis,0][2]) * cal_facs[resp_axis]
            axarr[resp_axis,0].errorbar(xdat, ydat*1e15, errs*1e15, \
                              label = lab, fmt='.-', ms=10, color = col)

            xdat_d = obj.avg_diag_force_v_pos[key][resp_axis,0][0]
            ydat_d = obj.avg_diag_force_v_pos[key][resp_axis,0][1] + offset_d
            errs_d = obj.avg_diag_force_v_pos[key][resp_axis,0][2]
            axarr[resp_axis,1].errorbar(xdat_d, ydat_d*1e15, errs_d*1e15, \
                              label = lab, fmt='.-', ms=10, color = col)


    axarr[0,0].set_title('Raw Imaging Response')
    axarr[0,1].set_title('Diagonalized Forces')


    for col in [0,1]:
        axarr[2,col].set_xlabel('Distance from Cantilever [um]')

    axarr[0,0].set_ylabel('X-direction Force [fN]')
    axarr[1,0].set_ylabel('Y-direction Force [fN]')
    axarr[2,0].set_ylabel('Z-direction Force [fN]')

    axarr[0,0].legend(loc=0, numpoints=1, ncol=2, fontsize=9)

    if len(fig_title):
        c_title = fig_title + ' ' + obj.label
        f.suptitle(c_title, fontsize=18)

plt.show()
