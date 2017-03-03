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

dirs = [375,]

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
#print ddict

calibrate = True

respdir = 'Y'
cant_axis = 2

load_charge_cal = True
maxfiles = 1000

fig_title = 'Force vs. Cantilever Position: Dipole Consistency'

tf_path = './trans_funcs/Hout_20160803.p'
step_cal_path = './calibrations/step_cal_20160803.p'

init_data = [0., 0., 20.]

heights = {}
biases = {}


def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader, maxfiles = maxfiles)

    for fobj in dir_obj.fobjs:
        pos = fobj.get_stage_settings(axis = 0)[0]
        if pos not in heights:
            heights[pos] = []
        heights[pos].append(fobj.fname)
        
        bias = fobj.electrode_settings[24]   # Index of the cantilever
        if bias not in biases:
            biases[bias] = []
        biases[bias].append(fobj.fname)


dir_objs = map(proc_dir, dirs)

thermal_cal_file_path = '/data/20160803/bead1/1_5mbar_zcool.h5'

settings = {}
for bias in biases.keys():
    for pos in heights.keys():
        setings[(bias, pos)] = list(set(biases[bias]) & set(heights[pos]))


for i, setting in enumerate(settings):
    bias, pos = setting
    newobj = cu.Data_dir(0, init_data, setting)
    newobj.files = settings[setting]
    newobj.load_dir(cu.diag_loader, maxfiles=maxfiles)
    newobj.get_avg_force_v_pos(cant_axis=cant_axis, bin_size = bin_size)

    newobj.load_H(tf_path)

    if load_charge_cal:
        newobj.load_step_cal(step_cal_path)
    else:
        newobj.charge_step_calibration = step_calibration

    newobj.calibrate_H()

    newobj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=200.,# plot_Happ=True, \
                             build_conv_facs=True, drive_freq=18.)

    plt.figure()
    for fobj in newobj.fobjs:
        




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
