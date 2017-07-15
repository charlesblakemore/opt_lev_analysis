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

dirs = [12,]
bdirs = [1,]
subtract_background = False

ddict = bu.load_dir_file( "/dirfiles/dir_file_july2017.txt" )
#print ddict

cant_axis = 2
step_axis = 1
respaxis = 0
bin_size = 1  # um
lpf = 150 # Hz

init_data = [0., 0., 10.4]
load_charge_cal = True
cal_drive_freq = 41.

fit_height = True
fit_dist = 60.   # um

maxfiles = 1000

fig_title = 'Force vs. Cantilever Position: Mock Gravity Data at Various Sep'
xlab = 'Distance Along Cantilever [um]'

tf_path = '/calibrations/transfer_funcs/Hout_20170707.p'
step_cal_path = '/calibrations/step_cals/step_cal_20170707.p'

################

def ffn(x, a, b, c):
    return a * (1. / x)**2 + b * (1. / x) + c

def ffn2(x, a, b, c):
    return a * (x - b)**2 + c



def proc_dir(d):
    dv = ddict[d]

    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)
if subtract_background:
    bdir_objs = map(proc_dir, bdirs)

pos_dict = {}
for obj in dir_objs:
    dirlabel = obj.label
    for fobj in obj.fobjs:
        cpos = fobj.get_stage_settings(axis=step_axis)[0]
        cpos = cpos * 80. / 10.   # 80um travel per 10V control
        if cpos not in pos_dict:
            pos_dict[cpos] = []
        pos_dict[cpos].append(fobj.fname)

if subtract_background:
    bpos_dict = {}
    for obj in bdir_objs:
        for fobj in obj.fobjs:
            cpos = fobj.get_stage_settings(axis=step_axis)[0]
            cpos = cpos * 80. / 10.   # 80um travel per 10V control
            if cpos not in bpos_dict:
                bpos_dict[cpos] = []
            bpos_dict[cpos].append(fobj.fname)


colors = bu.get_color_map(len(pos_dict.keys()))

pos_keys = pos_dict.keys()
pos_keys.sort()

force_at_closest = {}
fits = {}
diag_fits = {}

f, axarr = plt.subplots(3,2,sharex='all',sharey='all',figsize=(7,8),dpi=100)
if subtract_background:
    f2, axarr2 = plt.subplots(3,2,sharex='all',sharey='all',figsize=(10,12),dpi=100)


for i, pos in enumerate(pos_keys):
    newobj = cu.Data_dir(0, init_data, pos)
    newobj.files = pos_dict[pos]
    newobj.load_dir(cu.diag_loader, maxfiles=maxfiles)
    newobj.get_avg_force_v_pos(cant_axis=cant_axis, bin_size = bin_size)

    
    newobj.load_H(tf_path)

    if load_charge_cal:
        newobj.load_step_cal(step_cal_path)
    else:
        newobj.charge_step_calibration = step_calibration

    newobj.calibrate_H()

    newobj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=lpf, #plot_Happ=True, \
                             build_conv_facs=True, drive_freq=cal_drive_freq)
    newobj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)

    # Load background files
    if subtract_background:
        bobj = cu.Data_dir(0, init_data, pos)
        bobj.files = bpos_dict[pos]
        bobj.load_dir(cu.diag_loader, maxfiles=maxfiles)
        bobj.get_avg_force_v_pos(cant_axis=cant_axis, bin_size = bin_size)


        bobj.load_H(tf_path)

        if load_charge_cal:
            bobj.load_step_cal(step_cal_path)
        else:
            bobj.charge_step_calibration = step_calibration

        bobj.calibrate_H()

        bobj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=200.,# plot_Happ=True, \
                                 build_conv_facs=True, drive_freq=18.)
        bobj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = bin_size)



    keys = newobj.avg_diag_force_v_pos.keys()
    cal_facs = newobj.conv_facs
    #cal_facs = [1.,1.,1.]
    color = colors[i]
    #newpos = 90.4 - pos
    #posshort = '%g' % cu.round_sig(float(newpos),sig=2)
    if float(pos) != 0:
        posshort = '%g' % cu.round_sig(float(pos),sig=2)
    else:
        posshort = '0'

    for key in keys:
        diagdat = newobj.avg_diag_force_v_pos[key]
        dat = newobj.avg_force_v_pos[key]

        if subtract_background:
            diagbdat = bobj.avg_diag_force_v_pos[key]
            bdat = bobj.avg_force_v_pos[key]

        #offset = 0
        lab = posshort + ' um'
        for resp in [0,1,2]:
            offset = - dat[resp,0][1][-1]
            diagoffset = - diagdat[resp,0][1][-1]
            axarr[resp,0].errorbar(dat[resp,0][0], \
                                   (dat[resp,0][1]+offset)*cal_facs[resp]*1e15, \
                                   dat[resp,0][2]*cal_facs[resp]*1e15, \
                                   fmt='.-', ms=10, color = color, label=lab)
            axarr[resp,1].errorbar(diagdat[resp,0][0], \
                                   (diagdat[resp,0][1]+diagoffset)*1e15, \
                                   diagdat[resp,0][2]*1e15, \
                                   fmt='.-', ms=10, color = color, label=lab)

            if subtract_background:
                axarr2[resp,0].errorbar(dat[resp,0][0], \
                                   (dat[resp,0][1]-bdat[resp,0][1]+offset)*cal_facs[resp]*1e15, \
                                   dat[resp,0][2]*cal_facs[resp]*1e15, \
                                   fmt='.-', ms=10, color = color, label=lab)
                axarr2[resp,1].errorbar(diagdat[resp,0][0], \
                                   (diagdat[resp,0][1]-diagbdat[resp,0][1]+diagoffset)*1e15, \
                                   diagdat[resp,0][2]*1e15, \
                                   fmt='.-', ms=10, color = color, label=lab)

        if fit_height:
            offset = -dat[respaxis,0][1][-1]
            diagoffset = -diagdat[respaxis,0][1][-1]
            popt, pcov = curve_fit(ffn, dat[respaxis,0][0], \
                                           (dat[respaxis,0][1]+offset)*cal_facs[respaxis]*1e15, \
                                           p0=[1.,0.1,0])
            diagpopt, diagpcov = curve_fit(ffn, diagdat[respaxis,0][0], \
                                           (diagdat[respaxis,0][1]+diagoffset)*1e15, \
                                           p0=[1.,0.1,0])

            fits[pos] = (popt, pcov)
            diag_fits[pos] = (diagpopt, diagpcov)


axarr[0,0].set_title('Raw Imaging Response')
axarr[0,1].set_title('Diagonalized Forces')

for col in [0,1]:
    axarr[2,col].set_xlabel(xlab)

axarr[0,0].set_ylabel('X-direction Force [fN]')
axarr[1,0].set_ylabel('Y-direction Force [fN]')
axarr[2,0].set_ylabel('Z-direction Force [fN]')

axarr[0,1].legend(loc=0, numpoints=1, ncol=2, fontsize=9)

if len(fig_title):
    f.suptitle(fig_title + ' ' + dirlabel, fontsize=18)



if fit_height:
    keys = fits.keys()
    keys.sort()
    keys = map(float, keys)
    arr1 = []
    arr2 = []
    for key in keys:
        arr1.append(ffn(fit_dist, fits[key][0][0], fits[key][0][1], fits[key][0][2]))
        arr2.append(ffn(fit_dist, diag_fits[key][0][0], diag_fits[key][0][1], diag_fits[key][0][2]))

    diff1 = np.abs(np.amax(arr1) - np.amin(arr1))
    diff2 = np.abs(np.amax(arr2) - np.amin(arr2))

    p0_1 = [1, 10, 1]
    p0_2 = [1, 10, -1]

    fit1, err1 = curve_fit(ffn2, keys, arr1, p0 = p0_1, maxfev=10000)
    fit2, err2 = curve_fit(ffn2, keys, arr2, p0 = p0_2, maxfev=10000)
    xx = np.linspace(keys[0], keys[-1], 100)
    fxx1 = ffn2(xx, fit1[0], fit1[1], fit1[2]) 
    fxx2 = ffn2(xx, fit2[0], fit2[1], fit2[2]) 

    plt.figure()
    plt.suptitle("Fit of Raw Data")
    plt.plot(keys, arr1)
    plt.plot(xx, fxx1)
    plt.xlabel('Cantilever Height [um]')
    plt.ylabel('Force [fN]')
    plt.figure()
    plt.suptitle("Fit of Diagonalized Data")
    plt.plot(keys, arr2)
    plt.plot(xx, fxx2)
    plt.xlabel('Cantilever Height [um]')
    plt.ylabel('Force [fN]')

    print "Best fit positions: ", fit1[1], fit2[1]
    
plt.show()
    
        
