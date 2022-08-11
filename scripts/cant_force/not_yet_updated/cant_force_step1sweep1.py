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

###########################
# Generic script to display Force vs. Cant. Pos. with an
# option to fit the response for various positions, find
# the maximal response and thus locate the cantilever
#
# This script is specialized in analyze_grav_data.py,
# find_height_from_dipole.py and more
###########################

dirs = [32,]
bdirs = [1,]
subtract_background = False

ddict = bu.load_dir_file( "/dirfiles/dir_file_july2017.txt" )
#print ddict

cant_axis = 2    # Axis along which cantilever is actively driven
step_axis = 1    # Axis with different DC cantilever positions
respaxis = 1     # Expected signal axis
bin_size = 4     # um
lpf = 150        # Hz, acausal top-hat filter at this freq

init_data = [0., 0., 0]  # Separation data to initialize empy directories

fit_height = False #True
fit_dist = 2.     # um, distance to compute force from fit to locate cantilever

maxfiles = 1000   # Maximum number of files to load from a directory

#fig_title = 'Force vs. Cantilever Position: Dipole vs. Height'
fig_title = 'Force vs. Cantilever Position:'
#xlab = 'Distance from Cantilever [um]'
xlab = 'Distance along Cantilever [um]'

# Locate Calibration files
tf_path = '/calibrations/transfer_funcs/Hout_20170718.p'
step_cal_path = '/calibrations/step_cals/step_cal_20170718.p'
load_charge_cal = True
cal_drive_freq = 41.

################

# Use fitting functions

def ffn(x, a, b, c):
    return a * (1. / x)**2 + b * (1. / x) + c

def ffn2(x, a, b, c):
    return a * (x - b)**2 + c



def proc_dir(d):
    # simple directory processing function to load data and find
    # different cantilever positions
    dv = ddict[d]
    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader)
    return dir_obj

# Do intial processing
dir_objs = list(map(proc_dir, dirs))
if subtract_background:
    bdir_objs = list(map(proc_dir, bdirs))

# Loop over new directory objects and extract cantilever postions
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


colors = bu.get_colormap(len(list(pos_dict.keys())))

# Obtain the unique cantilever positions and sort them
pos_keys = list(pos_dict.keys())
pos_keys.sort()

# initialize some dictionaries that will be updated in a for-loop
force_at_closest = {}
fits = {}
diag_fits = {}

f, axarr = plt.subplots(3,2,sharex='all',sharey='all',figsize=(7,8),dpi=100)
if subtract_background:
    f2, axarr2 = plt.subplots(3,2,sharex='all',sharey='all',figsize=(10,12),dpi=100)

# Loop over files by cantilever position, make a new directory object for each 
# position then bin all of the files at that position
for i, pos in enumerate(pos_keys):
    newobj = cu.Data_dir(0, init_data, pos)
    newobj.files = pos_dict[pos]
    newobj.load_dir(cu.diag_loader, maxfiles=maxfiles)
    newobj.get_avg_force_v_pos(cant_axis=cant_axis, bin_size = bin_size)

    # Load the calibrations
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



    keys = list(newobj.avg_diag_force_v_pos.keys()) # Should only be one key here
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
        # Force objects are indexed as follows:
        # data[response axis][velocity mult.][bins, data, or errs]
        #     response axis  : X=0, Y=1, Z=2
        #     velocity mult. : both=0, forward=1, backward=-1
        #     b, d, or e     : bins=0, data=1, errors=2 
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
            # refer to indexing eplanation above if this is confusing!
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


# Analyze the fits (if fits were performed) and find the maximal
# response at the location specified in the preamble
if fit_height:
    keys = list(fits.keys())
    keys.sort()
    keys = list(map(float, keys))
    arr1 = []
    arr2 = []
    for key in keys:
        # compute response at the specified location
        arr1.append(ffn(fit_dist, fits[key][0][0], fits[key][0][1], 0 )) 
        arr2.append(ffn(fit_dist, diag_fits[key][0][0], diag_fits[key][0][1], 0 )) 

    p0_1 = [1, 10, 1]
    p0_2 = [1, 10, 1]

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

    print("Best fit positions: ", fit1[1], fit2[1])
    
plt.show()
    
        
