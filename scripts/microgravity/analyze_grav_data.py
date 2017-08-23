import cant_util as cu
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

###########################
# Script to analyze "gravity" data in which the cantilever is
# driven alongside the bead. The user can select the axis along 
# which the cantilever is driven.
###########################


dirs = [15,]
bdirs = [1,]
subtract_background = False

ddict = bu.load_dir_file( "/dirfiles/dir_file_aug2017.txt" )
maxfiles = 1000   # Maximum number of files to load from a directory

SWEEP_AX = 1     # Cantilever sweep axis, 1 for Y, 2 for Z
STEP_AX = 0      # Axis with differnt DC pos., 0 for height, 2 for sep 
bin_size = 4     # um, Binning for final force v. pos
lpf = 150        # Hz, acausal top-hat filter at this freq
cantfilt = True

fig_title = 'Force vs. Cantilever Position:'
xlab = 'Distance along Cantilever [um]'

# Locate Calibration files
tf_path = '/calibrations/transfer_funcs/Hout_20170822.p'
step_cal_path = '/calibrations/step_cals/step_cal_20170822.p'


##########################################################
# Don't edit below this unless you know what you're doing

init_data = [0., 0., -40]  # Separation data to initialize empy directories

cal_drive_freq = 41  # Hz


################


def proc_dir(d):
    # simple directory processing function to load data and find
    # different cantilever positions
    dv = ddict[d]
    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader)
    return dir_obj

# Do intial processing
dir_objs = map(proc_dir, dirs)
if subtract_background:
    bdir_objs = map(proc_dir, bdirs)

# Loop over new directory objects and extract cantilever postions
pos_dict = {}
for obj in dir_objs:
    dirlabel = obj.label
    for fobj in obj.fobjs:
        cpos = fobj.get_stage_settings(axis=STEP_AX)[0]
        cpos = cpos * 80. / 10.   # 80um travel per 10V control
        if cpos not in pos_dict:
            pos_dict[cpos] = []
        pos_dict[cpos].append(fobj.fname)

if subtract_background:
    bpos_dict = {}
    for obj in bdir_objs:
        for fobj in obj.fobjs:
            cpos = fobj.get_stage_settings(axis=STEP_AX)[0]
            cpos = cpos * 80. / 10.   # 80um travel per 10V control
            if cpos not in bpos_dict:
                bpos_dict[cpos] = []
            bpos_dict[cpos].append(fobj.fname)


colors = bu.get_color_map(len(pos_dict.keys()))

# Obtain the unique cantilever positions and sort them
pos_keys = pos_dict.keys()
pos_keys.sort()


f, axarr = plt.subplots(3,2,sharex='all',sharey='all',figsize=(7,8),dpi=100)
if subtract_background:
    f2, axarr2 = plt.subplots(3,2,sharex='all',sharey='all',figsize=(10,12),dpi=100)

# Loop over files by cantilever position, make a new directory object for each 
# position then bin all of the files at that position
for i, pos in enumerate(pos_keys):
    newobj = cu.Data_dir(0, init_data, pos)
    newobj.files = pos_dict[pos]
    newobj.load_dir(cu.diag_loader, maxfiles=maxfiles)

    newobj.filter_files_by_cantdrive(cant_axis=SWEEP_AX, nharmonics=10, noise=True, width=1.)

    newobj.get_avg_force_v_pos(cant_axis=SWEEP_AX, bin_size = bin_size, cantfilt=cantfilt)

    # Load the calibrations
    newobj.load_H(tf_path)
    newobj.load_step_cal(step_cal_path)
    newobj.calibrate_H()

    newobj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=lpf, #plot_Happ=True, \
                             build_conv_facs=True, drive_freq=cal_drive_freq, cantfilt=cantfilt)
    newobj.get_avg_force_v_pos(cant_axis=SWEEP_AX, bin_size = bin_size, diag=True, cantfilt=cantfilt)

    # Load background files
    if subtract_background:
        bobj = cu.Data_dir(0, init_data, pos)
        bobj.files = bpos_dict[pos]
        bobj.load_dir(cu.diag_loader, maxfiles=maxfiles)
        bobj.get_avg_force_v_pos(cant_axis=SWEEP_AX, bin_size = bin_size)

        bobj.load_H(tf_path)
        bobj.load_step_cal(step_cal_path)
        bobj.calibrate_H()

        bobj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=200.,# plot_Happ=True, \
                                 build_conv_facs=True, drive_freq=18.)
        bobj.get_avg_diag_force_v_pos(cant_axis = SWEEP_AX, bin_size = bin_size)



    keys = newobj.avg_diag_force_v_pos.keys() # Should only be one key here
    cal_facs = newobj.conv_facs
    #cal_facs = [1.,1.,1.]
    color = colors[i]
    #newpos = 90.4 - pos
    #posshort = '%g' % cu.round_sig(float(newpos),sig=2)
    if float(pos) != 0:
        posshort = '%g' % bu.round_sig(float(pos),sig=2)
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
            #offset = - dat[resp,0][1][-1]
            offset = - np.mean(dat[resp,0][1])
            #diagoffset = - diagdat[resp,0][1][-1]
            diagoffset = - np.mean(diagdat[resp,0][1])

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

plt.show()
    
        
