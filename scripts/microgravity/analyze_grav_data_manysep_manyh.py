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
#import cPickle as pickle
import dill as pickle
from mpl_toolkits.mplot3d import Axes3D
import grav_util as gu
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import warnings
warnings.filterwarnings("ignore")

###########################
# Script to analyze "gravity" data in which the cantilever is
# driven alongside the bead. The user can select the axis along 
# which the cantilever is driven.
###########################

stagestep = True
multistep = True
stepind = 0
stepind2 = 2

plot_vs_seps = True
height_to_plot = 0.0

plot_vs_heights = False
sep_to_plot = 80

dirs = [11,]
bdirs = [1,]
subtract_background = False

filstring = 'Z9um'
ddict = bu.load_dir_file( "/dirfiles/dir_file_sept2017.txt" )
maxfiles = 20000   # Maximum number of files to load from a directory

load_dir_objs = False
save_dir_objs = True
#dir_obj_save_path = '/processed_data/grav_data/manysep_20170906_10um_bead.p'
dir_obj_save_path = '/processed_data/grav_data/manysep_manyh_20170903_5um_bead.p'

resp = 0
SWEEP_AX = 1     # Cantilever sweep axis, 1 for Y, 2 for Z
bin_size = 1.2     # um, Binning for final force v. pos
lpf = 120        # Hz, acausal top-hat filter at this freq
cantfilt = True

fig_title = 'Force vs. Cantilever Position:'
#xlab = 'Distance along Cantilever [um]'

# Locate Calibration files
tf_path = '/calibrations/transfer_funcs/Hout_20170903.p'
step_cal_path = '/calibrations/step_cals/step_cal_20170903.p'
im_cal_path = '/calibrations/image_calibrations/stage_polynomial_1d_20170831.npy'

legend = True
leginds = [1,1]
ncol = 3
##########################################################
# Don't edit below this unless you know what you're doing

init_data = [0., 0., -40]  # Separation data to initialize empy directories

cal_drive_freq = 41  # Hz


################


def proc_dir(d):
    # simple directory processing function to load data and find
    # different cantilever positions
    dv = ddict[d]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])

    newfils = []
    for fil in dir_obj.files:
        if filstring in fil:
            newfils.append(fil)
    dir_obj.files = newfils

    # Load the calibrations
    dir_obj.load_H(tf_path)
    dir_obj.load_step_cal(step_cal_path)
    dir_obj.image_calibration = im_cal_path
    dir_obj.trapx_pixel = 340.0

    dir_obj.calibrate_H()

    dir_obj.load_dir(cu.diag_loader, maxfiles=maxfiles, prebin=True, nharmonics=10, \
                     noise=False, width=1., cant_axis=SWEEP_AX, reconstruct_lowf=True, \
                     lowf_thresh=lpf, drive_freq=cal_drive_freq, \
                     init_bin_sizes=[1.0, 1.0, 1.0], analyze_image=True, cantfilt=cantfilt)

    
    dir_obj.get_closest_sep_and_pos()


    dir_obj.get_avg_force_v_pos(cant_axis=SWEEP_AX, bin_size = bin_size, cantfilt=cantfilt, \
                               stagestep=stagestep, stepind=stepind, \
                                multistep=multistep, stepind2=stepind2)

    dir_obj.get_avg_force_v_pos(cant_axis=SWEEP_AX, bin_size = bin_size, diag=True, cantfilt=cantfilt, \
                                stagestep=stagestep, stepind=stepind, multistep=multistep, stepind2=stepind2)

    return dir_obj

# Do intial processing or load processed dir_objs
if not load_dir_objs:
    dir_objs = map(proc_dir, dirs)
    if subtract_background:
        bdir_objs = map(proc_dir, bdirs)
    if save_dir_objs:
        pickle.dump(dir_objs, open(dir_obj_save_path, 'wb') )
elif load_dir_objs:
    dir_objs = pickle.load( open(dir_obj_save_path, 'rb') )


f, axarr = plt.subplots(3,2,sharex='all',sharey='all',figsize=(7,8),dpi=100)
if subtract_background:
    f2, axarr2 = plt.subplots(3,2,sharex='all',sharey='all',figsize=(10,12),dpi=100)



plotpts = {}
for objind, obj in enumerate(dir_objs):
    seps = []
    heights = []

    keys = obj.avg_diag_force_v_pos.keys()
    for key in keys:
        seps.append(key[0])
        heights.append(key[1])
        
    seps.sort(key = lambda x: float(x))
    heights.sort(key = lambda x: float(x))
        
    seps = np.array(seps)
    heights = np.array(heights)
    
    seps = np.unique(seps)
    heights = np.unique(heights)

    plotpts[objind] = (seps, heights) 


newkeys = []
for objind, obj in enumerate(dir_objs):

    newkeys_obj = []

    seps = plotpts[objind][0]
    heights = plotpts[objind][1]

    if plot_vs_seps:
        heightind = np.argmin( np.abs(heights - height_to_plot) )
        height = heights[heightind]
        for sep in seps:
            newkeys_obj.append((sep, height))
            dat = obj.avg_diag_force_v_pos[(sep, height)]
            posvec = dat[resp,0][0]
            force = dat[resp,0][1]
            try:
                fgrid = np.vstack((fgrid, force))
                posgrid = np.vstack((posgrid, posvec))
            except:
                fgrid = force
                posgrid = posvec
    
    elif plot_vs_heights:
        sepind = np.argmin( np.abs(seps - sep_to_plot) )
        sep = seps[sepind]
        for height in heights:
            newkeys_obj.append((sep, height))
            dat = obj.avg_diag_force_v_pos[(sep, height)]
            posvec = dat[resp,0][0]
            force = dat[resp,0][1]
            try:
                fgrid = np.vstack((fgrid, force))
                posgrid = np.vstack((posgrid, posvec))
            except:
                fgrid = force
                posgrid = posvec

    newkeys.append(newkeys_obj)

testvec = posgrid[0]

for x in testvec:
    if plot_vs_seps:
        try:
            sepgrid = np.vstack((sepgrid, seps))
        except:
            sepgrid = seps
    elif plot_vs_heights:
        try:
            heightgrid = np.vstack((heightgrid, heights))
        except:
            heightgrid = heights

if plot_vs_seps:
    sepgrid = np.transpose(sepgrid)
elif plot_vs_heights:
    heightgrid = np.transpose(heightgrid)





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.plot_wireframe(sepgrid, xgrid, out)
if plot_vs_seps:
    surf = ax.plot_surface(sepgrid, posgrid, fgrid, rstride=1, cstride=1,\
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
elif plot_vs_heights:
    surf = ax.plot_surface(heightgrid, posgrid, fgrid, rstride=1, cstride=1,\
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    

fig.colorbar(surf, aspect=20)




for objind, obj in enumerate(dir_objs):    
    if subtract_background:
        bobj = bdir_objs[objind]

    keys = newkeys[objind]
    cal_facs = obj.conv_facs
    #cal_facs = [1.,1.,1.]

    keycolors = bu.get_color_map(len(keys))
    if plot_vs_seps:
        keys.sort(key = lambda x: x[0])
    if plot_vs_heights:
        keys.sort(key = lambda x: x[1])
    for keyind, key in enumerate(keys):
        color = keycolors[keyind]
        # Force objects are indexed as follows:
        # data[response axis][velocity mult.][bins, data, or errs]
        #     response axis  : X=0, Y=1, Z=2
        #     velocity mult. : both=0, forward=1, backward=-1
        #     b, d, or e     : bins=0, data=1, errors=2 
        diagdat = obj.avg_diag_force_v_pos[key]
        dat = obj.avg_force_v_pos[key]

        if subtract_background:
            diagbdat = bobj.avg_diag_force_v_pos[key]
            bdat = bobj.avg_force_v_pos[key]

        #offset = 0
        if plot_vs_seps:
            lab = str(key[0]) + ' um'
        if plot_vs_heights:
            lab = str(key[1]) + ' um'
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
    axarr[2,col].set_xlabel('Distance Along Cantilever [um]')

axarr[0,0].set_ylabel('X-direction Force [fN]')
axarr[1,0].set_ylabel('Y-direction Force [fN]')
axarr[2,0].set_ylabel('Z-direction Force [fN]')

if legend:
    axarr[leginds[0]][leginds[1]].legend(loc=0, numpoints=1, ncol=ncol, fontsize=9)

dirlabel = dir_objs[0].label
#dirlabel = '5 um Beads'

if len(fig_title):
    f.suptitle(fig_title + ' ' + dirlabel, fontsize=18)






plt.show()


    
        
