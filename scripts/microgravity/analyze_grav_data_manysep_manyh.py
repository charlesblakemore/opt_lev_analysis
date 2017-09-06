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

###########################
# Script to analyze "gravity" data in which the cantilever is
# driven alongside the bead. The user can select the axis along 
# which the cantilever is driven.
###########################

bias = False
stagestep = True
multistep = True
stepind = 0
stepind2 = 2

plot_vs_seps = True
height_to_plot = 0.0

plot_vs_heights = False
sep_to_plot = 10.

dirs = [11,]
bdirs = [1,]
subtract_background = False

filstring = ''
ddict = bu.load_dir_file( "/dirfiles/dir_file_sept2017.txt" )
maxfiles = 10000   # Maximum number of files to load from a directory

load_dir_objs = False
save_dir_objs = True
dir_obj_save_path = '/processed_data/grav_data/manysep_20170903.p'

resp = 0
SWEEP_AX = 1     # Cantilever sweep axis, 1 for Y, 2 for Z
bin_size = 1.2     # um, Binning for final force v. pos
lpf = 120        # Hz, acausal top-hat filter at this freq
cantfilt = True

#fig_title = 'Force vs. Cantilever Position:'
#xlab = 'Distance along Cantilever [um]'

# Locate Calibration files
tf_path = '/calibrations/transfer_funcs/Hout_20170903.p'
step_cal_path = '/calibrations/step_cals/step_cal_20170903.p'
im_cal_path = '/calibrations/image_calibrations/stage_polynomial_1d_20170831.npy'

legend = True
#leginds = [1,1]
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


#f, axarr = plt.subplots(3,2,sharex='all',sharey='all',figsize=(7,8),dpi=100)
#if subtract_background:
#    f2, axarr2 = plt.subplots(3,2,sharex='all',sharey='all',figsize=(10,12),dpi=100)



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
    
    plotpts[objind] = (seps, heights) 


newkeys_dic = {}
for objind, obj in enumerate(dir_objs):

    newkeys = []

    seps = plotpts[objind][0]
    heights = plotpts[objind][1]

    if plot_vs_seps:
        heightind = np.argmin( np.abs(heights - height_to_plot) )
        height = heights[heightind]
        for sep in seps:
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
        sepind = np.argmind( np.abs(seps - sep_to_plot) )
        sep = seps[sepind]
        for height in heights:
            dat = obj.avg_diag_force_v_pos[(sep, height)]
            posvec = dat[resp,0][0]
            force = dat[resp,0][1]
            try:
                fgrid = np.vstack((fgrid, force))
                posgrid = np.vstack((posgrid, posvec))
            except:
                fgrid = force
                posgrid = posvec

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
    surf = ax.plot_surface(sepgrid, posgrid, fgrid, rstride=2, cstride=2,\
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
elif plot_vs_heights:
    surf = ax.plot_surface(heightgrid, posgrid, fgrid, rstride=2, cstride=2,\
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    

fig.colorbar(surf, aspect=20)

plt.show()


    
        