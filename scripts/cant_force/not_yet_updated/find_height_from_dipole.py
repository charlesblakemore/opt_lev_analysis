import cant_util as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import tkinter
import tkinter.filedialog
import os, sys
from scipy.optimize import curve_fit
import scipy.optimize as optimize
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize 

###########################
# Script to analyze the microsphere's dipole response to a fixed
# voltage on the cantilver which is driven toward and away from
# the bead
###########################

bias = False
stagestep = True
stepind = 0

dirs = [18,]
bdirs = [1,]
subtract_background = False

ddict = bu.load_dir_file( "/dirfiles/dir_file_sept2017.txt" )
maxfiles = 10000   # Maximum number of files to load from a directory

get_h_from_z = True
show_fits = True

fit_height = False
fit_dist = 70.     # um, distance to compute force from fit to locate cantilever
init_data = [20., 0., 0]  # Separation data to initialize directories

bin_size = 4     # um, Binning for final force v. pos
lpf = 150        # Hz, acausal top-hat filter at this freq
cantfilt = True

fig_title = 'Force vs. Cantilever Position: Dipole vs. Height'
include_dirlab = False

# Locate Calibration files
tf_path = '/calibrations/transfer_funcs/Hout_20170903.p'
step_cal_path = '/calibrations/step_cals/step_cal_20170906_10um_bead.p'

legend = True
leginds = [1,1]

##########################################################
# Don't edit below this unless you know what you're doing

RESP_AX = 0
SWEEP_AX = 2     # Cantilever sweep axis, 1 for Y, 2 for Z
STEP_AX = 0      # Axis with differnt DC pos., 2 for height, 0 for sep 

cal_drive_freq = 41  # Hz

xlab = 'Distance from Cantilever [um]'

################

# Use fitting functions

def ffn_old(x, a, b, c):
    return a * (1. / x)**2 + b * (1. / x) + c

def ffn(x, a, b, c, sep=0., maxval=80.):
    return a * (1. / (x - sep - maxval))**2 + b * (1. / (x - sep - maxval))**2 + c

def ffn_wlin(x, a, b, c, d, sep=0., maxval=80.):
    return a * (1. / (x - sep - maxval))**2 + b * (1. / (x - sep - maxval))**2 + c + d * x

def ffn2(x, a, b, c):
    return a * x**2 + b * x + c

def ffn2_neg(x, a, b, c):
    return -1.0 * (a * x**2 + b * x + c)




def proc_dir(d):
    # simple directory processing function to load data and find
    # different cantilever positions
    dv = ddict[d]
    dir_obj = cu.Data_dir(dv[0], init_data, dv[1])
    dir_obj.load_dir(cu.diag_loader, maxfiles=maxfiles)
    
    dir_obj.filter_files_by_cantdrive(cant_axis=SWEEP_AX, nharmonics=10, noise=True, width=1.)

    dir_obj.get_avg_force_v_pos(cant_axis=SWEEP_AX, bin_size = bin_size, cantfilt=cantfilt, \
                               stagestep=stagestep, stepind=stepind, bias=bias)

    # Load the calibrations
    dir_obj.load_H(tf_path)
    dir_obj.load_step_cal(step_cal_path)
    dir_obj.calibrate_H()

    dir_obj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=lpf, #plot_Happ=True, \
                             drive_freq=cal_drive_freq, cantfilt=cantfilt)

    dir_obj.get_avg_force_v_pos(cant_axis=SWEEP_AX, bin_size = bin_size, diag=True, cantfilt=cantfilt, \
                               stagestep=stagestep, stepind=stepind, bias=bias)

    return dir_obj

# Do intial processing
dir_objs = list(map(proc_dir, dirs))
if subtract_background:
    bdir_objs = list(map(proc_dir, bdirs))

sys.stdout.flush()
# initialize some dictionaries that will be updated in a for-loop
force_at_closest = {}
fits = {}
diag_fits = {}

f, axarr = plt.subplots(3,2,sharex='all',sharey='all',figsize=(7,8),dpi=100)
if subtract_background:
    f2, axarr2 = plt.subplots(3,2,sharex='all',sharey='all',figsize=(10,12),dpi=100)

if show_fits:
    f3, fitax = plt.subplots(1,2)

for objind, obj in enumerate(dir_objs):

    sep = obj.seps[SWEEP_AX]
    maxval = obj.maxvals[SWEEP_AX]

    #print sep
    #print maxval

    def fitfunc(x, a, b, c, d):
        return ffn_wlin(x, a, b, c, d, sep=sep, maxval=maxval)

    if subtract_background:
        bobj = bdir_objs[objind]

    keys = list(obj.avg_diag_force_v_pos.keys()) 
    cal_facs = obj.conv_facs
    #cal_facs = [1.,1.,1.]

    keycolors = bu.get_colormap(len(keys))
    keys.sort(key = lambda x: float(x))

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
        lab = str(key) + ' um'
        for resp in [0,1,2]:
            #offset = - dat[resp,0][1][0]
            #diagoffset = - diagdat[resp,0][1][0]

            offset = 0
            diagoffset = 0
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
            offset = -dat[RESP_AX,0][1][0]
            diagoffset = -diagdat[RESP_AX,0][1][0]
            popt, pcov = curve_fit(fitfunc, dat[RESP_AX,0][0], \
                                           (dat[RESP_AX,0][1]+offset)*cal_facs[RESP_AX]*1e15, \
                                           p0=[1.,0.1,0, 0])
            diagpopt, diagpcov = curve_fit(fitfunc, diagdat[RESP_AX,0][0], \
                                           (diagdat[RESP_AX,0][1]+diagoffset)*1e15, \
                                           p0=[1.,0.1,0, 0])

            if show_fits:
                fitax[0].plot(dat[RESP_AX,0][0], fitfunc(dat[RESP_AX,0][0], *popt), color = color)
                fitax[0].plot(dat[RESP_AX,0][0], (dat[RESP_AX,0][1]+offset)*cal_facs[RESP_AX]*1e15, 'o', color = color)
                fitax[1].plot(diagdat[RESP_AX,0][0], (diagdat[RESP_AX,0][1]+diagoffset)*1e15, 'o', color = color)
                fitax[1].plot(diagdat[RESP_AX,0][0], fitfunc(diagdat[RESP_AX,0][0], *diagpopt), color = color)

            fits[key] = (popt, pcov)
            diag_fits[key] = (diagpopt, diagpcov)


axarr[0,0].set_title('Raw Imaging Response')
axarr[0,1].set_title('Diagonalized Forces')

for col in [0,1]:
    axarr[2,col].set_xlabel(xlab)

axarr[0,0].set_ylabel('X-direction Force [fN]')
axarr[1,0].set_ylabel('Y-direction Force [fN]')
axarr[2,0].set_ylabel('Z-direction Force [fN]')

if legend:
    axarr[leginds[0]][leginds[1]].legend(loc=0, numpoints=1, ncol=3, fontsize=9)

if include_dirlab:
    dirlabel = ' ' + dir_objs[0].label
else:
    dirlabel = ''

if len(fig_title):
    f.suptitle(fig_title + dirlabel, fontsize=18)


# Analyze the fits (if fits were performed) and find the maximal
# response at the location specified in the preamble
if fit_height:
    keys = list(fits.keys())
    keys.sort(key = lambda x: float(x))
    fkeys = list(map(float, keys))
    arr1 = []
    arr2 = []
    for key in keys:
        # compute response at the specified location
        arr1.append(ffn(fit_dist, fits[key][0][0], fits[key][0][1], 0 )) 
        arr2.append(ffn(fit_dist, diag_fits[key][0][0], diag_fits[key][0][1], 0 )) 

    p0_1 = [1, 10, 1]
    p0_2 = [1, 10, 1]

    fit1, err1 = curve_fit(ffn2, fkeys, arr1, p0 = p0_1, maxfev=10000)
    fit2, err2 = curve_fit(ffn2, fkeys, arr2, p0 = p0_2, maxfev=10000)
    xx = np.linspace(float(keys[0]), float(keys[-1]), 100)
    fxx1 = ffn2(xx, fit1[0], fit1[1], fit1[2]) 
    fxx2 = ffn2(xx, fit2[0], fit2[1], fit2[2]) 

    plt.figure()
    plt.suptitle("Fit of Raw Data")
    plt.plot(fkeys, arr1)
    plt.plot(xx, fxx1)
    plt.xlabel('Cantilever Height [um]')
    plt.ylabel('Force [fN]')
    plt.figure()
    plt.suptitle("Fit of Diagonalized Data")
    plt.plot(fkeys, arr2)
    plt.plot(xx, fxx2)
    plt.xlabel('Cantilever Height [um]')
    plt.ylabel('Force [fN]')

    min1fun = lambda x: ffn2_neg(x, *fit1)
    min2fun = lambda x: ffn2_neg(x, *fit2)

    min1res = optimize.minimize(min1fun, x0=0)
    min1 = min1res.x[0]

    min2res = optimize.minimize(min2fun, x0=0)
    min2 = min2res.x[0]

    print("Best fit positions: ", min1, min2)
    
plt.show()
    
        
