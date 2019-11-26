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


dirs = [45, 46]

ddict = bu.load_dir_file( "/home/charles/opt_lev_analysis/scripts/dirfiles/dir_file_june2017.txt" )

files = [0,1]
axes_to_plot = [2, 4]
maxfiles = 2

load_charge_cal = True

calibrate = True
tf_path = './trans_funcs/Hout_20170627.p'
step_cal_path = './calibrations/step_cal_20170627.p'
thermal_cal_file_path = '/data/20170627/bead4/1_6mbar_nocool.h5'
cal_drive_freq = 41.


init_data = [0., 0., 20.]

def proc_dir(d):
    dv = ddict[d]

    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader)
    
    return dir_obj


dir_objs = list(map(proc_dir, dirs))

time_dict = {}
for obj in dir_objs:
    for find, fobj in enumerate(obj.fobjs):
        if find not in files:
            continue
        time = fobj.Time
        if time not in time_dict:
            time_dict[time] = []
            time_dict[time].append(fobj.fname)
        else:
            time_dict[time].append(fobj.fname)


times = list(time_dict.keys())

colors_yeay = bu.get_color_map( len(times) )
f, axarr = plt.subplots(len(axes_to_plot),2,sharey='row',sharex='all',figsize=(10,12),dpi=100)

for i, time in enumerate(times):

    newobj = cu.Data_dir(0, init_data, time)
    newobj.files = time_dict[time]
    newobj.load_dir(cu.diag_loader, maxfiles=maxfiles)

    newobj.load_H(tf_path)
    newobj.load_step_cal(step_cal_path)

    newobj.calibrate_H()

    newobj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=200.,# plot_Happ=True, \
                             build_conv_facs=True, drive_freq=cal_drive_freq, close_dat=False)

    fobj = newobj.fobjs[0]
    #print fobj.pos_data
    #raw_input()
    
    for axind, ax in enumerate(axes_to_plot):
        
        axarr[axind,0].set_ylabel('Data Axis: %s' % str(ax))
        
        if ax <= 2:
            fac = newobj.conv_facs[ax]
            axarr[axind,0].plot(fobj.pos_data[ax] * fac)
            axarr[axind,1].plot(fobj.diag_pos_data[ax])
        elif ax > 2:
            newax = ax-3
            axarr[axind,0].plot(fobj.other_data[newax])
            axarr[axind,1].plot(fobj.other_data[newax])


axarr[0,0].set_title('Raw Data: Chosen Axes')
axarr[0,1].set_title('Diagonalized Data: Chosen Axes')

for col in [0,1]:
    axarr[-1,col].set_xlabel('Sample [arb]')


axarr[0,0].legend(loc=0, numpoints=1, ncol=2, fontsize=9)

#if len(fig_title):
#    f.suptitle(fig_title, fontsize=18)

plt.show()
