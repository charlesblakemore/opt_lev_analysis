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

#dirs = [42,38,39,40,41]
dirs = [419,]

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
#print ddict

show_avg_force = False
fft = False
calibrate = True
init_data = [0., 0., 20.]

load_charge_cal = True
#files = np.arange(0,42,1)
files = np.array([1,2,3,4,5])#,6,7,8,9,10,])
maxfiles = 1000

bin_size = 5

tf_path = './trans_funcs/Hout_20160808.p'
step_cal_path = './calibrations/step_cal_20160808.p'


def proc_dir(d):
    dv = ddict[d]

    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.diag_loader)
    #dir_obj.diagonalize_files

    return dir_obj

dir_objs = map(proc_dir, dirs)

time_dict = {}
for obj in dir_objs:
    for fobj in obj.fobjs:
        time = fobj.Time
        time_dict[time] = fobj


times = time_dict.keys()
times.sort()

colors_yeay = bu.get_color_map( len(times) )

f, axarr = plt.subplots(3,1,sharex='all')
for i, time in enumerate(times):
    col = colors_yeay[i]
    cfobj = time_dict[time]
    
    for ax in [0,1,2]:
        axarr[ax].loglog(cfobj.fft_freqs, np.abs(cfobj.data_fft[ax]), \
                         color=col, label=str(time))
        axarr[ax].set_ylim(1e-4,1e3)

#axarr[0].legend(loc=0,numpoints=1,ncol=2)
plt.show()
