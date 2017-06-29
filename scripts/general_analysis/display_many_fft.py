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



filnames = ["/data/20170613/bead2/nextday/overnight/turbobase_xyzcool_discharged_47.h5"] #, \
            #"/data/20170613/bead2/nextday/0_02mbar_xyzcool.h5", \
            #"/data/20170613/bead2/nextday/turbobase_xyzcool.h5", ]

ddict = bu.load_dir_file( "/home/charles/opt_lev_analysis/scripts/dirfiles/dir_file_june2017.txt" )
dirs = []

chan_to_plot = [0,1,2]
chan_labs = ['X', 'Y', 'Z']

NFFT = 2**12
xlim = [1, 3000]
ylim = [6e-6,1e-1]

maxfiles = 1000


tf_path = './trans_funcs/Hout_20160808.p'
step_cal_path = './calibrations/step_cal_20160808.p'



def proc_dir(d):
    dv = ddict[d]

    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader)
    #dir_obj.diagonalize_files

    return dir_obj

if dirs:
    dir_objs = map(proc_dir, dirs)
else:
    dir_objs = []


fil_objs = []
for fil in filnames:
    fil_objs.append(cu.simple_loader(fil, [0,0,0]))

#print fil_objs


time_dict = {}
for obj in dir_objs:
    for fobj in obj.fobjs:
        fobj.detrend()
        fobj.psd(NFFT = NFFT)
        time = fobj.Time
        time_dict[time] = fobj

for fobj in fil_objs:
    fobj.detrend()
    fobj.psd(NFFT = NFFT)
    time = fobj.Time
    time_dict[time] = fobj


times = time_dict.keys()
times.sort()

colors_yeay = bu.get_color_map( len(times) )
colors_yeay = ['r', 'g', 'b']

plots = len(chan_to_plot)
f, axarr = plt.subplots(plots,1,sharex='all')
for i, time in enumerate(times):
    col = colors_yeay[i]
    cfobj = time_dict[time]
    lab = str(cu.round_sig(cfobj.pressures[0],2)) + ' mbar'
    
    for ax in chan_to_plot:
        axarr[ax].loglog(cfobj.psd_freqs, np.sqrt(cfobj.psds[ax]), \
                         color=col, label=lab )
        axarr[ax].set_ylim(*ylim)
        axarr[ax].set_xlim(*xlim)

for ax in chan_to_plot:
    axarr[ax].set_ylabel(chan_labs[ax] + '   [V/rt(Hz)]')

axarr[2].set_xlabel('Frequency [Hz]')
axarr[0].legend(loc=0,numpoints=1,ncol=1)
plt.show()
