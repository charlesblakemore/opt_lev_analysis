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

#######################
# Simple script to plot electrode potentials and compute
# their average. When performing sensitive measurements,
# we expect to always set DC voltages.
#######################

#filnames = []
filnames = ["/data/20170728/nobead/electest2/1_6mbar_nocool3.h5"] #, \
            
#labs = ['Charged', 'Discharged']
use_labs = False #True

ddict = bu.load_dir_file( '/dirfiles/dir_file_july2017.txt' )
#dirs = [11]

dirs = []

plot = True
NFFT = 2**12
xlim = [1, 2500]
ylim = [6e-18,1.5e-14]

maxfiles = 140




def proc_dir(d):
    dv = ddict[d]

    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader, maxfiles=maxfiles)
    #dir_obj.diagonalize_files

    return dir_obj

if dirs:
    dir_objs = list(map(proc_dir, dirs))
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
        time = fobj.Time
        time_dict[time] = fobj

for fobj in fil_objs:
    fobj.detrend()
    time = fobj.Time
    time_dict[time] = fobj


times = list(time_dict.keys())
times.sort()

colors_yeay = bu.get_color_map( len(times) )
colors_elecs = bu.get_color_map( 8 )
#colors_yeay = ['b', 'r', 'g']

#if plot:
#    f, axarr = plt.subplots(4,2,sharex='all',sharey='all')

avgs = []
for i, time in enumerate(times):
    col = colors_yeay[i]
    cfobj = time_dict[time]
    elecdat = cfobj.electrode_data

    if not avgs:
        avgs = np.mean(elecdat, axis=-1)
    else:
        avgs += np.mean(elecdat, axis=-1)

    if plot:
        for i in range(7):
            lab = 'elec ' + str(i)
            plt.plot(elecdat[i], color=colors_elecs[i], label=lab)
    

avgs = avgs * (1. / len(times))
print(avgs * (1. / (1. -  100./10000.)))


plt.legend(loc=0)
plt.show()
