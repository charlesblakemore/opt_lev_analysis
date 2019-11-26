import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import tkinter
import tkinter.filedialog
import os, sys, re
from scipy.optimize import curve_fit
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize
import pickle as pickle

plot_vs_time = True
plot_vs_sep = False

filstring = ''

filnames = ['/data/20170903/bead1/1_5mbar_nocool.h5', \
            '/data/20170903/bead1/turbombar_xyzcool_discharged.h5']

labs = ['1.5 mbar', '1e-6 mbar']
use_labs = True

ddict = bu.load_dir_file( '/dirfiles/dir_file_sept2017.txt' )
dirs = []

chan_to_plot = [0, 1, 2]
chan_labs = ['X', 'Y', 'Z']

NFFT = 2**14
xlim = [1, 2500]
ylim = [6e-19,1.5e-15]

maxfiles = 100

calibrate = True
tf_path = '/calibrations/transfer_funcs/Hout_20170903.p'
step_cal_path = '/calibrations/step_cals/step_cal_20170903.p'

res_freqs = np.array([223.9, 223.6, 46.6])
mass = (4. /  3) * np.pi * (2.4e-6)**3 * 2000
force_pos_cal = mass * (2 * np.pi * res_freqs)**2
use_force_pos_cal = True

charge_cal = [[''], 'Cal', 0]

charge_cal_dir_obj = cu.Data_dir(charge_cal[0], [0,0,charge_cal[2]], charge_cal[1])
charge_cal_dir_obj.load_step_cal(step_cal_path)
charge_cal_dir_obj.load_H(tf_path)
charge_cal_dir_obj.calibrate_H()
charge_cal_dir_obj.get_conv_facs()

charge_step_facs = charge_cal_dir_obj.conv_facs


pressures = []


def proc_dir(d):
    dv = ddict[d]

    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    files = []
    # initially sort by time
    for path in dir_obj.paths:
        entries = [os.path.join(path, fn) for fn in os.listdir(path)]
        entries = [(os.stat(path), path) for path in entries]
        entries = [(stat.st_ctime, path) for stat, path in entries]
    entries.sort(key = lambda x: (x[0]))
    for thing in entries[-10:]:
        print(thing)
    for thing in entries:
        if '.npy' in thing[1]:
            continue
        files.append(thing[1])
    dir_obj.files = files[:]

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

sep_dict = {}
time_dict = {}
for obj in dir_objs:
    for fobj in obj.fobjs:
        if filstring not in fobj.fname:
            continue
        sep = re.findall('[0-9]*X([0-9]+)*\\u', fobj.fname)
        sep = float(sep[0])
        fobj.detrend()
        fobj.psd(NFFT = NFFT)
        pressures.append(fobj.pressures[0])
        time = fobj.Time
        time_dict[time] = fobj

        if sep not in sep_dict:
            sep_dict[sep] = []
        sep_dict[sep].append(fobj)

for fobj in fil_objs:
    #print fobj.fname
    if filstring not in fobj.fname:
        continue
    #sep = re.findall('[0-9]*X([0-9]+)*\u', fobj.fname)
    #sep = float(sep[0])
    fobj.detrend()
    fobj.psd(NFFT = NFFT)
    pressures.append(fobj.pressures[0])
    time = fobj.Time
    time_dict[time] = fobj

    #if sep not in sep_dict:
    #    sep_dict[sep] = []
    #sep_dict[sep].append(fobj)

seps = list(sep_dict.keys())
seps.sort()

times = list(time_dict.keys())
times.sort()

if plot_vs_time:
    colors_yeay = bu.get_color_map( len(times) )
    iterlist = times

if plot_vs_sep:
    colors_yeay = bu.get_color_map( len(seps) )
    iterlist = seps


lab = ''
plots = len(chan_to_plot)
f, axarr = plt.subplots(plots,1,sharex='all')

colors_yeay = ['r', 'k']

print(list(time_dict.keys()))

for i, iterobj in enumerate(iterlist):
    col = colors_yeay[i]
    if use_labs:
        lab = labs[i]
    else:
        lab = ''
    if plot_vs_time:
        cfobj = time_dict[iterobj]
        for ax in chan_to_plot:
            if calibrate:
                if use_force_pos_cal:
                    fac = charge_step_facs[ax] / force_pos_cal[ax]
                else:
                    fac = charge_step_facs[ax]

                axarr[ax].loglog(cfobj.psd_freqs, np.sqrt(cfobj.psds[ax]) * fac, \
                                 color=col, label=lab )
            else:
                axarr[ax].loglog(cfobj.psd_freqs, np.sqrt(cfobj.psds[ax]), \
                                 color=col, label=lab )
            #axarr[ax].set_ylim(*ylim)
            #axarr[ax].set_xlim(*xlim)

    if plot_vs_sep:
        for cfobj in sep_dict[iterobj]:
            for ax in chan_to_plot:
                if calibrate:
                    axarr[ax].loglog(cfobj.psd_freqs, np.sqrt(cfobj.psds[ax]) * charge_step_facs[ax], \
                                     color=col, label=lab )
                else:
                    axarr[ax].loglog(cfobj.psd_freqs, np.sqrt(cfobj.psds[ax]), \
                                     color=col, label=lab )
                axarr[ax].set_ylim(*ylim)
                axarr[ax].set_xlim(*xlim)
            

for ax in chan_to_plot:
    if calibrate:
        if use_force_pos_cal:
            axarr[ax].set_ylabel(chan_labs[ax] + '   [m/rt(Hz)]')
        else:
            axarr[ax].set_ylabel(chan_labs[ax] + '   [N/rt(Hz)]')
    else:
        axarr[ax].set_ylabel(chan_labs[ax] + '   [V/rt(Hz)]')

axarr[-1].set_xlabel('Frequency [Hz]')
if use_labs:
    axarr[0].legend(loc=0,numpoints=1,ncol=1)

plt.figure()
plt.plot(pressures)

plt.show()
