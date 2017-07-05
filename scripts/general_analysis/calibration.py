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



#dirs = [29,30,31,32]
#dirs = [132,133,172,]
dirs = [48,]

ddict = bu.load_dir_file( "/home/arider/opt_lev_analysis/scripts/dirfiles/dir_file_june2017.txt" )

load_charge_cal = False
step_cal_path = './calibrations/step_cal_20170629.p'
thermal_path = '/data/20170629/bead6/1_6mbar_nocool.h5'

date = '20170629'
save = True

maxfiles = 1000

#################################

if not load_charge_cal:
    cal = [['/data/20170629/bead6/discharge_fine'], 'Cal', 15, 1e-13]

    cal_dir_obj = cu.Data_dir(cal[0], [0,0,cal[2]], cal[1])
    cal_dir_obj.load_dir(cu.simple_loader)
    cal_dir_obj.build_step_cal_vec()
    cal_dir_obj.step_cal()
    if save:
        cal_dir_obj.save_step_cal('./calibrations/step_cal_'+date+'.p')

    for fobj in cal_dir_obj.fobjs:
        fobj.close_dat()

    step_calibration = cal_dir_obj.charge_step_calibration

#################################




def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.H_loader, maxfiles = maxfiles)

    dir_obj.build_uncalibrated_H(average_first=True, fix_HF=True)

    if load_charge_cal:
        dir_obj.load_step_cal(step_cal_path)
    else:
        dir_obj.charge_step_calibration = step_calibration

    #print dir_obj.charge_step_calibration.popt[0]

    dir_obj.calibrate_H()

    dir_obj.thermal_cal_file_path = thermal_path
    dir_obj.thermal_calibration()

    dir_obj.build_Hfuncs(fpeaks=[245, 255, 50], weight_peak=False, weight_lowf=True,\
                         plot_fits=True, plot_inits=False, weight_phase=True, grid=True)#, fit_osc_sum=True)
    
    return dir_obj





dir_objs = map(proc_dir, dirs)






counter = 0
for obj in dir_objs:
    if obj == dir_objs[-1]:
        obj.thermal_cal_fobj.plt_thermal_fit()



f1, axarr1 = plt.subplots(3,3, sharex='all', sharey='all')
f2, axarr2 = plt.subplots(3,3, sharex='all', sharey='all')

for obj in dir_objs:
    if obj != dir_objs[-1]:

        obj.plot_H(f1, axarr1, f2, axarr2, \
                   phase=True, show=False, label=True, show_zDC=True, \
                   inv=False, lim=False, cal=True)
        #obj.plot_H(phase=False, label=False, show=False, noise=True)
        continue

    obj.plot_H(f1, axarr1, f2, axarr2, \
               phase=True, label=True, show=True, show_zDC=True, \
               inv=False, lim=False, cal=True)
    plt.show()
    #obj.plot_H(phase=False, label=False, show=True, noise=True)
    #print obj.label
    if save:
        obj.save_H('./trans_funcs/Hout_'+date+'.p')
