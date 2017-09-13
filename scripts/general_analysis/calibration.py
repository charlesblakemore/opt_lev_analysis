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




dirs = [9,]
ddict = bu.load_dir_file( "/dirfiles/dir_file_sept2017.txt" )

load_charge_cal = True
step_cal_path = '/calibrations/step_cals/step_cal_20170903.p'
cal_dir = '/data/20170903/bead1/discharge_fine'

thermal_path = '/data/20170906/bead6/1_5mbar_zcool.h5'

date = '20170903'
save = False

maxfiles = 141

#################################

# [[1,1,1,1,1],[51,225,291,303,330],0.0003]

if not load_charge_cal:
    cal = [[cal_dir], 'Cal', 15]

    cal_dir_obj = cu.Data_dir(cal[0], [0,0,cal[2]], cal[1])
    cal_dir_obj.load_dir(cu.simple_loader, maxfiles=maxfiles)
    cal_dir_obj.build_step_cal_vec()#files=[70,140])
    cal_dir_obj.step_cal(amp_gain = 1.)
    if save:
        cal_dir_obj.save_step_cal('/calibrations/step_cals/step_cal_'+date+'.p')

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

    dir_obj.calibrate_H(step_cal_drive_channel = 0)

    dir_obj.thermal_cal_file_path = thermal_path
    dir_obj.thermal_calibration()

    dir_obj.build_Hfuncs(fpeaks=[400, 400, 50], weight_peak=False, weight_lowf=True,\
                         plot_fits=True, plot_inits=False, weight_phase=True, grid=True,\
                         deweight_peak=True)#, fit_osc_sum=True)
    
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
        obj.save_H('/calibrations/transfer_funcs/Hout_'+date+'.p')
