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


load_from_file = False
show_each_file = False
show_avg_force = False
fft = False
calibrate = True

get_bead_mass = True
bits_per_pi = 2000
bits_per_volt = (2**15 - 1) / 10.
micron_per_pi = 1.064 / 4.

######################
## Build Charge Cal ##
######################

# [[1,1,1,1,1,-1,2,1,1,1],[12,15,30,51,84,93,99,117,129,465],0.0001]
charge_cal = [['/data/20170627/bead4/discharge_finalsteps'], 'Cal', 0]

charge_cal_dir_obj = cu.Data_dir(charge_cal[0], [0,0,charge_cal[2]], charge_cal[1])
#charge_cal_dir_obj.load_dir(cu.simple_loader)
#charge_cal_dir_obj.build_step_cal_vec(pcol = 0, files = [0,1000])
#charge_cal_dir_obj.step_cal(amp_gain = 1.)
#charge_cal_dir_obj.save_step_cal('./calibrations/step_cal_20170627.p')
charge_cal_dir_obj.load_step_cal('./calibrations/step_cal_20170627.p')

#for fobj in charge_cal_dir_obj.fobjs:
#    fobj.close_dat()

#charge_cal_dir_obj.load_H("./trans_funcs/Hout_20160718.p")

#charge_cal_dir_obj.calibrate_H()
charge_cal_dir_obj.get_conv_facs()

######################
### Build Therm Cal ##
######################


therm_path = '/data/20170627/bead4/1_6mbar_nocool.h5'


charge_cal_dir_obj.thermal_cal_file_path = therm_path
charge_cal_dir_obj.thermal_calibration()
charge_cal_dir_obj.thermal_cal_fobj.plt_thermal_fit()

if get_bead_mass:
    zfit = charge_cal_dir_obj.thermal_cal_fobj.thermal_cal[2]
    print zfit.popt
    A = zfit.popt[0]
    w0 = 2. * np.pi * zfit.popt[1]

    volts_per_meter = (1. / (0.25 * 1064e-9)) * 2000. * (1. / 3276.7)

    MTratio = 2 * bu.kb / (A * w0**2) * volts_per_meter**2
    mass = MTratio * 450 * 10**12

    print "M/T Ratio: %0.2g kg/K" % MTratio
    print "Implied Mass at 300 K: %0.2g ng" % mass

# Resonant Frequencies

fits = charge_cal_dir_obj.thermal_cal_fobj.thermal_cal

freqs = []
for i in [0,1,2]:
    freqs.append(fits[i].popt[1])

print
print "X, Y and Z resonant freqs from thermal cal"
print freqs


######################
### Compare Calibs ###
######################

m = bu.bead_mass

charge_step_facs = charge_cal_dir_obj.conv_facs

therm_facs = charge_cal_dir_obj.thermal_cal_fobj.get_thermal_cal_facs()

print
print "Calibration Comparison"
print charge_step_facs
print therm_facs
#print therm_facs * np.sqrt(2)









