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

dirs = [36,]
cal = 5.0e-14

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )

load_from_file = False



def proc_dir(d):
    dv = ddict[d]
    print dv

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-2]], dv[1])
    dir_obj.load_dir(cu.pos_loader, maxfiles=30)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)

for obj in dir_objs:
    print "Charge Calibration:"
    obj.step_cal(obj)

    obj.load_H("./trans_funcs/Hout_20160613.p")

    obj.get_conv_facs()
    print obj.conv_facs







##### Compare to thermal calibration

cal_obj = cu.Data_file()
cal_obj.load("/data/20160613/bead1/1_5mbar_nocool.h5", [0,0,20])

cal_obj.thermal_calibration()
cal_obj.plt_thermal_fit()

norm_rats = cal_obj.get_thermal_cal_facs()

print
print "Thermal Calibration"
print [np.sqrt(norm_rats[0]), np.sqrt(norm_rats[1]), np.sqrt(norm_rats[2])]


