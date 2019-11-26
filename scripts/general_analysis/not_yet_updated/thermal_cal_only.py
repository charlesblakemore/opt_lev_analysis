import cant_util as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import os, sys
from scipy.optimize import curve_fit
import bead_util as bu

get_bead_mass = True
bits_per_pi = 2000

temp = 450


cal_dir_obj = cu.Data_dir('', [0,0,0], 'Therm Cal')


######################
### Build Therm Cal ##
######################


therm_path = '/data/20170903/bead1/1_5mbar_nocool.h5'


cal_dir_obj.thermal_cal_file_path = therm_path
cal_dir_obj.thermal_calibration(temp=temp)
cal_dir_obj.thermal_cal_fobj.plt_thermal_fit()

if get_bead_mass:
    zfit = cal_dir_obj.thermal_cal_fobj.thermal_cal[2]
    #print zfit.popt
    A = zfit.popt[0]
    w0 = 2. * np.pi * zfit.popt[1]

    volts_per_meter = (1. / (0.25 * 1064e-9)) * bits_per_pi * (1. / 3276.7)

    MTratio = 2 * bu.kb / (A * w0**2) * volts_per_meter**2
    mass = MTratio * temp * 10**12  # convert the mass to nanograms

    print("M/T Ratio: %0.2g kg/K" % MTratio)
    print("Implied Mass at %i K: %0.2g ng" % (temp, mass))

# Resonant Frequencies

fits = cal_dir_obj.thermal_cal_fobj.thermal_cal

freqs = []
for i in [0,1,2]:
    freqs.append(fits[i].popt[1])

print()
print("X, Y and Z resonant freqs from thermal cal")
print(freqs)



therm_facs = cal_dir_obj.thermal_cal_fobj.get_thermal_cal_facs(temp=temp)

print()
print("Calibration")
print(therm_facs)
#print therm_facs * np.sqrt(2)









