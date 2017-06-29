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
import time

####################################################
####### Input parameters for data processing #######

TESTING = True

ddict = bu.load_dir_file( "/home/charles/opt_lev_analysis/scripts/dirfiles/dir_file_june2017.txt" )
#print ddict

pow_axis = 4

cant_axis = 1           # stage control axis
straighten_axis = 2     # axis with coherent drive to straighten
fit_pows = True

load_charge_cal = True
maxfiles = 1000

plot_forward_backward = False #True
#subtract_background = True

drivefreq = 18.0
cant_volts_to_um = 8.0    # 80 um / 10 V

#fig_title = ('Force vs. Cantilever Position: %s Hz, %s - %s, ' + bead) % (drivefreq, gas, num)

#dirs = [1,2,3,4,5,6,7]
dirs = [8,9,10,11,12,13,14,15,16]



tf_path = './trans_funcs/Hout_20160808.p'

step_cal_path = './calibrations/step_cal_20160808.p'

thermal_cal_file_path = '/data/20160808/bead1/1_5mbar_zcool_final.h5'




def poly2(x, a, b, c):
    return a * (x - b)**2 + c



def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader, maxfiles = maxfiles)

    amps = []
    for fil_obj in dir_obj.fobjs:
        fil_obj.psd()
        stagestuff = fil_obj.get_stage_settings(axis=straighten_axis)
        amp = stagestuff[2] * cant_volts_to_um
        amps.append(amp)
    uamps = np.unique(amps)
    if len(uamps) > 1:
        print 'STUPIDITYERROR: Multiple dirve amplitudes in directory'
        
    newlist = []
    for i in [0,1,2]:
        if i == straighten_axis:
            newlist.append(uamps[0])
        else:
            newlist.append(0.0)
    dir_obj.drive_amplitude = newlist

    return dir_obj


dir_objs = map(proc_dir, dirs)

colors_yeay = bu.get_color_map( len(dir_objs) )

psds = {}
pows = {}
bpows = {}

for ind, obj in enumerate(dir_objs):
    psd = []

    col = colors_yeay[ind]

    amp = obj.drive_amplitude[straighten_axis]

    filcount = 0
    for fobj in obj.fobjs:
        filcount += 1
        fobj.psd()
        if not len(psd):
            freqs = fobj.other_psd_freqs
            psd = fobj.other_psds[pow_axis-3]
        else:
            psd += fobj.other_psds[pow_axis-3]

    psd = psd / float(filcount)
    psds[amp] = psd
    
    ind = np.argmin(np.abs(freqs - drivefreq))

    totpow = np.sum(psd[ind-1:ind+2])
    pows[amp] = totpow

    badind = int(ind*1.5)
    totbadpow = np.sum(psd[badind-1:badind+2])
    bpows[amp] = totbadpow
    

amps = pows.keys()
amps.sort()
powsarr = []
bpowsarr = []
for amp in amps:

    powsarr.append(pows[amp])
    bpowsarr.append(bpows[amp])



if fit_pows:
    p0 = [1, 0, 0]
    popt, pcov = curve_fit(poly2, amps, powsarr, p0 = p0, maxfev = 10000)
    
    fitpoints = np.linspace(amps[0], amps[-1], 100)
    fit = poly2(fitpoints, *popt)

    plt.plot(amps, powsarr, 'o')
    plt.plot(fitpoints, fit, color='r', linewidth=1.5)

    title = 'Best fit straightening amplitude: %0.2g um' % popt[1]
    plt.title(title)

else:
    plt.plot(amps, powsarr)
    plt.plot(amps, bpowsarr)

plt.show()

