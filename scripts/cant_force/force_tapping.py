## calculate the force on the bead while cantilever position is tapped

import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.optimize as sp
import scipy.signal as sig
import matplotlib.colors as colors
import matplotlib.cm as cmx

cal_file = "/data/20150415/Bead5/2mbar_zcool_cal_100mV_41Hz.h5"
data_path = "/data/20150415/Bead5/test"

dist = 10 ## um
freq = 1.7 ## Hz

dist_fac = 1.5e-3 ## rough factor to convert volts piezo displacement to um



cal_fac, bp_cal, _ = bu.get_calibration(cal_file, [10,500], False)
cal_fac_force = bu.bead_mass * (2*np.pi*bp_cal[1])**2 * cal_fac

xbins = np.linspace(-1000*dist_fac, 1000*dist_fac,2e2)
xcent = xbins[:-1] + np.diff(xbins)/2.0

def get_avg_dat( cfile, dist, freq ):
    
    dat, attribs, cf = bu.getdata( cfile )
    ##print attribs['displacement']
    Fs = attribs['Fsamp']    

    xr, xd = dat[:,bu.data_columns[0]], dat[:,bu.drive_column]

    xr *= cal_fac_force
    xd *= dist_fac

    mu, sig = np.zeros_like(xcent), np.zeros_like(xcent)
    for i,(xl,xh) in enumerate(zip(xbins[:-1], xbins[1:])):
        gpts = np.logical_and( xd > xl, xd <= xh )
        mu[i], sig[i] = np.median( xr[gpts] ), np.std( xr[gpts] )/np.sqrt(np.sum(gpts))

    #plt.figure()
    #plt.plot( xd, xr, '.', markersize=1)
    #plt.errorbar(xcent, mu, yerr=sig, fmt='r')
    #plt.plot(xd)
    #plt.plot(xr)

    return mu, sig

plt.figure()
bias_list = [""]
for i,dist in enumerate([1,3,10,]):

    dstr = "%.1f"%dist
    dstr = dstr.replace(".0", "")
    dstr = dstr.replace(".","_")

    #fname = "nobead_%sum_5000mV_%dHz.h5"%(dstr,np.round(freq))
    #fname = flist[dist] + "_5000mV_2Hz.h5"

    #if( dist < 0 ):
    #    fname = "urmbar_xyzcool_2um_5000mV_%dHz.h5"%(np.round(freq))   
    #else:
    #    fname = "urmbar_xyzcool_blade%d_2um_5000mV_%dHz.h5"%(dist,np.round(freq))   

    #if(i == 0):
    #fname = "urmbar_xyzcool_%sum2_%smV_5000mV_%dHz.h5"%(dstr,bias_list[i],np.round(freq))
    #if(i >= 0):
    #    fname = "urmbar_xyzcool_%sum2_%smV_app_5000mV_%dHz.h5"%(dstr,bias_list[i],np.round(freq))
    #else:
    #    fname = "urmbar_xyzcool_%sum_blade%d_5000mV_%dHz.h5"%(dstr,i,np.round(freq))

    fname = "test_%dum_5000mV_2Hz.h5"%dist

    if( not os.path.isfile( os.path.join(data_path, fname) ) ): 
        print "warning, couldn't find: ", fname
        continue

    m, s = get_avg_dat( os.path.join(data_path, fname), dist, freq )
    #m, s = get_avg_dat( os.path.join(data_path, fname), 2, freq )

    med_val = np.median( m[:5] )

    plt.errorbar(-(xcent+xcent[-1])-dist, m-med_val, yerr=s, label="%.2f"%dist)
    #plt.errorbar(-(xcent+xcent[-1]), m-med_val, yerr=s, label="%.2f"%dist)

plt.legend(loc="upper left")
plt.show()
