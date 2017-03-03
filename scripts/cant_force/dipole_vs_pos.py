## calculate the force on the bead as a function of its position relative to the cantilever position

import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import bead_util as bu
import scipy.optimize as sp
import scipy.signal as sig

cal_file = "/data/20150331/Bead6/2mbar_zcool_50mV_41Hz.h5"
#data_path = "/data/20150331/Bead6/tapping_close8"
data_path = "/data/20150331/Bead6/dipole2"
fname = "urmbar_xyzcool_8V_50mV_100Hz"
fidx = [0,1]
fdrive = 100

cal_fac, bp_cal, _ = bu.get_calibration(cal_file, [10,500], True)

cal_fac_force = bu.bead_mass * (2*np.pi*bp_cal[1])**2 * cal_fac

def sin_fun(x, p0, p1, p2, p3):
    return p0*np.sin(2*np.pi*x*p1 +p2) + p3

def get_avg_dat( cfile ):
    
    dat, attribs, cf = bu.getdata( cfile )
    ##print attribs['displacement']
    Fs = attribs['Fsamp']    

    xr, xd = dat[:,bu.data_columns[0]], dat[:,bu.aod_columns[0]]
    yr, yd = dat[:,bu.data_columns[1]], dat[:,bu.aod_columns[1]]
    vd = dat[:,bu.drive_column]

    xr *= cal_fac_force
    yr *= cal_fac_force

    ## buttersworth filter to voltage drive
    bpb, bpa = sig.butter(2, [2.*99/Fs, 2*101./Fs], btype='bandpass')

    tvec = np.linspace(0, 1.0*(len(xd)-1)/Fs, len(xd))    

    xrf = sig.filtfilt(bpb, bpa, xr)

    ## filter drive to reject noise
    b, a = sig.butter(2, 2.*20/Fs)
    xdf = sig.filtfilt(b,a,xd)

    ## also fit a sin wave
    spars = [1.4*np.std(xd),2,np.pi/2,0]
    bp, bcov = sp.curve_fit(sin_fun, tvec, xd, p0=spars)

    xds = sin_fun(tvec, bp[0], bp[1], bp[2], bp[3])

    tvec_per = (2*np.pi*tvec*bp[1]+bp[2]) % (2*np.pi)

    pts_per_step = int(Fs)/50
    ## step through bins of time and find the rms power around the drive
    msvec = []
    for n in range(0, len(xd), pts_per_step):
        cdat = xr[pts_per_step*n:pts_per_step*(n+1)]
        cdrive = vd[pts_per_step*n:pts_per_step*(n+1)]
        cx = xds[pts_per_step*n:pts_per_step*(n+1)]
        corr_full = bu.corr_func(cdrive, cdat, Fs, fdrive)
        msvec.append( [cx, corr_full[0]])

    return msvec

d1 = get_avg_dat( os.path.join(data_path, fname+".h5") )
#d2 = get_avg_dat( os.path.join(data_path, fname+"_%d.h5"%fidx[1]) )



plt.show()
