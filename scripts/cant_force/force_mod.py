## calculate the force on the bead while cantilever position is tapped

import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.optimize as sp
import scipy.signal as sig
import matplotlib.colors as colors
import matplotlib.cm as cmx

cal_file = "/data/20150421/Bead1/2mbar_xyzcool_5000mV_5Hz.h5"
data_path = "/data/20150421/Bead1/cantdrive"

dist = 6.5 ## um
freq = 5.1 ## Hz

dist_fac = 1.5e-3 ## rough factor to convert volts piezo displacement to um

cal_fac, bp_cal, _ = bu.get_calibration(cal_file, [50,500], False)
cal_fac_force = bu.bead_mass * (2*np.pi*bp_cal[1])**2 * cal_fac

xbins = np.linspace(-1000*dist_fac, 1000*dist_fac,2e2)
xcent = xbins[:-1] + np.diff(xbins)/2.0

def ffn(x, p0, p1, p2, p3):
    return p0*np.sin( 2*np.pi*p1*x + p2 ) + p3

def get_avg_dat( cfile ):
    
    dat, attribs, cf = bu.getdata( cfile )
    ##print attribs['displacement']
    Fs = attribs['Fsamp']    

    xr, xd, xm = dat[:,bu.data_columns[0]], dat[:,bu.drive_column], dat[:,3]

    xr *= cal_fac_force
    xd *= dist_fac

    ## find modulation envelope
    b,a = sig.butter(3, np.array([100])/(Fs/2.0), btype="low")
    xmf = sig.filtfilt(b,a,np.abs(xm-np.mean(xm)))

    xmft = np.fft.rfft( xmf-np.mean(xmf) )

    ## get frequency of tone
    drive_freq = 0.5*np.argmax( np.abs(xmft) )/(len(xmft))
    drive_phase = np.angle( xmft[drive_freq] )

    xx = np.arange(0, len(xmf))
    ## fit a sin wave
    spars = [np.std(xmf)*1.4, drive_freq, drive_phase, 0.5]
    bp, bcov = sp.curve_fit(ffn, xx, xmf, p0=spars)
    
    kvt = 1 + 0.1*ffn(xx, 1., bp[1], bp[2], -1.)
    fvt = xr 

    fvf = np.fft.rfft( fvt )
    fs = np.fft.rfftfreq( len(fvt) )*Fs

    ## make stupid comb
    fcomb = []
    for n in range(1,100):
        fcomb.append( np.argmin( np.abs( fs - n*freq ) ) )

    fvf_filt = np.zeros_like(fvf)
    for idx in fcomb:
        fvf_filt[idx] = fvf[idx]

    fvx_filt = np.fft.irfft( fvf_filt )
    #fvx_filt = np.fft.irfft( fvf )
    
    xr -= np.mean(xr)

    mu, sg = np.zeros_like(xcent), np.zeros_like(xcent)
    for i,(xl,xh) in enumerate(zip(xbins[:-1], xbins[1:])):
        gpts = np.logical_and( xd > xl, xd <= xh )
        mu[i], sg[i] = np.median( xr[gpts] ), np.std( xr[gpts] )/np.sqrt(np.sum(gpts))

    muf, sgf = np.zeros_like(xcent), np.zeros_like(xcent)
    for i,(xl,xh) in enumerate(zip(xbins[:-1], xbins[1:])):
        gpts = np.logical_and( xd > xl, xd <= xh )
        muf[i], sgf[i] = np.median( fvx_filt[gpts] ), np.std( fvx_filt[gpts] )/np.sqrt(np.sum(gpts))

    plt.figure()
    #plt.plot( xd, xr, 'k.', markersize=1)
    #plt.errorbar(xcent, mu, yerr=sg, fmt='r')
    #plt.plot( xd, fvx_filt, 'b.', markersize=1)
    #plt.errorbar(xcent, muf, yerr=sgf, fmt='g')

    #plt.plot(xd/np.max(xd))
    #plt.plot(xr/np.max(xr))
    #plt.plot(xm-np.mean(xm))
    plt.loglog( fs, np.abs(fvf) )
    #plt.loglog( fs, np.abs(fvf_filt) )
    #plt.plot( xd, fvx_filt, '.', markersize=1 )

    return mu, sg

vlist = [-50, ] #[-100, -70, -50, -30, 0]
for v in vlist:
    dstr = str( dist )
    dstr = dstr.replace(".","_")

    vstr = str(v)
    vstr = vstr.replace("-","n")

    fname = "urmbar_xyzcool_%sum_%smV_5000mV_%dHz.h5"%(dstr,vstr,freq)

    get_avg_dat( os.path.join(data_path, fname) ) 

plt.show()
