## calculate the force on the bead as a function of its position relative to the cantilever position

import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.optimize as sp
import scipy.signal as sig
import matplotlib.colors as colors
import matplotlib.cm as cmx

cal_file = "/data/20150331/Bead6/2mbar_zcool_50mV_41Hz.h5"
#data_path = "/data/20150331/Bead6/tapping_close8"
data_path = "/data/20150331/Bead6/dipole_diff"
fname = "urmbar_xyzcool_1V_50mV_100Hz"
fidx = [0,1]
fdrive = 0.2

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

    ##xr -= np.median(xr)

    xr *= cal_fac_force
    yr *= cal_fac_force

    ## filter drive to reject noise
    b, a = sig.butter(2, 2.*20/Fs)
    xdf = sig.filtfilt(b,a,xd)

    tvec = np.linspace(0, 1.0*(len(xd)-1)/Fs, len(xd))

    ## also fit a sin wave
    spars = [1.4*np.std(xd),fdrive,np.pi/2,0]
    bp, bcov = sp.curve_fit(sin_fun, tvec, xd, p0=spars)

    xds = sin_fun(tvec, bp[0], bp[1], bp[2], bp[3])

    tvec_per = (2*np.pi*tvec*bp[1]+bp[2]) % (2*np.pi)

    # plt.figure()
    # plt.plot(tvec, xd)
    # # plt.plot(tvec, xdf, 'r')
    # plt.plot(tvec, xds, 'g')
    # plt.show()

    # plt.figure()
    # plt.plot(tvec_per, xds - xd, '.' )

    ntbins = 2000
    msvec = []
    for n in range(ntbins-1):
        gidx = np.logical_and(tvec_per < 2*np.pi*(n+1)/ntbins, tvec_per > 2*np.pi*n/ntbins)
        curr_dat = xd[gidx]
        curr_resp = xr[gidx]
        Npts = np.sum(gidx)
        xvals = xds[gidx]
        msvec.append([2*np.pi*(n+0.5)/ntbins, np.median(curr_dat), np.std(curr_dat)/np.sqrt(Npts), np.median(curr_resp), np.std(curr_resp)/np.sqrt(Npts), np.median(xvals)])
    msvec = np.array(msvec)
    # plt.errorbar( msvec[:,0], msvec[:,1], yerr=msvec[:,2], fmt='r', linewidth=1.5)


    # hh, be = np.histogram( xds - xd, bins=100 )
    # hh2, be2 = np.histogram( xdf - xd, bins=100 )
    # plt.figure()
    # plt.step(be[:-1], hh, 'k', where='post')
    # plt.step(be2[:-1], hh2, 'r', where='post')

    # plt.figure()
    # plt.plot( xd, xr, '.' )

    return msvec

nv = [-999,-0.4]
#col_list = ['k', 'b', 'r', 'g', 'c', 'm', 'y', 'k']

## make a list of colors for plotting
jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=len(nv))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
col_list = []
for n in range(len(nv)):
    col_list.append( scalarMap.to_rgba(n) )

plist = []
plt.figure()

for j,v in enumerate(nv):
    print "Working on V=%f"%v
    sval = "%.1f"%v
    sval = sval.replace(".", "_")
    if( v < -100 ):
        cfile = os.path.join(data_path, "urmbar_xyzcool_pulled_back_-0_4_50mV_41Hz.h5")
    else:
        cfile = os.path.join(data_path, "urmbar_xyzcool_pulled_foreward3_%s_50mV_41Hz.h5"%sval)
    if( not os.path.isfile(cfile) ): 
        print "Warning, couldn't find: ", cfile
        continue
    d1 = get_avg_dat( cfile )

    # plt.figure()
    # plt.plot( d1[:,-1], d1[:,3], '.')
    # plt.show()

    if( j == 0 ):
        xbins=np.linspace( np.min(d1[:,-1]), np.max(d1[:,-1]), 10 )
        
    cdat = []
    for i,x in enumerate(xbins[:-1]):
        gidx = np.logical_and(d1[:,-1] > xbins[i], d1[:,-1] < xbins[i+1])
        cdat.append([ (x+xbins[i+1])/2.0, np.median(d1[gidx,3]), np.std(d1[gidx,3])/np.sqrt(np.sum(gidx))])
    cdat = np.array(cdat)
    cdat[:,1] -= cdat[0,1]

    if( j > 0):
        cdat[:,1] -= cal_val
        cdat[:,2] = np.sqrt(cdat[:,2]**2 + cal_err**2)
    else:
        cal_val, cal_err = cdat[:,1], cdat[:,2]
        continue

    ##plt.figure()
    #plt.errorbar( d1[:,-1], d1[:,3], xerr=d1[:,2], yerr=d1[:,4], fmt='k.', linewidth=1.5)
    plt.errorbar( cdat[:,0], cdat[:,1], yerr=cdat[:,2], fmt='.', linewidth=1.5, color=col_list[j] )
    p = np.polyfit( cdat[:,0], cdat[:,1], 2 )
    xx=np.linspace( np.min(d1[:,-1]), np.max(d1[:,-1]), 1e3 )
    plt.plot(xx, np.polyval(p, xx), color=col_list[j], linewidth=1.5, label="%.2f V"%v)


    plist.append([v,p[0],p[1],p[2]])

# print plist
# plt.figure()
# for cp in plist:
#     plt.plot(xx, np.polyval(cp[1:], xx), label="V=%d"%cp[0], linewidth=1.5 )

plt.legend(loc="upper left")
# plt.figure()
# plt.errorbar( d1[:,-1], d1[:,3]-d2[:,3], xerr=np.sqrt(d1[:,2]**2+d2[:,2]**2), yerr=np.sqrt(d1[:,4]**2+d2[::-1,4]**2), fmt='k.', linewidth=1.5)
# plt.xlabel("Drive signal [arb units]")
# plt.ylabel("Force [N]")

plt.show()
