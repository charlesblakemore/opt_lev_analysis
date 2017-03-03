## calculate the force on the bead while cantilever position is tapped

import glob, re, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.optimize as sp
import scipy.signal as sig
import matplotlib.colors as colors
import matplotlib.cm as cmx

cal_file = "/data/20150603/Bead2/2mbar_zcool_leveled_400mV_41Hz_200mVDC.h5"
data_path = "/data/20150603/Bead2/anglelp"

dir = 'x'

dist = 10 ## um
freq = 0 ## Hz

dist_fac = 1.5e-3 ## rough factor to convert volts piezo displacement to um
mix_fac = 0.1 ## factor of force that gets mixed into the fundamental peak
corr_fac = 8. ##0.1*2.8**2 ## correction factor to account for changes in

cal_fac, bp_cal, _ = bu.get_calibration(cal_file, [50,500], False)
cal_fac_force = bu.bead_mass * (2*np.pi*bp_cal[1])**2 * cal_fac

xbins = np.linspace(-1000*dist_fac, 1000*dist_fac,2e2)
xcent = xbins[:-1] + np.diff(xbins)/2.0

def ffn(x, p0, p1, p2, p3):
    return p0*np.sin( 2*np.pi*p1*x + p2 ) + p3

def get_avg_dat( cfile, col, dist ):
    
    dat, attribs, cf = bu.getdata( cfile )
    ##print attribs['displacement']
    Fs = attribs['Fsamp']    

    if( dir == 'x' ):
        dcol = 0
    elif( dir == 'y' ):
        dcol = 1
    elif( dir == 'z' ):
        dcol = 2
    else:
        print "Warning, direction must be x, y, or z"
        dcol = None

    xr, xd, xm = dat[:,bu.data_columns[dcol]], dat[:,bu.drive_column], dat[:,3]
    #xr, xd, xm = dat[:,2], dat[:,bu.drive_column], dat[:,3]

    xr *= cal_fac_force
    xd *= dist_fac

    ## find modulation envelope
    b,a = sig.butter(3, np.array([100])/(Fs/2.0), btype="low")
    xmf = sig.filtfilt(b,a,np.abs(xm-np.mean(xm)))

    xmft = np.fft.rfft( xmf-np.mean(xmf) )
    freqs = np.fft.rfftfreq( len(xmf) ) * Fs

    ## get frequency of tone
    drive_freq = freqs[np.argmax( np.abs(xmft) )]
    drive_phase = np.angle( xmft[drive_freq] )

    spars = [1.4*np.std(xd),drive_freq,np.pi/2,0]
    tvec = np.linspace(0, 1.0*(len(xd)-1)/Fs, len(xd))
    bp, bcov = sp.curve_fit(ffn, tvec, xmf, p0=spars)
    xds = ffn(tvec, bp[0], bp[1], bp[2], bp[3])
    xds -= np.mean(xds)
    xds /= np.max(xds)

    # plt.figure()
    # plt.plot( tvec, xmf, 'k.' )
    # plt.plot( tvec,  xds, 'r')
    # plt.show()

    corr_val = np.sum( xds * xr )

    fr, freqs = matplotlib.mlab.psd(xr, Fs = Fs, NFFT = 2**20) 

    plt.loglog(freqs,np.sqrt(fr*(freqs[1]-freqs[0])*corr_fac), color=col, label="%d um"%dist)

    min_bin = np.argmin( np.abs( freqs - (freq + drive_freq) ) )
    min_bin2 = np.argmin( np.abs( freqs - (freq - drive_freq) ) )

    yy = plt.ylim()
    plt.plot( [freqs[min_bin], freqs[min_bin]], yy, 'k--')
    plt.plot( [freqs[min_bin2], freqs[min_bin2]], yy, 'k--')
    plt.ylim( yy )

    #plt.loglog( freqs[min_bin], np.sqrt( fr[min_bin] ), 'x')
    
    return np.sqrt( fr[min_bin] * (freqs[1]-freqs[0]) * 1.0/mix_fac), corr_val


#plist = [50,60,70,80,90,100,110,120,130]
#plist = [[10, 0],[10, 1.1],[10, 1.2],[10, 2.4],]
#plist = [[5,1.1],[5,0.8],[5,0.6],]
plist = [[20,0],[50,0],[80,0]]
## make a list of colors for plotting
jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=len(plist))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
col_list = []
for n in range(len(plist)):
    col_list.append( scalarMap.to_rgba(n) )

plt.figure()
corr_list = []
for i,p in enumerate(plist):
    #pstr = str(p[0])
    #pstr = pstr.replace(".","_")

    #vstr = str(p[1])
    #vstr = "%.2f"%p[1]
    #vstr = vstr.replace(".","_")

    fname = "urmbar_xyzcool_relev_y%d_z2268_50mV_41Hz_450mVDC.h5"%(p[0])

    print fname
    curr_val = get_avg_dat( os.path.join(data_path, fname), col_list[i], p[0] ) 

    corr_list.append( [p[0], curr_val[1]] )

    #plt.plot( 10-p, curr_val, 'ko')


corr_list = np.array(corr_list)

plt.legend()
plt.xlim([1,100])
#plt.xlim((freq)+np.array([-3,3]))
#plt.xlim([20, 30])
#plt.ylim([1e-16, 1e-13])
plt.xlabel("Freq. [Hz]")
plt.ylabel("Mixed force per bin [N]")

plt.figure()
plt.plot(corr_list[:,0], corr_list[:,1], 'o-')
plt.ylabel("Correlation [arb units]")
plt.xlabel("Voltage [V]")

plt.show()
