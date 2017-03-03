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
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib import dates

cal_file = "/data/20150603/Bead2/2mbar_zcool_leveled_400mV_41Hz_200mVDC.h5"
data_path = "/data/20150603/Bead2"

make_plots = False
dir = 'x'

dist = 10 ## um
freq = 0 ## Hz

dist_fac = 1.5e-3 ## rough factor to convert volts piezo displacement to um
mix_fac = 0.1 ## factor of force that gets mixed into the fundamental peak
corr_fac = 8. ##0.1*2.8**2 ## correction factor to account for changes in

cal_fac, bp_cal, _ = bu.get_calibration(cal_file, [50,500], True)
cal_fac_force = bu.bead_mass * (2*np.pi*bp_cal[1])**2 * cal_fac

xbins = np.linspace(-1000*dist_fac, 1000*dist_fac,2e2)
xcent = xbins[:-1] + np.diff(xbins)/2.0

def ffn(x, p0, p1, p2, p3):
    return p0*np.sin( 2*np.pi*p1*x + p2 ) + p3

def get_avg_dat( cfile, col, dist ):
    
    dat, attribs, cf = bu.getdata( cfile )

    if( len(dat) == 0 ):
        return None

    ##print attribs['displacement']
    Fs = attribs['Fsamp']    

    ct = bu.labview_time_to_datetime(attribs['time'])

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

    spars = [0.2*np.std(xd),drive_freq,np.pi/2,0]
    tvec = np.linspace(0, 1.0*(len(xd)-1)/Fs, len(xd))
    bp, bcov = sp.curve_fit(ffn, tvec, xmf, p0=spars)
    xds = ffn(tvec, bp[0], bp[1], bp[2], bp[3])

    # plt.figure()
    # plt.plot( tvec, xmf, 'k.' )
    # plt.plot( tvec,  xds, 'r')
    # plt.show()

    xds -= np.mean(xds)
    xds /= np.max(xds)


    corr_val = np.sum( xds * xr )

    fr, freqs = matplotlib.mlab.psd(xr, Fs = Fs, NFFT = 2**17) 

    min_bin = np.argmin( np.abs( freqs - (freq + drive_freq) ) )
    min_bin2 = np.argmin( np.abs( freqs - (freq - drive_freq) ) )

    if( make_plots ):
        plt.loglog(freqs,np.sqrt(fr*(freqs[1]-freqs[0])*corr_fac), color=col, label="%d um"%dist)
        yy = plt.ylim()
        plt.plot( [freqs[min_bin], freqs[min_bin]], yy, 'k--')
        plt.plot( [freqs[min_bin2], freqs[min_bin2]], yy, 'k--')
        plt.ylim( yy )

    #plt.loglog( freqs[min_bin], np.sqrt( fr[min_bin] ), 'x')

    sn_rat = np.sqrt( np.median(fr[min_bin+10:min_bin+20])*(freqs[1]-freqs[0])*1.0/mix_fac )

    ctemp = attribs['temps'][0]

    return np.sqrt( fr[min_bin] * (freqs[1]-freqs[0]) * 1.0/mix_fac), corr_val, ct, sn_rat, ctemp


#plist = [50,60,70,80,90,100,110,120,130]
#plist = [[10, 0],[10, 1.1],[10, 1.2],[10, 2.4],]
#plist = [[5,1.1],[5,0.8],[5,0.6],]
#dlist = [1000,300,100,50,40,30,20,10,7,5,3]
#dlist = [50,40,30,20,10,7,5,3]
## make a list of colors for plotting
jet = plt.get_cmap('jet') 

ccol = ['k','r','b','g','c','m','y']

def sort_fun(s):
    cv = re.findall("\d+.h5", s)
    cv = int( cv[0][0:-3] )
    return cv

f1 = plt.figure()
corr_list = []

fname = "cant2_1000um_overnight/urmbar_xyzcool_50mV_41Hz_450mVDC_*.h5"


flist = sorted(glob.glob( os.path.join(data_path, fname) ),key=sort_fun)

curr_list = []

for j,cf in enumerate(flist):

    if( j % 100 == 0): print "At file: ", j

    curr_val = get_avg_dat( os.path.join(data_path, cf), 'k', j ) 

    if( not curr_val ): continue
        
    curr_list.append( [curr_val[2], curr_val[1], curr_val[3], curr_val[-1]] )

curr_list = np.array(curr_list)

scale_fac = 8e-15/1e-10 ## rough scale factor to convert to N

#print curr_list
fig=plt.figure()
plt.errorbar( dates.date2num(curr_list[:,0]), curr_list[:,1]*scale_fac, yerr=curr_list[:,2], fmt='ko-', markersize=4 )
ax = plt.gca()
ax.xaxis.set_major_locator(dates.HourLocator())
hfmt = dates.DateFormatter('%m/%d %H:%M')
ax.xaxis.set_major_formatter(hfmt)
fig.autofmt_xdate()
plt.ylabel("Force [N]")

plt.savefig("resp_vs_time.pdf")

plt.figure()
plt.plot( dates.date2num(curr_list[:,0]), curr_list[:,-1], 'ro-', markersize=4 )

plt.show()
