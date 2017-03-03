## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

## first make calibrated spectrum
noise_path = "/data/20140801/Bead6/plates_terminated"
spec_path = "/data/20140801/Bead6/chargelp_cal"
single_charge_fnums = [0, 60]

## first get avg spectrum from noise path
nfiles = glob.glob( noise_path + "/*.h5")
jx2,_,_,Jfreqs = bu.get_avg_noise( nfiles, 0, [0,0,0], make_plot=False, norm_by_sum=False )
sfiles = sorted(glob.glob( spec_path + "/*Hz*.h5"), key=bu.find_str)
sfiles = sfiles[single_charge_fnums[0]:single_charge_fnums[1]]
#jx,_,_,Jfreqs = bu.get_avg_noise( sfiles, 0, [0,0,0], make_plot=False, norm_by_sum=False )

print "freq diff: ", np.median( np.diff( Jfreqs ) )

## now get calibration from fit to 2 mbar file
cal_2mb = "/data/20140801/Bead6/2mbar_zcool_50mV_41Hz.h5"
cal_fac, bp_cal, _ = bu.get_calibration(cal_2mb, [10,200], True)
##plt.show()

no_bead_files = ["/data/20140801/Bead6/urmbar_xyzcool_no_bead_50mV_no_synth.h5",]
jx3,_,_,Jfreqs3 = bu.get_avg_noise( no_bead_files, 0, [0,0,0], make_plot=False, norm_by_sum=False )

jx3int = np.interp(Jfreqs, Jfreqs3, jx3)
jxSub = jx2 - jx3int

final_fig = plt.figure()
#plt.loglog(Jfreqs, jx)
plt.loglog(Jfreqs, cal_fac*np.sqrt(jx2)*1e9, 'k', label="Total, meas.")
plt.loglog(Jfreqs3, cal_fac*np.sqrt(jx3)*1e9, 'r', label="Imaging, meas.")
plt.loglog(Jfreqs, cal_fac*np.sqrt(jxSub)*1e9, 'b', label="Laser + vib.")
plt.xlim([10,1e3])
plt.ylim([3e-2,5])
plt.xlabel("Frequency [Hz]")
plt.ylabel("X position PSD [nm Hz$^{-1/2}$]")
plt.legend(prop={"size": 13})

final_fig.set_size_inches(5,4)
plt.subplots_adjust(left=0.14, bottom=0.13, top=0.99, right=0.97)
plt.savefig("noise_floor.pdf")

plt.show()

