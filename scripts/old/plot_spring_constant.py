import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as sp
import matplotlib.mlab as mlab


f1 = "/data/20150831/Bead1/trapmod/1_5mbar_zcool_nomod.h5"
f2 = "/data/20150831/Bead1/trapmod/1_5mbar_zcool_mody0_17.h5"

d1,_,_ = bu.getdata(f1)
d2,_,_ = bu.getdata(f2)

NFFT = 2**10

cal_fac1, bp1, _, = bu.get_calibration(f1, [50,500], make_plot=False, NFFT=NFFT, data_columns=[1,1])
cal_fac2, bp2, _, = bu.get_calibration(f2, [50,500], make_plot=False, NFFT=NFFT, data_columns=[1,1])

Fs = 5000
xpsd1, freqs = matplotlib.mlab.psd(d1[:, 1]-np.mean(d1[:,1]), Fs = Fs, NFFT = NFFT) 
xpsd2, freqs = matplotlib.mlab.psd(d2[:, 1]-np.mean(d2[:,1]), Fs = Fs, NFFT = NFFT) 

fig=plt.figure()
plt.loglog(freqs, np.sqrt(xpsd1)*cal_fac1*1e9, 'ko', markersize=3, markeredgecolor='k')
plt.loglog(freqs, np.sqrt(xpsd2)*cal_fac2*1e9, 'ro', markersize=3, markeredgecolor='r')

xx = np.linspace( 10, 500 )
norm_rat = (2*bu.kb*293)/(bu.bead_mass) * 1/bp1[0]
plt.loglog( xx, np.sqrt(norm_rat * bu.bead_spec_rt_hz( xx, bp1[0], bp1[1], bp1[2] )**2)*1e9, color='k', linewidth=1.5, label="No modulation, $f=%.0f$ Hz"%(bp1[1]))
norm_rat = (2*bu.kb*293)/(bu.bead_mass) * 1/bp2[0]
plt.loglog( xx, np.sqrt(norm_rat * bu.bead_spec_rt_hz( xx, bp2[0], bp2[1], bp2[2] )**2)*1e9, 'r', linewidth=1.5, label="With modulation, $f=%.0f$ Hz"%(bp2[1]))

plt.xlabel("Freq. [Hz]")
plt.ylabel("X PSD [nm Hz$^{-1/2}$]")

plt.xticks([50,60,70,80,90,100,200,300,400,500])
plt.gca().set_xticklabels([50,60,"",80,"",100,200,300,400,500])

plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
plt.gca().set_yticklabels(["","","","","",1,"","","","","","","","",10,""])

plt.xlim([50, 500])
plt.ylim([0.5,20])

plt.legend(loc="lower left", prop={"size": 11})

fig.set_size_inches(4.5,3.4)
plt.subplots_adjust(bottom=0.13, top=0.99, right=0.96)
plt.savefig("spring_const_mod.pdf")

plt.show()
