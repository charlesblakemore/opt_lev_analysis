import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu

NFFT = 2**18
fig=plt.figure()

cal_fac = 1.6e-15/125. * 3276.7   ## N/bit

path = "/data/20140803/Bead8/chargelp_cal"
f = "urmbar_xyzcool_200Hz_95.h5"

## now old setup
ypsd = []
path = "/data/20140803/Bead8/no_charge"
for nn in range(0,20,2):
    f = "urmbar_xyzcool_2500mV_RANDFREQ_%d.h5"%nn
    print f
    cdat,attribs,_ = bu.getdata(os.path.join(path, f))
    Fs = attribs['Fsamp']

    nam='Old setup'
    cypsd, freqs = matplotlib.mlab.psd(cdat[:, 0]-numpy.mean(cdat[:, 0]), Fs = Fs, NFFT = NFFT)
    if( len(ypsd)==0 ):
        ypsd = cypsd
    else:
        ypsd += cypsd
ypsd /= 10.

plt.loglog(freqs, np.sqrt(ypsd)*cal_fac, color='r', linewidth=1.5, label=nam)



path = "/data/20150831/Bead3/lowfreq_noise/"

flist = ["urmbar_discharged_lid_on.h5",]

name_list = ["Old setup",]


cal_fac = 1.6e-15/150. * 3276.7  ## N/bit

f = flist[0]
cdat,attribs,_ = bu.getdata(os.path.join(path, f))
Fs = attribs['Fsamp']

col='k'
nam='New setup'
ypsd, freqs = matplotlib.mlab.psd(cdat[:, 1]-numpy.mean(cdat[:, 1]), Fs = Fs, NFFT = NFFT)
plt.loglog(freqs, np.sqrt(ypsd)*cal_fac, color=col, linewidth=1.5, label=nam)



plt.xlim([0.5, 4e2])
plt.ylim([5e-19, 1e-14])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Noise PSD [N Hz$^{-1/2}$]")
plt.legend()

plt.subplots_adjust(bottom=0.125,top=0.95,right=0.98) 

fig.set_size_inches(6,4.5)
plt.savefig("noise_psd_old_setup.pdf")

plt.show()
