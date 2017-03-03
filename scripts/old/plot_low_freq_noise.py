import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu

path = "/data/20150831/Bead3/lowfreq_noise/"

flist = ["urmbar_discharged_lid_off_hepa_on.h5", "urmbar_discharged_lid_on.h5","urmbar_discharged_lid_on_nobead.h5"]

name_list = ["Cover off", "Cover on", "Imaging noise"]

col_list = ['r','k',[0.5,0.5,0.5]]

NFFT = 2**18

## convert V to N
calib_file = "/data/20150831/Bead3/chargelp/URmbar_xyzcool3_elec3_400mV41Hz400mVdc_58.h5"

cdat,attribs,_ = bu.getdata(calib_file)
Fs = attribs['Fsamp']
#plt.figure()
#plt.plot(cdat[:,1])
#plt.show()

cal_fac = 1.6e-15/150.  ## N/bit

fig=plt.figure()
for nam,col,f in zip(name_list,col_list,flist):

    cdat,attribs,_ = bu.getdata(os.path.join(path, f))
    Fs = attribs['Fsamp']

    ypsd, freqs = matplotlib.mlab.psd(cdat[:, 1]-numpy.mean(cdat[:, 1]), Fs = Fs, NFFT = NFFT) 

    plt.loglog(freqs, np.sqrt(ypsd)*cal_fac, color=col, linewidth=1.5, label=nam)

plt.xlim([1e-2, 4e2])
plt.ylim([5e-19, 1e-14])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Noise PSD [N Hz$^{-1/2}$]")
plt.legend()

plt.subplots_adjust(bottom=0.125,top=0.95,right=0.98) 

fig.set_size_inches(6,4.5)
plt.savefig("noise_psd.pdf")

plt.show()
