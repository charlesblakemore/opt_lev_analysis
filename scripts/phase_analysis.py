import matplotlib.pyplot as plt
import numpy as np
import bead_util as bu
import os
import matplotlib.mlab

path = "/data/20180605/bead1/discharge/coarse2"

f0 = "turbombar_xyzcool_elec3_3000mV41Hz0mVdc_0.h5"
f1 = "turbombar_xyzcool_elec3_3000mV41Hz0mVdc_50.h5"

df0 = bu.DataFile()
df1 = bu.DataFile()

df0.load(os.path.join(path, f0))
df1.load(os.path.join(path, f1))

def phixy(df):
    phix = ((df.phase[2]+df.phase[3])-(df.phase[0]+df.phase[1]))/2.

    phiy = ((df.phase[2]+df.phase[0])-(df.phase[3]+df.phase[1]))/2.

    return phix, phiy

phix, phiy = phixy(df1)

psdphix, freqs = matplotlib.mlab.psd(phix, NFFT = 2**14, Fs = 5E3, detrend = 'linear')

psdphiy, freqs = matplotlib.mlab.psd(phiy, NFFT = 2**14, Fs = 5E3, detrend = 'linear')

psdpx , freqs = matplotlib.mlab.psd(df1.pos_data[0], NFFT = 2**14, Fs = 5E3, detrend = 'linear')


psdpy , freqs = matplotlib.mlab.psd(df1.pos_data[1], NFFT = 2**14, Fs = 5E3, detrend = 'linear')


plt.loglog(freqs, psdphix, label = "phix")
plt.loglog(freqs, psdphiy, label = "phiy")
plt.loglog(freqs, psdpx, label = "x pos")
plt.loglog(freqs, psdpy, label ="y pos")
plt.xlabel("freq[Hz]")
plt.ylabel("psd[arb]")
plt.legend()
plt.show()
