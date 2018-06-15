import matplotlib.pyplot as plt
import numpy as np
import bead_util as bu
import bead_util_funcs as buf
import os
import matplotlib.mlab

path = "/data/20180613/bead1/dipole_vs_height/10V_70um_17hz"

fs = buf.find_all_fnames(path)

df0 = bu.DataFile()
df1 = bu.DataFile()

df0.load(fs[0])
df1.load(fs[-20])

def phixy(df, make_plot = True, NFFT = 2**14):
    phix = ((df.phase[2]+df.phase[3])-(df.phase[0]+df.phase[1]))/2.

    phiy = ((df.phase[2]+df.phase[0])-(df.phase[3]+df.phase[1]))/2.

    psdphix, freqs = matplotlib.mlab.psd(phix, NFFT = NFFT, \
            Fs = df.fsamp, detrend = 'linear')

    psdphiy, freqs = matplotlib.mlab.psd(phiy, NFFT = NFFT, \
            Fs = df.fsamp, detrend = 'linear')
    plt.loglog(freqs, psdphix, label = 'phi_x')
    plt.loglog(freqs, psdphiy, label = 'phi_y')
    return phix, phiy


phix, phiy = phixy(df1)
phix, phiy = phixy(df0)
plt.legend()
plt.show()

