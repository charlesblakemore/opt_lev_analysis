import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from hs_digitizer import *
from scipy.optimize import curve_fit
import matplotlib
import re

in_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/tests/freq_sweep"
in_base_fname = "freq_sweep"


c_freqs = np.load(in_path+in_base_fname + "c_freqs.npy")
c_freqs/=2.
lsb_freqs = np.load(in_path+in_base_fname + "lsb_freqs.npy")

def sqrt_fun(x, c1, c2, c3):
        return np.sqrt(c1*(c2-x)) + c3
    

#plt_scale = 1000
fmin = 10000
b = c_freqs>fmin
c_freqs = c_freqs[b]
lsb_freqs = lsb_freqs[b]

p0 = [1., 2e5, 1.]
popt, pcov = curve_fit(sqrt_fun, c_freqs, lsb_freqs, p0 = p0)


matplotlib.rcParams.update({'font.size':12})
f, ax = plt.subplots(2, 1, dpi = 200, sharex = True)
#ax.axvline(x = f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "50kHz rotation frequency")
ax[0].plot(c_freqs/1e3, lsb_freqs, 'o')
ax[0].plot(c_freqs/1e3, sqrt_fun(c_freqs, *popt), 'r', label = r"$\sqrt{k(f_{0}-f}) + c_{1}}+ c_{2}$")
ax[0].legend()
ax[0].set_ylabel("Sideband Frequency [Hz]")
ax[1].plot(c_freqs/1e3, sqrt_fun(c_freqs, *popt)-lsb_freqs, 'o')
ax[1].set_ylabel("Residual [Hz]")
#ax.set_xlim([2.*f_rot-2, 2.*f_rot+2])
plt.xlabel("Rotation Frequency [kHz]")
#plt.ylabel("Sideband Frequency [Hz]")
plt.legend()
plt.show()




