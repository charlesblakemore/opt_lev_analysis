import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from hs_digitizer import *
from scipy.optimize import curve_fit
import matplotlib
import re

g = 50.*0.66
s = 0.004

#in_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/ampramp_data_0/"
in_base_fname = "amp_ramp_50k"
in_path = '/home/dmartin/analyzedData/20190514/amp_ramp/'


amps = np.load(in_path+in_base_fname + "amps.npy")
print(amps)
amps*=g
amps/=s


freqs = np.load(in_path+in_base_fname + "wobble_freq.npy")
#freqs/=2
freqs*=2.*np.pi
def sqrt_fun(x, poi, toi):
        return np.sqrt(x*poi)
    
popt, pcov = curve_fit(sqrt_fun, amps, freqs)

plt_scale = 1000
matplotlib.rcParams.update({'font.size':14})
f, ax = plt.subplots(dpi = 200, sharex = True, figsize = (5, 3))
#ax.axvline(x = f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "50kHz rotation frequency")
ax.plot(amps/plt_scale, freqs, '.', markersize = 2)
ax.plot(amps/plt_scale, sqrt_fun(amps, *popt), 'r',alpha = 0.5,  label = r"$(d/I) = 108 \pm 2$ s$\cdot$A/(kg$\cdot$m)", linestyle = '--', linewidth = 5)
#ax.plot(amps/plt_scale, freqs, '.', markersize = 2)
ax.legend()
ax.set_ylabel(r"$\omega_{\phi}$ [rad/s]")
#ax.set_xlim([2.*f_rot-2, 2.*f_rot+2])
ax.set_xlabel("E [kV/m]")
#plt.ylabel("Sideband Frequency [Hz]")
ax.legend()
plt.subplots_adjust(top = 0.96, bottom = 0.15, left = 0.15, right = 0.99)
plt.tight_layout()
#f.savefig("/home/arider/plots/20181221/small_oscillations.png", dpi = 200)
plt.show()




