import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
import scipy.signal as ss
from scipy import interpolate


tarrt = np.load("trash_npy_files/t_100kHz_spindown.npy")
freq_arr = np.load("trash_npy_files/freq_100kHz_spindown.npy")

t = tarrt.flatten()
freqs = freq_arr.flatten()
freqs/=2

finterp = interpolate.interp1d(t, freqs)
npts = 1000000
t = np.linspace(0.01, np.max(t)-0.01, npts)
freqs = finterp(t)

tb1 = t<900.
tb2 = t>900.

def exp_fun(x, t, a, b):
    return a*np.exp(x/(-t)) + b

popt, pcov = curve_fit(exp_fun, t[tb1], freqs[tb1], p0 = [100., 8e4, 0.])

tstr = str(popt[0])[:3]
ststr = str(np.sqrt(pcov[0, 0]))[:4]
flab =  r'$\tau$ = ' +  tstr + ' s'


matplotlib.rcParams.update({'font.size':14})
f, ax = plt.subplots(dpi = 200, figsize = (5, 3))
ax.plot(t, freqs*2.*np.pi/1e3, '.', markersize = 2)
ax.plot(t[tb1], exp_fun(t[tb1], *popt)*2.*np.pi/1e3, 'r',alpha = 0.5,  linewidth = 5, linestyle = "--", label = flab)
ax.set_xticks([0, 1500, 3000, 4500, 6000])
#ax.plot(t, freqs*2.*np.pi/1e3, '.', markersize = 2)
#ax.plot(t[tb2], freqs/1e3)
#ax.set_xscale('log')
plt.xlabel("Time [s]")
plt.ylabel(r"$\omega_{MS}$ [krad/s]")
plt.legend(fontsize = 12)
plt.subplots_adjust(top = 0.91, bottom = 0.14, left = 0.15, right = 0.92, hspace = 0.6)
plt.tight_layout()
plt.show()
f.savefig("/home/arider/plots/20181221/spindown.png", dpi = 200)



