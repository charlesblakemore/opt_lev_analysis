import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re
import matplotlib

#Ns = 500000
#Fs = 200000.
path = "/data/20181030/bead1/high_speed_digitizer/golden_data/amp_ramp_50k_good"
files = glob.glob(path + "/*.h5")
fi_init = 1e5
init_file = 0
final_file = len(files)
n_file = final_file-init_file

sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3]) 

files.sort(key = sfun)
bw = 2000.
bw_sb = 0.02
obj0 = hsDat(files[init_file])
t0 = obj0.attribs['time']
Ns = obj0.attribs['nsamp']
Fs = obj0.attribs['fsamp']
freqs = np.fft.rfftfreq(Ns, d = 1./Fs)
tarr0 = np.linspace(0, Ns/Fs, Ns)



def line(x, m, b):
    return m*x + b

def dec2(arr, fac):
    return ss.decimate(ss.decimate(arr, fac), fac)

def sqrt_fun(x, a):
    return a*np.sqrt(x)


fc = fi_init
plot_dat = True

matplotlib.rcParams.update({'font.size':12})
f, ax = plt.subplots(dpi = 200)
files = np.array(files)
inds = [0, 100, 200, 300, 400, 499]
files = files[inds]
labels = ["62.5kV/m", "50.0kV/m", "37.5kV/m", "25.0kV/m", "12.5kV/m", "0.0kV/m"]
files = list(files)
p_bool = np.abs(freqs-fc)<bw
freqs /= 1000
fc/=1000
bw/=1000
for i, f in enumerate(files):
    print i
    try:
        obj = hsDat(f)
        fft = np.fft.rfft(obj.dat[:, 0])
        if plot_dat:
            ax.plot(freqs, np.abs(fft), label = labels[i])
    except:
        print "bad file"

ax.set_yscale("log")
ax.set_xlim([fc-bw/2., fc+bw/2.])
plt.xlabel("Frequency[kHz]")
plt.ylabel("Optical Power [arb]")
plt.legend()
plt.tight_layout()
plt.show()
