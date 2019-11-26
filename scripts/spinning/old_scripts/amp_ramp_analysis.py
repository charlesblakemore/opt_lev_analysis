import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re


#Ns = 500000
#Fs = 200000.
path = "/data/20181030/bead1/high_speed_digitizer/golden_data/amp_ramp_50k_good"
out_f = "processed_data/golden/ampramp_data/amp_ramp_50k"
files = glob.glob(path + "/*.h5")
fi_init = 1e5
init_file = 0
final_file = len(files)
n_file = final_file-init_file

sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3]) 

files.sort(key = sfun)
bw = 2000.
obj0 = hsDat(files[init_file])
t0 = obj0.attribs['time']
Ns = obj0.attribs['nsamp']
Fs = obj0.attribs['fsamp']

tarr0 = np.linspace(0, Ns/Fs, Ns)

def line(x, m, b):
    return m*x + b

def dec2(arr, fac):
    return ss.decimate(ss.decimate(arr, fac), fac)

def sqrt_fun(x, a):
    return a*np.sqrt(x)

freqs = np.fft.rfftfreq(Ns, d = 1./Fs)
times = np.zeros(n_file)
phases = np.zeros(n_file)
dphases = np.zeros(n_file)
d_amps = np.zeros(n_file)
r_amps = np.zeros(n_file)
pressures = np.zeros((n_file, 3))

fc = fi_init
bfreq = np.abs(freqs-fc)>bw/2.
bfreq2 = np.abs(freqs-fc/2.)>bw/2.
plot_dat = True

matplotlib.rcParams.update({'font.size':12})
f, ax = plt.subplots(dpi = 200)
files = np.array(files)

for i, f in enumerate(files[init_file:final_file:100]):
    print(i)
    try:
        obj = hsDat(f)
        fft = np.fft.rfft(obj.dat[:, 0])
        if plot_dat:
            ax.plot(freqs, np.abs(fft))
    except:
        print("bad file")

ax.set_yscale("log")
ax.set_xlim([fc-bw/2., fc+bw/2.])
plt.show()
