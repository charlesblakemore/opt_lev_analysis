import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re

save = True

path = "/data/20181204/bead1/high_speed_digitizer/pramp/50k_zhat_8vpp_3"
out_f = "processed_data/20181204/pramp_data/50k_8vpp"
files = glob.glob(path + "/*.h5")
fi_init = 5e4
init_file = 0
final_file = len(files)
n_file = final_file-init_file

sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3]) 

files.sort(key = sfun)
bw = 5.
obj0 = hsDat(files[init_file])
t0 = obj0.attribs['time']
Ns = obj0.attribs['nsamp']
Fs = obj0.attribs['fsamp']

tarr0 = np.linspace(0, Ns/Fs, Ns)

def line(x, m, b):
    return m*x + b

def dec2(arr, fac):
    return ss.decimate(ss.decimate(arr, fac), fac)

freqs = np.fft.rfftfreq(Ns, d = 1./Fs)
times = np.zeros(n_file)
phases = np.zeros(n_file)
dphases = np.zeros(n_file)
pressures = np.zeros((n_file, 3))

fc = 2.*fi_init
bfreq = np.abs(freqs-fc)>bw/2.
bfreq2 = np.abs(freqs-fc/2.)>bw/2.

plot_dat = False

for i, f in enumerate(files[init_file:final_file]):
    print i
    try:
        obj = hsDat(f)
        fft = np.fft.rfft(obj.dat[:, 0])
        fft2 = np.fft.rfft(obj.dat[:, 1])
        fft[bfreq] = 0.
        fft2[bfreq2] = 0.
        if plot_dat:
            plt.loglog(freqs, np.abs(fft))
            plt.loglog(freqs, np.abs(fft2))
            plt.show()
        phases[i] = np.angle(np.sum(fft))
        dphases[i] = np.angle(np.sum(fft2))
        pressures[i, :] = obj.attribs["pressure"]
        times[i] = obj.attribs['time']
#plt.show()
    except:
        print "bad file"

dphases[dphases<0.]+=np.pi


phi = phases - 2.*dphases
phi[phi>np.pi]-=2.*np.pi
phi[phi<-np.pi]+=2.*np.pi
if save:
    np.save(out_f + 'phi.npy', phi)
    np.save(out_f + "pressures.npy", pressures)
    np.save(out_f + "time.npy", times)
