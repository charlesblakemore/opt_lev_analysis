import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re


#Ns = 500000
#Fs = 200000.
fi_init = 149377./2.
init_file = 90
final_file = 300
path = "/data/20181204/bead1/high_speed_digitizer/spindown/base_pressure_150kstart_1"
files = glob.glob(path + "/*.h5")

sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3]) 

files.sort(key = sfun)
bw = 400.
chop_pts = 1000
dec_fac = 13
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
nout = int(np.ceil((Ns-2*chop_pts)/(dec_fac)**2)) + 1
ifreqs = np.zeros((final_file-init_file, nout))
times = np.zeros((final_file-init_file, nout))
#inds = np.arange(Ns)
fc = 2.*fi_init
for i, f in enumerate(files[init_file:final_file]):
    print(i)
    obj = hsDat(f)
    if i !=0:
        fc = line(tarr0[-1]*(1./2.)+(obj.attribs['time']-t)/10**9, *popt)
    
    fb = np.abs(freqs-fc)>bw/2.
    fft = np.fft.rfft(obj.dat[:, 0])
    plt.loglog(freqs[np.logical_not(fb)], np.abs(fft[np.logical_not(fb)]), label = str(i))
    fft[fb] = 0.
    sig = np.fft.irfft(fft)
    phase = np.unwrap(np.angle(ss.hilbert(sig)))
    t = obj.attribs["time"]
    times[i, :] = dec2(tarr0[chop_pts:-chop_pts] + (t-t0)/10**9, dec_fac)
    inst_freqs = np.gradient(phase, 1./Fs)/(2.*np.pi)
    ifreqs[i, :] = dec2(inst_freqs[chop_pts:-chop_pts], dec_fac)
    popt, pcov = curve_fit(line, tarr0, inst_freqs)

plt.show()




