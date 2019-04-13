import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re, sys

save = True

#path = "/data/20181204/bead1/high_speed_digitizer/pramp/50k_zhat_8vpp_3"

path = '/daq2/20190408/bead1/spinning/49kHz_200Vpp_pramp-N2_1'

drive_ax = 0
data_ax = 1

out_f = "/processed_data/spinning/pramp_data/49k_200vpp"
bu.make_all_pardirs(out_f)

files, lengths = bu.find_all_fnames(path, sort_time=True)

fi_init = 49000
init_file = 0
final_file = len(files)
n_file = final_file-init_file

bw = 5.
obj0 = hsDat(files[init_file])
t0 = obj0.attribs['time']
Ns = obj0.attribs['nsamp']
Fs = obj0.attribs['fsamp']

tarr0 = np.linspace(0, (Ns-1)/Fs, Ns)

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
    bu.progress_bar(i, n_file)
    sys.stdout.flush()
    try:
        obj = hsDat(f)
        fft = np.fft.rfft(obj.dat[:, data_ax])
        fft2 = np.fft.rfft(obj.dat[:, drive_ax])
        fft[bfreq] = 0.
        fft2[bfreq2] = 0.
        if plot_dat:
            plt.loglog(freqs, np.abs(fft))
            plt.loglog(freqs, np.abs(fft2))
            plt.show()
        phases[i] = np.angle(np.sum(fft))
        dphases[i] = np.angle(np.sum(fft2))
        pressures[i, :] = obj.attribs["pressures"]
        times[i] = obj.attribs['time']
    except:
        print "bad file"

dphases[dphases<0.]+=np.pi


phi = phases - 2.*dphases
phi[phi>np.pi]-=2.*np.pi
phi[phi<-np.pi]+=2.*np.pi

if save:
    np.save(out_f + '_phi.npy', phi)
    np.save(out_f + "_pressures.npy", pressures)
    np.save(out_f + "_time.npy", times)
