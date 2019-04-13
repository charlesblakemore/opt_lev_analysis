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
out_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/golden/ampramp_data"
out_base_fname = "amp_ramp_50k_good"
files = glob.glob(path + "/*.h5")
fi_init = 1e5
init_file = 0
final_file = len(files)
n_file = final_file-init_file
save = False
sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3]) 

files.sort(key = sfun)

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



fc = 1e5
bw = 3# fft bins
plot_dat = True

matplotlib.rcParams.update({'font.size':12})
f, ax = plt.subplots(dpi = 200)
files = np.array(files)
files = list(files)
wobbles = np.zeros(len(files)) 
amps = np.zeros(len(files))
b_freqs = np.abs(freqs-fc)>2000.
plot = False
labels = ["62.0kV/m", "49.9kV/m", "37.3kV/m", "24.9kV/m", "12.4kV/m", "0kV/m"]
for i, f in enumerate(files[:1]):
    print i
    try:
        obj = hsDat(f)
        fft = np.fft.rfft(obj.dat[:, 0])
        fft[b_freqs] = 0.
        phase = np.unwrap(np.angle(ss.hilbert(np.fft.irfft(fft))))
        d_phase = ss.detrend(phase)
        fft_phase = np.fft.rfft(d_phase)
        ig = np.argmax(np.abs(fft_phase))
        wobbles[i] = np.average(freqs[ig-bw:ig+bw], weights = np.abs(fft_phase[ig-bw:ig+bw]))
        amps[i] = obj.attribs["network amp"]
        if (i%100==0)*(i==len(files)):
            ax.plot(freqs[freqs<1000], np.abs(fft_phase[freqs<1000]))#/len(fft), label = labels[i])
    except IOError:
        print "bad file"
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Phase Modulation [rad]")
plt.show()


if save:
    np.save(out_path + out_base_fname + "amps", amps)
    np.save(out_path + out_base_fname + "sb_freqs", wobbles)

