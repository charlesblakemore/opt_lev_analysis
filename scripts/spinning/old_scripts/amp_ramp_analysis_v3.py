import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re
import matplotlib

save = True

path = "/data/20181204/bead1/high_speed_digitizer/amp_ramp/50k_zhat"
out_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/ampramp_data_0/"
out_base_fname = "amp_ramp_50k"
files = glob.glob(path + "/*.h5")
fc = 1e5
bw = 1e3
init_file = 0
final_file = len(files)
n_file = final_file-init_file
ns = 1
sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3]) 

files.sort(key = sfun)
files = np.array(files)
obj0 = hsDat(files[init_file])
t0 = obj0.attribs['time']
Ns = obj0.attribs['nsamp']
Fs = obj0.attribs['fsamp']
freqs = np.fft.rfftfreq(Ns, d = 1./Fs)
tarr0 = np.linspace(0, Ns/Fs, Ns)
freq_bool = np.abs(freqs-fc)>bw
d_amps = np.zeros(n_file)
f_wobs = np.zeros(n_file)
f_wob = 427.39
bwa = 10.
sbw = 0.5

def line(x, m, b):
    return m*x + b

def dec2(arr, fac):
    return ss.decimate(ss.decimate(arr, fac), fac)



matplotlib.rcParams.update({'font.size':12})
f, ax = plt.subplots(dpi = 200)


for i, f in enumerate(files[init_file:final_file:ns]):
    print(i)
    try:
        obj = hsDat(f)
        d_amps[i] = obj.attribs["network amp"]
        fft = np.fft.rfft(obj.dat[:, 0])
        fft[freq_bool] = 0.
        a_sig = ss.hilbert(np.fft.irfft(fft))
        phase = ss.detrend(np.unwrap(np.angle(a_sig)))
        fft_phase = np.fft.rfft(phase)
        b_init = np.abs(freqs-f_wob)<bwa
        f_wobi = freqs[b_init][np.argmax(np.abs(fft_phase[b_init]))]
        b_small = np.abs(freqs-f_wobi)<sbw
        f_wob = np.average(freqs[b_small], weights = np.abs(fft_phase[b_small])**2)
        f_wobs[i] = f_wob
        #plt.loglog(freqs, np.abs(fft_phase))
        #plt.xlim([0, bw])
        #plt.show()

    
    except IOError:
        print("bad file")


def sqrt_fun(x, poi, toi):
    return np.sqrt(x*poi + toi)

if save:
    np.save(out_path + out_base_fname + "amps", d_amps)
    np.save(out_path + out_base_fname + "wobble_freq", f_wobs)

