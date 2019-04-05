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
path = "/data/20181030/bead1/high_speed_digitizer/freq_steps/div_4_base_pressure"
out_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/tests/freq_sweep"
out_base_fname = "freq_sweep"
files = glob.glob(path + "/*.h5")
init_file = 0
final_file = len(files)
n_file = final_file-init_file

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

def freq_to_ind(f):
    return np.argmin(np.abs(freqs-f))

def line_center(fft, i_guess, bw_small = 2): 
    return np.sum(freqs[i_guess-bw_small:i_guess+bw_small]*np.abs(fft[i_guess-bw_small:i_guess+bw_small]))/\
            np.sum(np.abs(fft[i_guess-bw_small:i_guess+bw_small]))


def find_biggest_line_ind(fft, center, bw = 3):
    return np.argmax(np.abs(fft[center-bw:center+bw])) + center - bw



plot_dat = True

matplotlib.rcParams.update({'font.size':12})
f, ax = plt.subplots(dpi = 200)
files = np.array(files)
files = list(files)
lsb_freqs = []
c_freqs = []
sb_freq = 495.52
sb_ind = np.argmin(np.abs(freqs-sb_freq))
plot = False

for i, f in enumerate(files[::-1]):
    print i
    try:
        obj = hsDat(f)
        fft = np.fft.rfft(obj.dat[:, 0])
        fft2 = np.fft.rfft(obj.dat[:, 1])
        fft2[freqs>55e3] = 0.
        cent_ind = freq_to_ind(4.*freqs[np.argmax(fft2)])
        i_guess_cent = find_biggest_line_ind(fft, cent_ind)   
        d_freq = line_center(fft, i_guess_cent)
        sb_ind = freq_to_ind(d_freq-sb_freq)
        i_guess_sb = find_biggest_line_ind(fft, sb_ind)
        sb_cf = line_center(fft, i_guess_sb)
        sb_freq = d_freq-sb_cf
        lsb_freqs.append(sb_freq)
        c_freqs.append(d_freq)

        if plot:
            plt.semilogy(freqs[i_guess_sb-5000:i_guess_sb+5000], np.abs(fft[i_guess_sb-5000:i_guess_sb+5000]))
            plt.axvline(x = freqs[i_guess_sb], color = 'r')
            plt.axvline(x = sb_cf)
            plt.axvline(x = d_freq, color = 'k')
            plt.show()
    except IOError:
        print "bad file"


c_freqs = np.array(c_freqs)
lsb_freqs = np.array(lsb_freqs)



np.save(out_path + out_base_fname + "c_freqs", c_freqs)
np.save(out_path + out_base_fname + "lsb_freqs", lsb_freqs)

