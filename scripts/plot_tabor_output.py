import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import curve_fit
from amp_ramp_3 import flattop
from amp_ramp_3 import bp_filt
from ring_down_analysis_v3 import track_frequency
import bead_util_funcs as buf
import bead_util as bu
import hs_digitizer as hd
import os

files = ['/data/old_trap/20191105/bead4/phase_mod/deriv_feedback_1/turbombar_powfb_xyzcool_mod_pos_0.h5',\
         '/data/old_trap/20191105/bead4/phase_mod/deriv_feedback_1/turbombar_powfb_xyzcool_mod_neg_0.h5',\
         '/data/old_trap/20191105/bead4/phase_mod/deriv_feedback_1/turbombar_powfb_xyzcool_mod_no_dg_1_0.h5']
        
drive_ind = 1
drive_freq = 25.e3

filt_bandwidth = 20000
def sine(x, A, f0, c):
    w = 2 * np.pi * f0

    return A * np.sin(w*x + c)

def plot_efield(filename):
    obj = hd.hsDat(filename)

    Ns = obj.attribs['nsamp']
    Fs = obj.attribs['fsamp']

    t = np.arange(0, Ns/Fs, 1./Fs)
    
    drive_sig = obj.dat[:,drive_ind]
    
    drive_sig = bp_filt(drive_sig, drive_freq, Ns, Fs, filt_bandwidth)

    zeros = np.zeros(Ns)
    voltages = np.array([zeros, zeros, zeros, drive_sig, zeros, zeros, zeros,zeros]) 
    
    efield = buf.trap_efield(voltages*100.)

    fft = np.fft.rfft(efield[0])
    freqs = np.fft.rfftfreq(Ns, 1./Fs)


    plt.loglog(freqs,np.abs(fft))
    plt.show()

    freq_ind = np.argmax(np.abs(fft))
    freq_guess = freqs[freq_ind]
    amp_guess = np.sqrt(2) * np.std(efield[0])

    p0 = [amp_guess,freq_guess,0.1]
    

    popt,pcov = curve_fit(sine, t, efield[0], p0)

    print(popt)
    
    plt.plot(t, sine(t,*popt))
    plt.plot(t, efield[0])
    plt.show()


for i in range(len(files)):
    plot_efield(files[i])
