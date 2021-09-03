from hs_digitizer import *
import bead_util as bu
import bead_util_funcs as buf
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy import signal
from amp_ramp_3 import bp_filt
from amp_ramp_3 import hp_filt
directory = '/data/old_trap/20201030/bead1/spinning/dds_phase_impulse_6Vpp/trial_0005/'
directory = '/data/old_trap/20200924/bead1/spinning/dds_phase_impulse_6Vpp/trial_0005/'

files, lengths = bu.find_all_fnames(directory, sort_time=True)

rot_freq = 25e3
bw = 8000
lib_bw = 250
lib_freq_cut = 200
data_ind = 0
downs_num = 500


def get_data(obj, ind):
    return obj.dat[:,ind], obj.attribs['nsamp'],\
            obj.attribs['fsamp']
def get_phase(sig):
    z = signal.hilbert(sig)

    phase = np.angle(z)
    phase = np.unwrap(phase)
    phase = signal.detrend(phase)
    
    return phase

def find_lib_freq(phase, freqs, freq):
    phase_fft = np.fft.rfft(phase)
   
    max_amp = -2*16
    for i, point in enumerate(np.abs(phase_fft)):
        if freq !=0 and freqs[i] < freq:
            continue
        if point > max_amp:
            max_amp = point
            max_ind = i
    #plt.loglog(freqs, np.abs(phase_fft))
    #plt.scatter(freqs[max_ind], np.abs(phase_fft)[max_ind],\
    #color='r')
    #plt.show()


    return freqs[max_ind]
    
def create_tarr(obj):
    sig, Ns, Fs = get_data(obj, data_ind)
    tarr = np.arange(Ns)/Fs
    return tarr

def measure_damp_coef(files):
    
    phase_arr = []
    time_arr = []
    for i, f in enumerate(files):
        obj = hsDat(f)
        sig, Ns, Fs = get_data(obj, data_ind)
        freqs = np.fft.rfftfreq(Ns, 1./Fs)
        
        sig = bp_filt(sig, 2*rot_freq, Ns, Fs, bw)
        amp, phase = buf.demod(sig, rot_freq, Fs, filt=True, \
            bandwidth=bw, plot=True, detrend=True, \
            tukey=True, tukey_alpha=5e-4)
        #phase = get_phase(sig)
        lib_freq = find_lib_freq(phase, freqs, lib_freq_cut)
      
        phase = bp_filt(phase, lib_freq, Ns, Fs, lib_bw)
        
        tarr = create_tarr(obj)

        tarr += obj.attribs['time']*1e-9
        
        phase = buf.rebin_mean(phase, Ns/downs_num)
        tarr = buf.rebin_mean(tarr, Ns/downs_num)
       
        window = signal.tukey(len(phase), alpha = 5.0e-2)
        z = signal.hilbert(phase * window)
        amp = np.abs(z)

        plt.plot(tarr, phase, color='b')
        plt.plot(tarr, amp, color = 'r')
        if i == 25:
            plt.show()
        print(i)

def measure_damp_coef_parallel(f):
    
    obj = hsDat(f)
    sig, Ns, Fs = get_data(obj, data_ind)
    freqs = np.fft.rfftfreq(Ns, 1./Fs)
    
    sig = bp_filt(sig, 2*rot_freq, Ns, Fs, bw)
    phase = get_phase(sig)
    lib_freq = find_lib_freq(phase, freqs, lib_freq_cut)
    
    phase = bp_filt(phase, lib_freq, Ns, Fs, lib_bw)
    
    tarr = create_tarr(obj)

    tarr += obj.attribs['time']*1e-9
    
    phase = buf.rebin_mean(phase, Ns/downs_num)
    tarr = buf.rebin_mean(tarr, Ns/downs_num)
    
    z = signal.hilbert(phase)
    amp = np.abs(z)



#Parallel(n_jobs=5)(delayed(measure_damp_coef_parallel)(f) for f in files)
measure_damp_coef(files)
