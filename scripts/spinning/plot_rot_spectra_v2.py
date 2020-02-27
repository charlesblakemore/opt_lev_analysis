from hs_digitizer import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import window_func as window
import bead_util as bu
from amp_ramp_3 import bp_filt, lp_filt, hp_filt
from scipy import signal
from transfer_func_util import damped_osc_amp 
from scipy.optimize import curve_fit
from memory_profiler import profile
from memory_profiler import memory_usage
from plot_phase_vs_pressure_many_gases import build_full_pressure


import gc

mpl.rcParams['figure.figsize'] = [7,5]
mpl.rcParams['figure.dpi'] = 150

directory = '/data/old_trap/20200130/bead1/spinning/series_5/change_phi_offset_0_to_0_3_dg/long_int/long_int_0000/'
#directory = '/data/old_trap/20200130/bead1/spinning/series_3/base_press/change_phi_offset/0_dg/'
files, lengths = bu.find_all_fnames(directory, sort_time=True)

files = files[-2:]

bandwidth = 100
libration_freq = 360
data_ind = 0

plot_rot_sig_bool = False
plot_phase_vs_freq = True
plot = False
plot_phase_v_time_multiple = True
filt = False


def plot_phase_v_time(files):
    for i, f in enumerate(files):
        obj = hsDat(f)

        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        sig = obj.dat[:,data_ind]
        
        tarr = np.arange(Ns)/Fs

        z = signal.hilbert(sig)

        phase = signal.detrend(np.unwrap(np.angle(z)))

        if filt:
            phase = bp_filt(phase, libration_freq, Ns, Fs, bandwidth)
            
        plt.plot(tarr, phase)
        plt.show()

def plot_phase_v_time_multiple(files):
    for i, f in enumerate(files):
        print(f)
        obj = hsDat(f)

        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        sig = obj.dat[:,data_ind]

        tarr = np.arange(Ns)/Fs

        z = signal.hilbert(sig)

        phase = signal.detrend(np.unwrap(np.angle(z)))

        psd, freqs = mlab.psd(phase, NFFT=len(phase), Fs=Fs, window=mlab.window_hanning(phase))

        if filt:
            phase = bp_filt(phase, libration_freq, Ns, Fs, bandwidth)

        plt.plot(tarr, phase)
    plt.show()

def plot_phase_v_freq(files):
    for i, f in enumerate(files):
        obj = hsDat(f)

        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        sig = obj.dat[:,data_ind]

        tarr = np.arange(Ns)/Fs

        z = signal.hilbert(sig)

        phase = signal.detrend(np.unwrap(np.angle(z)))

        psd, freqs = mlab.psd(phase, NFFT=len(phase), Fs=Fs, window=mlab.window_hanning(phase))

        plt.loglog(freqs, psd)
        plt.show()

def plot_rot_sig(files):
    for i, f in enumerate(files):
        obj = hsDat(f)

        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        sig = obj.dat[:,data_ind]

        tarr = np.arange(Ns)/Fs
        freqs = np.fft.rfftfreq(Ns, 1./Fs)
        #psd, freqs = mlab.psd(sig, NFFT=len(sig),Fs=Fs, window=mlab.window_none(sig))#mlab.window_hanning(sig))
        fft = np.fft.rfft(sig)

        plt.loglog(freqs, np.abs(fft))
        #plt.loglog(freqs, np.sqrt(psd))
        plt.show()
 
if plot_rot_sig_bool:
    print('plot_rot_sig')
    plot_rot_sig(files)

if plot_phase_vs_freq:
    print('plot_phase_v_freq')
    plot_phase_v_freq(files)
if plot:
    print('plot_phase_v_time')
    plot_phase_v_time(files)
if plot_phase_v_time_multiple:
    print('plot_phase_v_time_multiple')
    plot_phase_v_time_multiple(files)
    
