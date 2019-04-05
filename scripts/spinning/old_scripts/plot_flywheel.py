import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.signal as ss
from scipy.optimize import curve_fit
import re

plot_drive = False
path = "/data/20180927/bead1/spinning/flywheel_1211Hz"

miner = lambda arr, x: np.argmin(np.abs(arr-x))

files = bu.find_all_fnames(path)

df = bu.DataFile()
df.load(files[1])
df.load_other_data()
df.diagonalize()
s_off = 59176
p_ind = 0
block = 25000
fcent = 1210.7
bw = 150.
fmin = fcent-bw/2.
fmax = fcent + bw/2.
Fs = 5000.
lint = 50.
k = 1E-13*(2.*np.pi*370.)**2


if plot_drive:
    si = 10
    sf = -50
    b, a = ss.butter(1, 0.1)
    ds = df.other_data[2]
    dc = np.fft.irfft(np.fft.rfft(ds)*np.exp(-1.j*np.pi/2.))
    a = ss.filtfilt(b, a, np.sqrt(ds**2 + dc**2))
    scale = 200./0.004
    t = np.linspace(0, lint, lint*Fs)[si:sf]
    plt.plot(t, a[si:sf]*scale)
    plt.xlabel("time[s]")
    plt.ylabel("Drive Electric Field Amplitude [V/m]")
    plt.show()

ntotal = len(df.pos_data[0])
plt_blocks = True
freqs = np.fft.rfftfreq(block, d = 1./Fs)

li = miner(freqs, fmin)
ui = miner(freqs, fmax)

freq_wind = freqs[li:ui]
nblocks = int(np.floor((ntotal-s_off)/block))

def lab(i, t_block = 5.):
    if i == 0:
        return "Drive on"
    else:
        return str((i-1)*t_block) + 's-'+ str((i)*t_block) + 's'


def anal_signal(arr):
    anal_sig = ss.hilbert(arr)
    amp = np.abs(anal_sig)
    phi = np.unwrap(np.angle(anal_sig))
    return amp, phi 

def line(x, m, b):
    return m*x + b

def get_drive_phase(arr, end = s_off, make_plot = True):
    amp, phi = anal_signal(arr[:end])
    popt, pcov = curve_fit(line, np.arange(len(phi)), phi)
    if make_plot:
        amp, phi = anal_signal(arr)
        plt.plot(phi, label = "data phase")
        plt.plot(line(np.arange(len(phi)), *popt), label = "linear extrapolation")
        plt.legend()
        plt.xlabel("sample")
        plt.ylabel("phase [rad]")
        plt.show()
    return popt


def get_drive_ind(darr, guess, kern = 59, window = 5, aw = 2):
    dft = np.fft.rfft(darr)
    mf = ss.medfilt(np.abs(dft), kernel_size = kern)
    snr = np.abs(dft)/mf
    inds = np.arange(len(dft))
    snr[np.abs(inds-guess)>window]=0
    d_ind =  np.argmax(snr)
    d_amp = np.abs(np.sum(dft[np.abs(np.arange(len(dft))-d_ind)<aw]))/len(dft)
    return np.argmax(snr), d_amp



#def diff_phase(dataf, bw = 10.):

if plt_blocks:
    f, axarr = plt.subplots(nblocks + 1, 1, sharex = True, sharey = True)
    for i in range(nblocks+1):
        fft =df.conv_facs[p_ind]*np.abs(np.fft.rfft(df.pos_data[p_ind, block*(i-1) + s_off:block*(i) + s_off]))*10**12
        axarr[i].plot(freq_wind, fft[li:ui], label = lab(i))
        axarr[i].legend(loc = 1)
        axarr[i].axvline(x = 1210.7, linestyle = '--', color = 'k')
        if i==4:
            axarr[i].set_ylabel("amplitude [arb]")

    plt.xlabel("frequency[Hz]")
    #plt.tight_layout()
    #plt.subplots_adjust(hspace = 10.)
    plt.show()




