import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import curve_fit
from amp_ramp_3 import flattop, bp_filt, hp_filt
from ring_down_analysis_v3 import track_frequency
import bead_util_funcs as buf
import bead_util as bu
import hs_digitizer as hd
import os

matplotlib.rcParams['figure.figsize'] = [7,5]
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['agg.path.chunksize'] = 10000

directory = '/data/old_trap/20191223/bead1/spinning/tests/noise_inject/'

libr_freq = 579
bandwidth = 50

files, length = bu.find_all_fnames(directory, ext='.h5', sort_time=True)

obj = hd.hsDat(files[0])
print(files[0])

Ns = obj.attribs['nsamp']
Fs = obj.attribs['fsamp']

tarr = np.arange(Ns)/Fs
freqs = np.fft.rfftfreq(Ns, 1./Fs)

sig = obj.dat[:,0]

window = np.hanning(len(sig))

z = ss.hilbert(sig)

phase = ss.detrend(np.unwrap(np.angle(z)))


phase_filt = bp_filt(phase, libr_freq, Ns, Fs, bandwidth)

fft = np.fft.rfft(phase_filt)

plt.loglog(freqs, np.abs(fft))
plt.show()

z_phase_filt = ss.hilbert(phase_filt * window)

phase_phase_filt = np.unwrap(np.angle(z_phase_filt))

inst_freq_phase_filt = np.gradient(phase_phase_filt)/(2*np.pi) * Fs

plt.plot(tarr, inst_freq_phase_filt)
plt.show()



