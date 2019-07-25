import numpy as np
import matplotlib.pyplot as plt
import hs_digitizer as hd
import matplotlib.mlab

from scipy import signal
import bead_util_funcs as buf #Need to import this after hs_digitizer for some reason
import amp_ramp_3
path = '/daq2/20190626/bead1/spinning/wobble/wobble_many/wobble_0000/'

files, zero = buf.find_all_fnames(path)

print(files[10])
obj = hd.hsDat(files[10])

Ns = obj.attribs['nsamp']
Fs = obj.attribs['fsamp']

drive_freq = 50e3

fft = np.fft.rfft(obj.dat[:,1])
freqs=np.fft.rfftfreq(Ns,1/Fs)

low_freq = (drive_freq - 1000)/freqs[-1]
high_freq = (drive_freq + 1000)/freqs[-1]


b, a = signal.butter(2,[low_freq,high_freq],btype='bandpass')

sig = signal.filtfilt(b,a,obj.dat[:,1])

win = amp_ramp_3.flattop(len(sig))

psd_filt, freqs = matplotlib.mlab.psd(sig, Ns, Fs, window=win)

N = len(np.abs(fft))

#plt.loglog(freqs,np.abs(psd_filt))
#plt.show()

amp = 2*np.sum(np.sqrt(psd_filt))

print(amp)

