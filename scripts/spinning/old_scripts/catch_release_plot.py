import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.signal as ss

path = "/data/20180927/bead1/spinning/catch_release_20181016"
s_off = 30438
s_on = 121023
ns = 250000
Fs = 5000.
fc = 1210.7
bw = 150


files = bu.find_all_fnames(path)

df = bu.DataFile()
df.load(files[1])
df.load_other_data()


freqs0 = np.fft.rfftfreq(s_off, d = 1./Fs)
freqs1 = np.fft.rfftfreq(s_on-s_off, d = 1./Fs)
freqs2 = np.fft.rfftfreq(ns-s_on, d = 1./Fs)
fft0 = np.fft.rfft(df.pos_data[0, :s_off])
b0 = np.abs(freqs0-fc)<bw/2.
b1 = np.abs(freqs1-fc)<bw/2.
b2 = np.abs(freqs2-fc)<bw/2.
fft1 = np.fft.rfft(df.pos_data[0, s_off:s_on])
fft2 = np.fft.rfft(df.pos_data[0, s_on:])
plt.plot(freqs0[b0], np.abs(fft0)[b0], label = "drive on")
plt.plot(freqs1[b1], np.abs(fft1)[b1], label = "drive off")
plt.plot(freqs2[b2], np.abs(fft2)[b2], label = "drive on again")
plt.axvline(x = fc, alpha = 0.25, label = "1210.7 Hz", linestyle = '--', color = 'k')
plt.legend()
plt.show()

freqst = np.fft.rfftfreq(ns, d = 1./Fs)
#bf = np.abs(freqst-fcent)<


