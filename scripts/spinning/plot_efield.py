from hs_digitizer import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import window_func as window
import bead_util as bu
from amp_ramp_3 import bp_filt, lp_filt, hp_filt
from scipy import signal

filename = '/data/old_trap/20191223/bead1/spinning/tests/test_amp_mod/turbombar_powfb_xyzcool_smaller_amp_0.h5'
filename = 'turbombar_powfb_xyzcool_smaller_amp_0.h5
#filename = '/data/old_trap/20190626/bead1/spinning/wobble/long_wobble/wobble_0000/turbombar_powfb_xyzcool_9.h5'

fc = 45e3
fc1 = 10e3
obj = hsDat(filename)

Ns = obj.attribs['nsamp']
Fs = obj.attribs['fsamp']

freqs = np.fft.rfftfreq(Ns, 1./Fs)

window = np.hanning(Ns)

drive_sig = obj.dat[:,1]
#drive_sig *= window
#drive_sig_filt = lp_filt(drive_sig, fc, Ns, Fs)
#drive_sig_filt = hp_filt(drive_sig, fc1, Ns, Fs)

drive_sig_filt = drive_sig#bp_filt(drive_sig, 25e3, Ns, Fs, 2e3)

drive_fft = np.fft.rfft(drive_sig)
drive_fft_filt = np.fft.rfft(drive_sig_filt)

plt.plot(drive_sig)
plt.plot(drive_sig_filt)
plt.show()

plt.loglog(freqs, np.abs(drive_fft))
plt.loglog(freqs, np.abs(drive_fft_filt))
plt.show()

