import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from hs_digitizer import *
from scipy.optimize import curve_fit
import matplotlib
import re
import scipy.signal as ss


path = "/data/20181030/bead1/high_speed_digitizer/procession/turn_over_1kHz_500s"
files= glob.glob(path + "/*.h5")

sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3])
files.sort(key = sfun)

b, a = ss.butter(3, 0.1)

obj = hsDat(files[1])
fft = np.fft.rfft(obj.dat[:, 0])

freqs = np.fft.rfftfreq(obj.attribs['nsamp'], 1./obj.attribs['fsamp'])
tarr = np.linspace(0, 500, 500*1000)
f0 = 2.354
n_harms = 6
bw = 1.
f_bool = np.zeros(len(fft), dtype = 'bool')
for i in range(n_harms):
    bi = np.abs(freqs-(i+1)*f0)<bw
    f_bool[bi] = True

f_bool = np.logical_not(f_bool)
#fft[f_bool] = 0.
filtered = np.fft.irfft(fft)
#ss.filtfilt(b, a, obj.dat[:, 0])

matplotlib.rcParams.update({'font.size':12})
f, ax = plt.subplots(dpi = 200)
#ax.axvline(x = f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "50kHz rotation frequency")
ax.axvline(x = 2.36, linestyle = '--', color = 'k', alpha = 0.5, label = r"$\Omega = 2 \pi \times 2.36$Hz")
ax.plot(freqs, np.abs(fft))


ax.set_xscale("log")
ax.set_yscale("log")
#ax.set_xlim([0, 40.])
#plt.xlabel("Time[s]")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Optical Power [arb]")
plt.legend()
plt.show()




