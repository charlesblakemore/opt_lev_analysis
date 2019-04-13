import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from hs_digitizer import *
from scipy.optimize import curve_fit
import matplotlib
import re

path = "/data/20181030/bead1/high_speed_digitizer/procession/sudden_tilt_2"
files= glob.glob(path + "/*.h5")

sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3])
files.sort(key = sfun)

obj = hsDat(files[-1])
fft = np.fft.rfft(obj.dat[:, 0])

freqs = np.fft.rfftfreq(obj.attribs['nsamp'], 1./obj.attribs['fsamp'])
f_rot = 50

freqs/=1000

matplotlib.rcParams.update({'font.size':12})
f, ax = plt.subplots(dpi = 200)
#ax.axvline(x = f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "50kHz rotation frequency")
ax.axvline(x = 2.*f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "100kHz")
ax.plot(freqs, np.abs(fft))

ax.set_xscale("log")
ax.set_yscale("log")
#ax.set_xlim([2.*f_rot-2, 2.*f_rot+2])
plt.xlabel("Frequency [kHz]")
plt.ylabel("Optical Power [arb]")
plt.legend()
plt.show()




