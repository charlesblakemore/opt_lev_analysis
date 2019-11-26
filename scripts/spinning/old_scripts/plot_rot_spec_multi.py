import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from hs_digitizer import *
from scipy.optimize import curve_fit
import matplotlib
import re
import scipy.signal as ss


path = "/data/20181030/bead1/high_speed_digitizer/procession/sudden_z_turn_on/1V_50s_turn_on_500s_int"

files= glob.glob(path + "/*.h5")

sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3])
files.sort(key = sfun)

inds = np.arange(len(files)-1)[::1]

dater = lambda ind: hsDat(files[ind])
timer = lambda ob: ob.attribs["time"]/1e9

objs = list(map(dater, inds))
times = np.array(list(map(timer, objs)))
times-=times[0]
ffter = lambda ob: np.fft.rfft(ss.detrend(ob.dat[:, 0]))
ffts = list(map(ffter, objs))
freqs = np.fft.rfftfreq(objs[0].attribs['nsamp'], 1./objs[0].attribs['fsamp'])
f_rot = 0.

#freqs/=1000

matplotlib.rcParams.update({'font.size':12})
f, ax = plt.subplots(dpi = 200)
#ax.axvline(x = f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "50kHz rotation frequency")
#ax.axvline(x = 2.*f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "100kHz")

for ii, fft in enumerate(ffts):
    ax.plot(freqs, np.abs(fft), label = str(times[ii])[:4] + 's')

#ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0, 500])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Optical Power [arb]")
plt.legend()
plt.show()




