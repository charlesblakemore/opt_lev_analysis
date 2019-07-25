import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from hs_digitizer import *
from scipy.optimize import curve_fit
import matplotlib
import re
import scipy.signal as ss


#path = "/data/20181204/bead1/high_speed_digitizer/general_tests/spinning_50k_z_hat"

#path = "/daq2/20190430/bead1/spinning/he/1vpp_50kHz_2"
#path = "/daq2/20190507/bead1/spinning/test/50kHz_4vpp"
path = "/daq2/20190514/bead1/spinning/wobble/manual_data"
fc = 1e5
files= glob.glob(path + "/*.h5")

sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3])
files.sort(key = sfun)

t = 0.7 
vpa = 1e5
apw = 0.1
g =500
p0 = 0.001
wpv = 1e6/(t*vpa*apw*g*p0) #1e6 for ppm 
bw = 400

obj = hsDat(files[0])
freqs = np.fft.rfftfreq(obj.attribs["nsamp"], d = 1./obj.attribs["fsamp"])
fft = np.fft.rfft(obj.dat[:, 0])
fft/=len(fft)
fft *= wpv
fft_sig = np.zeros_like(fft)
bf = np.abs(freqs-fc) < bw

fft_sig[bf] = fft[bf]
a_sig = ss.hilbert(np.fft.irfft(fft_sig))
phase = ss.detrend(np.unwrap(np.angle(a_sig)))
phase_fft = np.fft.rfft(phase)
phase_fft/=len(phase_fft)
matplotlib.rcParams.update({'font.size':14})
f, ax = plt.subplots(2, 1, dpi = 200)
#ax.axvline(x = f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "50kHz rotation frequency")
#ax.axvline(x = 2.*f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "100kHz")

ax[0].plot((freqs-fc)*2.*np.pi, np.abs(fft))
ax[0].set_yscale("log")
ax[0].set_xlim([-bw*2.*np.pi, bw*2.*np.pi])
ax[0].set_ylim([5e-3, 2e1])
ax[0].set_xlabel(r"$\omega-2\omega_{0}$[rad/s]")
ax[0].set_ylabel(r"$P_{\bot}/P_{0}$ [ppm]")
#ax[0].set_title("a)", loc = "left")
ax[1].plot(freqs*2.*np.pi, np.abs(phase_fft))
ax[1].set_xlim([0, bw*2.*np.pi])
ax[1].set_xlabel(r"$\omega_{\phi}$ [rad/s]")

ax[1].set_ylabel(r"$\phi$ [rad]")
ax[0].set_yticks([1e-2, 1e-1, 1, 1e1])
#ax[1].set_title("b)", loc = "left")
f.subplots_adjust(hspace = 0.5)

plt.subplots_adjust(top = 0.91, bottom = 0.14, left = 0.15, right = 0.92, hspace = 0.6)
plt.legend(fontsize = 12)
plt.show()
#f.savefig("/home/arider/plots/20181219/spinning_spec_sidebands.png", dpi = 200)




