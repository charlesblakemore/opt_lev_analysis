import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from hs_digitizer import *
from scipy.optimize import curve_fit
import matplotlib
import re
import scipy.signal as ss


#path = "/data/20181204/bead1/high_speed_digitizer/general_tests/spinning_50k_z_hat"
path = "/daq2/20190408/bead1/high_speed_test/t17_no-laser_no-mon"

path = "/daq2/20190514/bead1/spinning/test2/"

fc = 100000
files= glob.glob(path + "/*.h5")

data_ax = 0

print files

sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3])
files.sort(key = sfun)

t = 1.0 # 0.7 
vpa = 1e5
apw = 0.3
g = 1000
p0 = 0.001
wpv = 1e6/(t*vpa*apw*g*p0) #1e6 for ppm 
bw = 477.46


obj = hsDat(files[0])
freqs = np.fft.rfftfreq(obj.attribs["nsamp"], d = 1./obj.attribs["fsamp"])
fft = np.fft.rfft(obj.dat[:, data_ax])
fft/=len(fft)
fft *= wpv

plt.loglog(freqs, np.abs(fft))
plt.show()

fft_sig = np.zeros_like(fft)
bf = np.abs(freqs-fc) < bw

fft_sig[bf] = fft[bf]
a_sig = ss.hilbert(np.fft.irfft(fft_sig))
phase = ss.detrend(np.unwrap(np.angle(a_sig)))
phase_fft = np.fft.rfft(phase)
phase_fft/=len(phase_fft)
################################################################################################
g = 50.
s = 0.004

matplotlib.rcParams.update({'font.size':14})
#f, ax = plt.subplots(dpi = 200, sharex = True)
#ax.axvline(x = f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "50kHz rotation frequency")
#ax.plot(amps/plt_scale, freqs_wob, '.', markersize = 2)
#ax.plot(amps/plt_scale, sqrt_fun(amps, *popt), 'r',alpha = 0.5,  label = r"$\sqrt{\frac{dE}{I}}$", linestyle = ':', linewidth = 5)
#ax.plot(amps/plt_scale, freqs, '.', markersize = 2)
#ax.legend()
#ax.set_ylabel(r"$\omega_{\phi}$ [rad/s]")
#ax.set_xlim([2.*f_rot-2, 2.*f_rot+2])
#ax.set_xlabel("E [kV/m]")
#plt.ylabel("Sideband Frequency [Hz]")
#ax.legend()
#plt.subplots_adjust(top = 0.96, bottom = 0.15, left = 0.15, right = 0.99)


#plt.show()







wob_freq_plt = np.linspace(0, 2785, 10000)


#################################################################################################
matplotlib.rcParams.update({'font.size':14})
f, ax = plt.subplots(dpi = 200)
#ax.axvline(x = f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "50kHz rotation frequency")
#ax.axvline(x = 2.*f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "100kHz")

ax.plot((freqs-fc)*2.*np.pi, np.abs(fft))
ax.set_yscale("log")
ax.set_xlim([-bw*1.8*np.pi, bw*1.8*np.pi])
#ax.set_ylim([5e-3, 2e1])
ax.set_xlabel(r"$\omega-2\omega_{0}$ [rad/s]")
ax.set_ylabel(r"$P_{\bot}/P_{0}$ [ppm]")
#ax.set_yticks([1e-2, 1e-1, 1, 1e1])
#ax[0].set_title("a)", loc = "left")


plt.subplots_adjust(top = 0.91, bottom = 0.14, left = 0.15, right = 0.92, hspace = 0.5)
plt.legend(fontsize = 12)
plt.show()
#f.savefig("/home/arider/plots/20181219/just_spinning_spec.png", dpi = 200)




