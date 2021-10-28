import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from hs_digitizer import *
from scipy.optimize import curve_fit
import matplotlib
import re
import scipy.signal as ss


path = "/data/20181204/bead1/high_speed_digitizer/general_tests/spinning_50k_z_hat"
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
bw = 477.46

obj = hsDat(files[3])
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
################################################################################################
g = 50.
s = 0.004

in_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/ampramp_data_0/"
in_base_fname = "amp_ramp_50k"


amps = np.load(in_path+in_base_fname + "amps.npy")
amps*=g
amps/=s

freqs_wob = np.load(in_path+in_base_fname + "wobble_freq.npy")
#freqs/=2
freqs_wob*=2.*np.pi
def sqrt_fun(x, poi, toi):
    return np.sqrt(x*poi + toi)

def sqre_fxn(x, poi, e0):
    return x**2/poi + e0
            
popt, pcov = curve_fit(sqrt_fun, amps, freqs_wob)
popte, pcove = curve_fit(sqre_fxn, freqs_wob, amps)
plt_scale = 1000
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
f, ax = plt.subplots(4, 1, dpi = 200)
#ax.axvline(x = f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "50kHz rotation frequency")
#ax.axvline(x = 2.*f_rot, linestyle = '--', color = 'k', alpha = 0.5, label = "100kHz")

ax[0].plot((freqs-fc)*2.*np.pi, np.abs(fft))
ax[0].set_yscale("log")
ax[0].set_xlim([-bw*1.8*np.pi, bw*1.8*np.pi])
ax[0].set_ylim([5e-3, 2e1])
ax[0].set_xlabel(r"$\omega-2\omega_{0}$ [rad/s]")
ax[0].set_ylabel(r"$P_{\bot}/P_{0}$ [ppm]")
ax[0].set_yticks([1e-2, 1e-1, 1, 1e1])
#ax[0].set_title("a)", loc = "left")

ax[1].axis("off")
ax[2].plot(freqs*2.*np.pi, np.abs(phase_fft))
ax[2].set_xlim([0, 3000])
ax[3].set_xlabel(r"$\omega_{\phi}$ [rad/s]")

ax[2].set_ylabel(r"$\phi$ [rad]")
ax[2].tick_params(labelbottom = False)
ax[2].set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
ax[2].set_yticks([0, .05, 0.1])
ax[3].plot(freqs_wob, amps/plt_scale, 'x', markersize = 4)
ax[3].plot(wob_freq_plt, sqre_fxn(wob_freq_plt, *popte)/plt_scale, 'r', alpha = 0.7)
ax[3].set_xlim([0, 3000])
ax[3].set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])
ax[3].set_yticks([0, 50, 100])
ax[3].set_ylabel("E [kV/m]")
#ax[1].set_title("b)", loc = "left")

plt.subplots_adjust(top = 0.91, bottom = 0.14, left = 0.15, right = 0.92, hspace = 0.5)
plt.legend(fontsize = 12)
plt.show()
f.savefig("/home/arider/plots/20181219/spec_sidebands.png", dpi = 200)




