from hs_digitizer import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import window_func as window
import bead_util as bu
from amp_ramp_3 import bp_filt, lp_filt, hp_filt
from scipy import signal
from transfer_func_util import damped_osc_amp
from scipy.optimize import curve_fit
from memory_profiler import profile
from memory_profiler import memory_usage

import gc

mpl.rcParams['figure.figsize'] = [7,5]
mpl.rcParams['figure.dpi'] = 150

filename = '/data/old_trap/20191223/bead1/spinning/deriv_feedback_4/tests/turbombar_powfb_xyzcool_4_downsamp_phi_dc_offset_transient_0.h5'
filename = '/data/old_trap/20200130/bead1/spinning/transient/change_dc_offset/turbombar_powfb_xyzcool_off_to_on_0.h5'


def exp(t, tc, A, B, C):

    return A*np.exp(B*(t-tc)) + C  

obj = hsDat(filename)

print(obj.attribs['start_pm_dg'], obj.attribs['pressures'])
raw_input()

Ns = obj.attribs['nsamp']
Fs = obj.attribs['fsamp']
tarr = np.arange(Ns)/Fs

print(Ns, Fs)
lib_freq = 393
bw = 50

freqs = np.fft.rfftfreq(Ns, 1./Fs)

sig_rot = obj.dat[:,0]

z_rot = signal.hilbert(sig_rot)

phase_rot = signal.detrend(np.unwrap(np.angle(z_rot)))

phase_rot_filt = bp_filt(phase_rot, lib_freq, Ns, Fs, bw)

z_phase_rot = signal.hilbert(phase_rot_filt)

amp_zpr = np.abs(z_phase_rot)

fft_phase = np.fft.rfft(phase_rot)


plt.loglog(freqs, np.abs(fft_phase))
plt.show()

plt.plot(tarr, phase_rot, label=r'un-filtered $\phi$')
plt.plot(tarr, phase_rot_filt, label=r'filtered $\phi$')
plt.plot(tarr, amp_zpr, label=r'Inst. amplitude of $\phi$')
plt.xlabel('Time [s]')
plt.ylabel('Signal Amplitude')
plt.legend()


plt.show()
raw_input()
#
#t_cut = (tarr > 5.7) & (tarr < 6.7)
#
#A_guess = np.amax(amp_zpr[t_cut])
#B_guess = -1
#C_guess = 0
#
#p0 = [5.7, A_guess, B_guess, C_guess]
#popt, pcov = curve_fit(exp, tarr[t_cut], amp_zpr[t_cut], p0=p0)
#
#
#print(popt)
#
#x = np.linspace(tarr[t_cut][0], tarr[t_cut][-1], len(tarr[t_cut])*10)
#
##plt.plot(x, exp(x, *p0))
##plt.show()
#
#fit_label = r'$Ae^{B(t-tc)}$ + C' + ', tc={}s, A={}, B={}Hz, C={}'.format(popt[0].round(2),popt[1].round(2), popt[2].round(2), popt[3].round(2))
#
#plt.plot(tarr[t_cut], phase_rot_filt[t_cut], label=r'filtered $\phi$')
#plt.plot(tarr[t_cut], amp_zpr[t_cut], label='Inst. amplitude of $\phi$')
#plt.plot(x, exp(x, *popt), label=fit_label)
#plt.ylabel('Amplitude')
#plt.xlabel('Time [s]')
#plt.legend()
#plt.show()




