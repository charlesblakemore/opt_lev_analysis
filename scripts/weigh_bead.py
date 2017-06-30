import matplotlib.pyplot as plt
import numpy as np
import bead_util as bu

path_g = "/data/20170623/bead9/powermod/turbobase_xyzcool_0_1pmod_41Hz.h5"
path_e = "/data/20170623/bead9/powermod/turbobase_xyzcool_elec1_5000mV41Hz0mVdc_0.h5"

g = 9.8#m/s^2

data_column = 2
dfreq = 41.
p_mod = 0.001
bead_charge = 4#es
v_amp = 5.0#volts
elec_spacing = 0.004 #m

E_amp = v_amp/elec_spacing
q_bead = bead_charge*1.062E-19
f_amp = E_amp*q_bead 

dat_g, attribs_g, f_g = bu.getdata(path_g)
Fs = attribs_g["Fsamp"]
f_g.close()
dat_e, attribs_e, f_e = bu.getdata(path_e)
f_e.close()

fft_g = np.fft.rfft(dat_g[:, data_column])
fft_e = np.fft.rfft(dat_e[:, data_column])
freqs = np.fft.rfftfreq(len(dat_g[:, data_column]), 1./Fs)

dfreqind = np.argmin(np.abs(freqs-dfreq))
gerat = np.abs(fft_g[dfreqind]/fft_e[dfreqind])

m = 10.**12*gerat*f_amp/(g*p_mod)#ng
plt.loglog(freqs, np.abs(fft_g), label = "0.1%mg")
plt.loglog(freqs, np.abs(fft_e), label = "3e*5V")
plt.loglog(freqs[dfreqind], np.abs(fft_g[dfreqind]), 'og')
plt.loglog(freqs[dfreqind], np.abs(fft_e[dfreqind]), 'ob')
plt.xlabel("Frequency[Hz]")
plt.ylabel("FFT amplitude [V]")
plt.legend()
plt.show()
