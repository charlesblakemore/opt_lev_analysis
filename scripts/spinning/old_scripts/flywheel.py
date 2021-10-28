from flywheel import *

axis = 0

fft = np.fft.rfft(df.pos_data[axis])
freqs = np.fft.rfftfreq(250000, 1./5000.)
li = miner(freqs, fmin)
ui = miner(freqs, fmax)

bt = np.logical_or(freqs>fmax, freqs<fmin)
fft[bt] = 0.


a, f = anal_signal(np.fft.irfft(fft))
t = np.linspace(0, 50, 250000)
cal = df.conv_facs[0]/k

plt.plot(t[1:s_off], f[:s_off]*cal, label = "drive on")
plt.plot(t[s_off:1], f[s_off:]*cal, label = "drive off")
plt.xlabel("time[s]")
plt.ylabel("instantaneous frequency [Hz]")
plt.legend()
plt.show()
