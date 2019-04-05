import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.signal as ss

path = "/data/20180927/bead1/spinning/50s_monitor_5min_gaps"
files = bu.find_all_fnames(path)
index = 0
fdrive = 1210.7
bw = 0.5
bwp = 5.
Ns = 250000
Fs = 5000.
k = 1e-13*(2.*np.pi*370.)**2

df = bu.DataFile()
df.load(files[-2])
df.load_other_data()
df.diagonalize()

drive = df.other_data[2]
resp = ss.detrend(df.pos_data[index])*df.conv_facs[0]/k
drive = ss.detrend(df.other_data[2])*df.conv_facs[0]/k
respft = np.fft.rfft(resp)
driveft = np.fft.rfft(drive)
freqs = np.fft.rfftfreq(Ns, d = 1./Fs)

#plot the data
plt.loglog(freqs, np.abs(respft)*2./Ns)
plt.axvline(x = fdrive, linestyle = '--', color = 'k', label = str(fdrive)+"Hz drive", alpha = 0.5)
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Apparent Displacement [m]")
plt.show()

#plot the zoom

plt.semilogy(freqs, np.abs(respft)*2./Ns)
plt.axvline(x = fdrive, linestyle = '--', color = 'k', label = str(fdrive)+"Hz drive", alpha = 0.5)
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Apparent Displacement [m]")
plt.xlim([fdrive-bwp/2., fdrive+bwp/2.])
plt.show()

#get inst amp and phase

tarr = np.linspace(0., 50., 250000)
respft_line = respft
driveft_line = driveft
respft_line[np.abs(freqs - fdrive)>bw] = 0.
driveft_line[np.abs(freqs - fdrive)>bw] = 0.
anal_signal_resp = ss.hilbert(np.fft.irfft(respft_line))
anal_signal_drive = ss.hilbert(np.fft.irfft(driveft_line))

phir = np.unwrap(np.angle(anal_signal_resp)) - np.unwrap(np.angle(anal_signal_drive))

plt.plot(tarr, np.abs(anal_signal_resp))
plt.xlabel("Time [s]")
plt.ylabel("Instantaneous Amplitude [m]")
plt.ylim([0, 4e-10])
plt.xlim([0, 50])
plt.show()



plt.plot(tarr, np.abs(phir))
plt.xlabel("Time [s]")
plt.ylabel("Drive Response Phase Difference [rad]")
plt.xlim([0, 50])
#plt.ylim([0, 3])
plt.show()

