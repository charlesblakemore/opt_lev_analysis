import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.signal as ss

path = "/data/20180927/bead1/spinning/0_6mbar"
files = bu.find_all_fnames(path)
index = 0
fdrive = 1210.7
bw = 0.1
bwp = 5.
Ns = 250000
Fs = 5000.
k = 1e-13*(2.*np.pi*370.)**2

df = bu.DataFile()
def proc_f(f):
    df.load(f)
    df.load_other_data()
    df.diagonalize()

    drive = df.other_data[2]
    resp = ss.detrend(df.pos_data[index])*df.conv_facs[0]/k
    drive = ss.detrend(df.other_data[2])*df.conv_facs[0]/k
    respft = np.fft.rfft(resp)
    driveft = np.fft.rfft(drive)
    freqs = np.fft.rfftfreq(Ns, d = 1./Fs)

    tarr = np.linspace(0., 50., 250000)
    respft_line = respft
    driveft_line = driveft
    respft_line[np.abs(freqs - fdrive)>bw] = 0.
    driveft_line[np.abs(freqs - fdrive)>bw] = 0.
    anal_signal_resp = ss.hilbert(np.fft.irfft(respft_line))
    anal_signal_drive = ss.hilbert(np.fft.irfft(driveft_line))
    phirfft = np.angle(np.sum(respft_line)/np.sum(driveft_line))
    phirh = np.unwrap(np.angle(anal_signal_resp)) - np.unwrap(np.angle(anal_signal_drive))
    return freqs, respft, driveft, tarr, anal_signal_resp, phirh, phirfft

freqs0, respft0, driveft0, tarr0, anal_signal_resp0, phir0h, phi0fft = proc_f(files[-1])
freqs, respft1, driveft1, tarr, anal_signal_resp1, phir1h, phi1fft = proc_f(files[-2])
freqs, respft2, driveft2, tarr, anal_signal_resp2, phir2h, phi2fft = proc_f(files[-3])
#plot the data

plt.plot(tarr, np.abs(anal_signal_resp2), label = "0-50s")
plt.plot(tarr, np.abs(anal_signal_resp1), label = "350-400s")
plt.plot(tarr, np.abs(anal_signal_resp0), label = "700-750s")
plt.xlabel("Time [s]")
plt.legend()
plt.ylabel("Instantaneous Amplitude [m]")
plt.ylim([0, 4e-10])
plt.xlim([0, 50])
plt.show()



plt.plot(tarr, phir0h, label = "0-50s")
plt.plot(tarr, phir1h-2.*np.pi, label = "350-400s")
plt.plot(tarr, phir2h, label = "700-750s")
plt.xlabel("Time [s]")
plt.ylabel("Drive Response Phase Difference [rad]")
plt.xlim([0, 50])
#plt.ylim([0, 3])
plt.show()

