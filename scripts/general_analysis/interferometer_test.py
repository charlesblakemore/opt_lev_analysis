import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import glob
import os
from scipy import signal
import scipy
import matplotlib

data_path = '/data/20170531/int_calib'
files = glob.glob(os.path.join(data_path + '/*.h5'))
bppi = 100.
adcbits = 2**15
bpv = adcbits/10.
mupv = 80./10.
mupc = 0.532

data0, attribs, f = bu.getdata(files[0])
f.close()

z = mupc*(data0[:, 2] - np.mean(data0[:, 2]))*(bpv/bppi)*0.5
cant_z = (data0[:, 17] - np.mean(data0[:, 17]))*mupv

b, a = signal.butter(4, 0.001, btype = 'high')
#z = signal.filtfilt(b, a, z)

def scale(s):
    #finds sum square error difference after scaling data
    return np.sum((s*z-cant_z)**2)

res = scipy.optimize.minimize_scalar(scale) 
t = np.linspace(0., 10., 50000)
res.x = 1.
sz = z*res.x
error = cant_z-z*res.x

plt.plot(t, cant_z, label = "cantilever position")
plt.plot(t, sz, label = 'interferometer')
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("displacement[um]")
plt.show()

plt.plot(t, error, label = 'interferometer error')
plt.xlabel("time [s]")
plt.ylabel("displacement[um]")
plt.legend()

plt.show()

psd, freqs = matplotlib.mlab.psd(error, NFFT = 2**12, Fs = 5000.)
plt.loglog(freqs, np.sqrt(psd))
plt.xlabel("frequency[Hz]")
plt.ylabel("um/sqrt[Hz]")
plt.show()
