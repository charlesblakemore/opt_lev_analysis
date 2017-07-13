import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import glob
import os
import scipy
import matplotlib

data_path = '/data/20170602/direct_digitize'
files = glob.glob(os.path.join(data_path + '/*.h5'))

data0, attribs, f = bu.getdata(files[0])
f.close()

back_reflect = data0[:, 0] - np.mean(data0[:, 0])
pref_c = data0[:, 1] - np.mean(data0[:, 1])
pref_s = np.fft.irfft(np.exp(0.5j)*np.fft.rfft(pref_c))

b, a = scipy.signal.butter(4, 0.1)
c = scipy.signal.filtfilt(b, a, pref_c*back_reflect)
s = scipy.signal.filtfilt(b, a, pref_s*back_reflect)

phi = np.arctan2(c, s)

ax = plt.gca()
ax.plot(phi)
#ax.set_ylim([-1, 2])

plt.show()
