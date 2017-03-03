import glob
import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu

dir1 = "/data/20140811/Bead8/no_bead/*RAND*.h5"
dir2 = "/data/20140811/Bead8/no_bead_pt/*RAND*.h5"

f1 = glob.glob(dir1)
f2 = glob.glob(dir2)
print f1

J2x, J2y, J2z = bu.get_avg_noise(f2,0,[0,0,0], norm_by_sum=False)
J1x, J1y, J1z = bu.get_avg_noise(f1,0,[0,0,0], norm_by_sum=False)

f = np.fft.rfftfreq((len(J1x)-1)*2, 1./5000)

dbins = bu.get_drive_bins(f)

plt.figure()
plt.loglog(f,J1x)
plt.loglog(f,J2x)
plt.loglog(f[dbins],J1x[dbins],'ro')

plt.title("x")

##### plt.figure()
##### plt.loglog(f,J1y)
##### plt.loglog(f,J2y)
##### plt.title("y")

plt.show()
