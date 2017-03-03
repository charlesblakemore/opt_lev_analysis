import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import os


fname = r"allon_drive_pxi6259_chan0_10000mV_41Hz.h5"
path = "/data/20140620/xtalk"

channels = np.array([1, 2, 3, 4, 5, 6])
drive_column = -1
dat, attribs, f = bu.getdata(os.path.join(path, fname))

corrs = []
corrmaxs = []

for i in range(7):
    if i != drive_column:
        
        corrmax = np.max(bu.corr_func(dat[:, drive_column], dat[:, i], attribs['Fsamp'], attribs['drive_freq'], filt = True))
 
        sub_corrs = bu.corr_blocks(dat[:, drive_column], dat[:, i], attribs['Fsamp'], attribs['drive_freq'], filt = True, N_blocks = 3)

        corrs.append(sub_corrs)
        corrmaxs.append(corrmax)

        


plt.figure()
corrs = np.array(corrs)
plt.errorbar(range(len(corrs[:, 0])), corrs[:, 0], yerr = corrs[:, 1], fmt = 'o')
plt.errorbar(range(len(corrs[:, 0])), np.array(corrmaxs), yerr = corrs[:, 1], fmt = 'o' )

plt.show()

