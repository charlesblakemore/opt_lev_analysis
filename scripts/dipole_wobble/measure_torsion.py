import matplotlib.pyplot as plt
import numpy as np
import bead_util as bu
import cant_utils as cu
import matplotlib
import scipy.signal as sig
from scipy.optimize import curve_fit as cf

path0 = "/data/20160418/bead2/cantdc_sweep_noosc"

dirobj = cu.Data_dir(path0, [0, 0, 0])
cbind = 24
cfind = 16


dirobj.load_dir(cu.sb_loader)
#fobj = dirobj.fobjs[0]

#ts = fobj.pos_data[1]
#fd = fobj.electrode_settings[16]
#Fs = fobj.Fsamp

#bw = 10.
#cfs = np.array([fd-bw, fd+bw])*2./(1.*Fs) 


sfun = lambda obj: obj.electrode_settings[24]

dirobj.fobjs = sorted(dirobj.fobjs, key = sfun)

for fobj in dirobj.fobjs[::2]:
    fobj.plt_psd()

plt.xlim([0.5, 100])
plt.legend()
plt.show()


