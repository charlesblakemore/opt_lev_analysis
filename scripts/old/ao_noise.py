import glob, re, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.optimize as sp
import scipy.signal as sig
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib import dates

dat, attribs, cf = bu.getdata( "/data/20150618/aotest/urmbar_xyzcool_50mV_41Hz_450mVDC.h5"  )

xr = dat[:,0]
xr -= np.mean(xr)
fr, freqs = matplotlib.mlab.psd(xr, Fs = attribs['Fsamp'], NFFT = 2**17) 

sfac = 3e-15 ## N/V

plt.figure()
plt.loglog(freqs, fr*sfac)
plt.xlabel( "Freq. (Hz)")
#plt.ylabel( "AO voltage noise (V Hz$^{-1/2}$)" )
plt.ylabel( "PXI-6723 Analog Out equivalent force noise (N Hz$^{-1/2}$)" )
plt.xlim([0.1, 5e3])
plt.savefig("ao_noise.pdf")

plt.show()
