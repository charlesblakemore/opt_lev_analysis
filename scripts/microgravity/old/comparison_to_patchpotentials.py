import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

import bead_util as bu
import cant_util as cu
import grav_util as gu

import scipy.signal as signal
import scipy.optimize as optimize 
import scipy.stats as stats

import sys


patch_asds = pickle.load( open('/home/charles/sshtemp/out_asds.p', 'rb') )

patchfreq = patch_asds['freqs']
patchasd = patch_asds[12.5]



### Load backgrounds

background_data = cu.Force_v_pos()
background_data.load('/force_v_pos/20170822_grav_background_sep10um_h15um.p')

bins = background_data.bins
force = background_data.force
errs = background_data.errs

diagbins = background_data.diagbins
diagforce = background_data.diagforce
diagerrs = background_data.diagerrs

wvnum = np.fft.rfftfreq(len(force), d=(bins[1]-bins[0]))
datfft = np.fft.rfft(force)
datasd = np.abs(datfft)

diagwvnum = np.fft.rfftfreq(len(diagforce), d=(diagbins[1]-diagbins[0]))
diagdatfft = np.fft.rfft(diagforce)
diagdatasd = np.abs(diagdatfft)






plt.loglog(diagwvnum, diagdatasd, color='b', label='Actual Data')
plt.loglog(patchfreq, patchasd, color='r', label='COMSOL Patch Potential Simulation')
plt.xlabel('Wavenumber [um^-1]', fontsize=16)
plt.ylabel('ASD [N / rt(um^-1)]', fontsize=16)
plt.legend(numpoints=1, fontsize=10)
plt.title('COMSOL Calculation vs. Data', fontsize=18)




plt.show()
