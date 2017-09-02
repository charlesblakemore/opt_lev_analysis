###random functions for looking at no bead data
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cant_util as cu
import bead_util as bu
import glob

path = '/data/20170822/bead6/nobead/grav_data_2'
fs = glob.glob(path + '/*X70um*.h5')
dat_column = 0


def proc_f(fname):
    '''gets info from file'''
    dat, attribs, f = bu.getdata(fname)
    psd, freqs = matplotlib.mlab.psd(dat[:, dat_column], Fs = attribs['Fsamp'], NFFT = 2**16)
    f.close()
    z = np.mean(dat[:, 19])
    x = np.mean(dat[:, 17])
    return x, z, freqs, psd



