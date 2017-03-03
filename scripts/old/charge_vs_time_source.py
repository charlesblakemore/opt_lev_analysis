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

path = "/data/20150702/Bead1/chargelp_50V"

def sort_fun( s ):
    idx = re.findall( "\d+.h5", s)
    return int( idx[0][:-3] )

flist = sorted(glob.glob(os.path.join(path, "*.h5")), key=sort_fun)

for f in flist:

    print f
    dat, attribs, cf = bu.getdata( f )

    fnum = sort_fun(f)
    if(fnum < 334): continue

    x = dat[:,bu.data_columns[0]]
    x -= np.mean(x)
    y = dat[:,bu.data_columns[1]]
    y -= np.mean(y)
    Fs = attribs['Fsamp']
    b,a = sig.butter(3, np.array([49, 53])/(Fs/2.0), btype="bandpass")

    xf = sig.filtfilt(b,a,x)

    t = np.linspace( 0, (len(x)-1)/Fs, len(x) )

    plt.figure()
    plt.plot(t, x)
    plt.plot(t, y)
    plt.plot(t, xf, 'r')
    plt.show()
