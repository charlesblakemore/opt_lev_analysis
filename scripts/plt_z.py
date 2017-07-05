import matplotlib.pyplot as plt
import bead_util as bu
import numpy as np
import os
import matplotlib


bpv = 2.0**15/10.
pipb = 1./1000.
muppi = 1.0639/4. 
mupv = muppi*pipb*bpv
path = "/data/20170606/bead2_2"

fname0 = "2mbar_nocool1kbppi.h5"
fname1 = "2mbar_zcool1kbppi.h5"

dat0, attribs0, f0 = bu.getdata(os.path.join(path, fname0))
Fs = attribs0["Fsamp"]
f0.close()

dat1, attribs1, f1 = bu.getdata(os.path.join(path, fname1))
Fs = attribs1["Fsamp"]
f1.close()

def plt_zpos(dat, Fs, mupv, n_plt):
    n = len(dat[:, 0])
    t = np.arange(0, n/Fs, 1./Fs)
    plt.plot(t[:n_plt], dat0[:n_plt, 2]*mupv)
    plt.xlabel("time[s]")
    plt.ylabel("displacement[um]")
    plt.show()

def plt_zpsd(dat, Fs, mupv, nfft = 2**12):
    psd, freqs = matplotlib.mlab.psd(dat[:, 2]*mupv, NFFT = nfft, Fs = Fs, detrend = matplotlib.mlab.detrend_linear)
    plt.loglog(freqs, np.sqrt(psd))
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("$\mu m / \sqrt{Hz}$")
    #plt.show()

def vz_rms(dat, Fs, mupv, nfft = 2**12):
    psd, freqs = matplotlib.mlab.psd(dat[:, 2]*mupv, NFFT = nfft, Fs = Fs, detrend = matplotlib.mlab.detrend_linear)
    ws = 2.*np.pi*freqs
    vpsd = psd*ws**2
    df = freqs[1]-freqs[0]
    return np.sqrt(np.sum(vpsd[1:])*df)

#bu.get_calibration(os.path.join(path, fname0), [1, 300], make_plot=True, 
                    #data_columns = [2,1], drive_column=-1, NFFT=2**12, exclude_peaks=False, spars=[])
