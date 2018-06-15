import matplotlib.pyplot as plt
import numpy as np
import bead_util as bu
import bead_util_funcs as buf
import os
import matplotlib.mlab
from sklearn.decomposition import FastICA
import scipy.signal as ss



path = "/data/20180613/bead1/dipole_vs_height/10V_70um_17hz"

fs = buf.find_all_fnames(path)

df0 = bu.DataFile()
df1 = bu.DataFile()

df0.load(fs[0])
df1.load(fs[-1])

def do_ICA(df, lf = 0.005, hf = 0.3, filt = False):
    if filt:
        b, a = ss.butter(3, [lf, hf], btype = 'bandpass')
        filt_amps = ss.filtfilt(b, a, df.amp)
        filt_phase = ss.filtfilt(b, a, df.phase)
    else:
    
        filt_amps = np.transpose(np.transpose(df.amp) - df.amp.mean(axis = 1))
        filt_phase = np.transpose(np.transpose(df.phase) - \
                        df.phase.mean(axis = 1))
    
    filt_amps = np.transpose(np.transpose(filt_amps)/filt_amps.std(axis = 1))
    filt_phase = np.transpose(np.transpose(filt_phase)/filt_phase.std(axis = 1))

    dmat = np.vstack([filt_amps, filt_phase])
    ica = FastICA(n_components = 10)
    s_ = ica.fit_transform(np.transpose(dmat))
    
    
    for i in range(s_.shape[1]):
        plt.plot(s_[:, i], label = 'ic '+ str(i))
    plt.legend()
    plt.show()
    
    
    for i in range(s_.shape[1]):
        psd, freqs = matplotlib.mlab.psd(s_[:, i], Fs = df.fsamp, NFFT =s_.shape[0])
        plt.loglog(freqs, psd, label = 'ic '+ str(i))
    plt.legend()
    plt.show()
    return s_
