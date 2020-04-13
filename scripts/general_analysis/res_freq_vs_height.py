import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize as opti

import bead_util as bu
import transfer_func_util as tf

plt.rcParams.update({'font.size': 16})

dirname = '/data/old_trap/20200330/gbead3/res_freq_vs_height/coarse_3'

files, _ = bu.find_all_fnames(dirname, ext='.h5', sort_time=True)
nfiles = len(files)

userNFFT = 2**12

fullNFFT = False

window = mlab.window_none

plot_raw_data = False
fit_debug = False

freq_lims = (0, 1000)


#######################################

fit_freqs = np.zeros((nfiles, 2))
zavg = np.zeros(nfiles)

for filind, filename in enumerate(files):
    bu.progress_bar(filind, nfiles)

    df = bu.DataFile()
    df.load(filename)

    zavg[filind] = np.mean(df.pos_data[2]) 

    if fullNFFT:
        NFFT = len(df.nsamp)
    else:
        NFFT = userNFFT

    for i in [0,1]:
        psd, freqs = mlab.psd(df.pos_data[i], Fs=df.fsamp, NFFT=NFFT, window=window)
        asd = np.sqrt(psd)

        inds = (freqs > freq_lims[0]) * (freqs < freq_lims[1])

        if plot_raw_data:
            plt.loglog(freqs, asd)
            plt.show()

        p0 = [np.std(df.pos_data[i])*df.nsamp, 300, 100]

        try:
            popt, pcov = opti.curve_fit(tf.damped_osc_amp, freqs[inds], asd[inds], p0=p0, \
                                        maxfev=10000)
        except:
            popt = p0

        if fit_debug:
            plt.loglog(freqs, asd)
            plt.loglog(freqs, tf.damped_osc_amp(freqs, *p0), lw=2, ls='--', \
                        color='k', label='init guess')
            plt.loglog(freqs, tf.damped_osc_amp(freqs, *popt), lw=2, ls='--', \
                        color='r', label='fit result')
            plt.legend(fontsize=10)
            plt.show()

        fit_freqs[filind, i] = popt[1]

fig, axarr = plt.subplots(2,1,sharex=True)

axarr[0].plot(fit_freqs[:,0], label='X freqs')
axarr[0].plot(fit_freqs[:,1], label='Y freqs')
axarr[1].plot(zavg)

axarr[0].set_ylabel('Frequency [Hz]')
axarr[1].set_ylabel('Mean Z-posotion')
axarr[1].set_xlabel('File index')

fig.tight_layout()
plt.show()
