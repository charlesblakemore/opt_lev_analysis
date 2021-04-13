import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize as opti

import bead_util as bu
import transfer_func_util as tf

plt.rcParams.update({'font.size': 16})

dirname = '/data/old_trap/20201222/gbead1/res_freq_vs_height/coarse_2'

files, _ = bu.find_all_fnames(dirname, ext='.h5', sort_time=True)
nfiles = len(files)

userNFFT = 2**12

fullNFFT = False

window = mlab.window_none

plot_raw_data = False
fit_debug = False

freq_lims = (20, 1000)


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

        # try:
        popt, pcov = opti.curve_fit(bu.damped_osc_amp, freqs[inds], asd[inds], p0=p0, \
                                    maxfev=10000)
        # except:
        #     popt = p0

        if fit_debug:
            plt.loglog(freqs, asd)
            plt.loglog(freqs, bu.damped_osc_amp(freqs, *p0), lw=2, ls='--', \
                        color='k', label='init guess')
            plt.loglog(freqs, bu.damped_osc_amp(freqs, *popt), lw=2, ls='--', \
                        color='r', label='fit result')
            plt.legend(fontsize=10)
            plt.show()
            input()

        fit_freqs[filind, i] = np.abs(popt[1])

fig, ax = plt.subplots(1,1,figsize=(8,6))

ax.plot(zavg, fit_freqs[:,0], label='X freqs')
ax.plot(zavg, fit_freqs[:,1], label='Y freqs')

ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Mean Z-posotion')
ax.legend(fontsize=12)

fig.tight_layout()
plt.show()
