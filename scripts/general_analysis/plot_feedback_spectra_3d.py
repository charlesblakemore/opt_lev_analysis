import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interpolate
import scipy.optimize as optimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

import bead_util as bu
import configuration as config

plt.rcParams.update({'font.size': 14})




dir1 = '/data/old_trap/20200312/beam_profiling/xprof_init/'
use_dir = False



allfiles  = [
             '/data/old_trap/20201215/bead1/10mbar_nofb_nocool.h5',
             '/data/old_trap/20201215/bead1/10mbar_nofb_nocool-low1.h5',
             '/data/old_trap/20201215/bead1/10mbar_nofb_nocool-low2.h5',
             '/data/old_trap/20201215/bead1/10mbar_nofb_nocool-low3.h5',
             '/data/old_trap/20201215/bead1/10mbar_nofb_nocool-low4.h5',
            ]



allfiles  = [
             '/data/old_trap/20201113/bead1/1_5mbar_powfb_zcool-pid.h5',
             '/data/old_trap/20201113/bead1/1_5mbar_powfb_zcool-low1.h5',
             '/data/old_trap/20201113/bead1/1_5mbar_powfb_zcool-low2.h5',
             '/data/old_trap/20201113/bead1/1_5mbar_powfb_zcool-low3.h5',
             '/data/old_trap/20201113/bead1/1_5mbar_powfb_zcool-low4.h5',
             '/data/old_trap/20201113/bead1/1_5mbar_powfb_zcool-low5.h5',
             '/data/old_trap/20201113/bead1/1_5mbar_powfb_zcool-low6.h5',
             '/data/old_trap/20201113/bead1/1_5mbar_powfb_zcool-low7.h5',
            ]

files_to_plot = [0, 1, 2, 3, 4]



new_trap = False


tfdate = '20190619'  # Bangs bead
tfdate = ''
tf_plot = False

# filename_labels = True 
filename_labels = False

#labs = ['1','2', '3']

figsize = (6,7)

ax_to_plot = 0

#################################################
#####  THESE ARGUMENTS HAVE TO BE SET REAL  #####
#####  GOOD OTHERWISE NOTHING WILL WORK     #####
#################################################
# xlim = ()
ylim = (1, 2500)
yticks = [1, 10, 100, 1000]

# zlim = ()
zlim = (1e1, 1e4)
fac_for_resfreq = 0.02

zticks = [1e0, 1e1, 1e2, 1e3, 1e4]
zticklabels = ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$']
#################################################


lpf = 2500   # Hz

userNFFT = 2**12
diag = False

fullNFFT = True

#window = mlab.window_hanning
window = mlab.window_none

###########################################################

# cmap = 'inferno'
cmap = 'plasma'
#cmap = 'jet'

posdic = {0: 'x', 1: 'y', 2: 'z'}



def plot_spectra_3d(files, ax_to_plot=0, diag=False, colormap='plasma'):
    '''Makes a cool 3d plot since waterfalls/cascaded plots end up kind 
       being fucked up.
    '''

    res_freqs = []
    powers = []

    fig = plt.figure(figsize=(7,5))
    ax = fig.gca(projection='3d')

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.6, 1]))

    # fig.suptitle('XYZ Data', fontsize=18)


    files = files

    colors = bu.get_colormap(len(files_to_plot), cmap=colormap)
    i = 0
    #colors = ['C0', 'C1', 'C2']

    print("Processing %i files..." % len(files))
    for fil_ind, fil in enumerate(files):
        
        # Display percent completion
        bu.progress_bar(fil_ind, len(files))

        # Load data
        df = bu.DataFile()
        if new_trap:
            df.load_new(fil)
        else:
            df.load(fil)

        df.calibrate_stage_position()


        if diag:
            df.diagonalize(maxfreq=lpf, date=tfdate, plot=tf_plot)


        try:
            fac = df.conv_facs[ax_to_plot]# * (1.0 / 0.12e-12)
        except:
            fac = 1.0

        if fullNFFT:
            NFFT = len(df.pos_data[ax_to_plot])
        else:
            NFFT = userNFFT
    

        if diag:
            psd, freqs = mlab.psd(df.diag_pos_data[ax_to_plot], Fs=df.fsamp, \
                                    NFFT=NFFT, window=window)
        else:
            psd, freqs = mlab.psd(df.pos_data[ax_to_plot], Fs=df.fsamp, \
                                  NFFT=NFFT, window=window)

        inds = (freqs > ylim[0]) * (freqs < ylim[1]) * (np.sqrt(psd) > zlim[0]*fac_for_resfreq)

        freqs = freqs[inds]
        psd = psd[inds]

        norm = bu.fft_norm(df.nsamp, df.fsamp)
        new_freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)

        xs = np.zeros_like(freqs) + fil_ind

        if fil_ind in files_to_plot:
            popt, pcov = bu.fit_damped_osc_amp(df.pos_data[ax_to_plot], fit_band=[10, 2000], \
                                                fsamp=df.fsamp, plot=False)
            res_freqs.append(popt[1])

            color = colors[i]
            i += 1
            ax.plot(xs, np.log10(freqs), np.log10(np.sqrt(psd)), color=color)


    zlim_actual = (zlim[0] * fac_for_resfreq, zlim[1])

    x = np.arange(len(res_freqs))
    interpfunc = interpolate.UnivariateSpline(x, res_freqs, k=2)

    ax.scatter(x, np.log10(res_freqs), zs=np.log10(zlim_actual[0]), \
                zdir='z', s=25, c=colors, alpha=1)
    ax.plot(x, np.log10(interpfunc(x)), zs=np.log10(zlim_actual[0]), \
            zdir='z', lw=2, color='k', zorder=1)


    # ax.grid()

    if ylim:
        ax.set_ylim(np.log10(ylim[0]), np.log10(ylim[1]))

    if zlim:
        ax.set_zlim(np.log10(zlim_actual[0]), np.log10(zlim_actual[1]))

    ax.set_xticks([])

    ax.set_yticks(np.log10(yticks))
    ax.set_yticklabels(yticks)

    ax.set_zticks(np.log10(zticks))
    ax.set_zticklabels(zticklabels)

    # ax.ticklabel_format(axis='z', style='sci')

    ax.set_xlabel('Closer to Focus $\\rightarrow$', labelpad=0)
    ax.set_ylabel('Frequency [Hz]', labelpad=20)
    ax.set_zlabel('ASD [Arb/$\\sqrt{\\rm Hz}$]', labelpad=15)


    # if xlim:
    #     ax.set_xlim(*xlim)

    # if ylim:
    #     ax.set_ylim(*ylim)

    # if zlim:
    #     ax.set_zlim(*zlim)


    # fig.tight_layout()

    ax.view_init(elev=15, azim=-15)

    fig.tight_layout()

    fig.subplots_adjust(top=1.35, left=-0.07, right=0.95, bottom=-0.05)
    plt.show()



if use_dir:
    allfiles, lengths = bu.find_all_fnames(dir1, sort_time=True)


plot_spectra_3d(allfiles, ax_to_plot=ax_to_plot, \
                diag=diag, colormap=cmap)