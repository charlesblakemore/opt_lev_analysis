import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti
import scipy.signal as signal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

plt.rcParams.update({'font.size': 16})

dirname  = '/data/old_trap/20230410/bead1/spinning/change_direction_xy_to_xz'

movie_frames_path = '/home/cblakemore/plots/20230410/spinning/xy_to_xz'

show_first_frame = False

# data_axes = [0,1]
# axes_labels = ['Cross-polarized light', 'Tabor Monitor']
data_axes = [0]
axes_labels = ['Cross-polarized light']

use_filename_labels = False
labels = []

#file_inds = (-10,-1)
file_inds = (5, 55)
file_step = 1

userNFFT = 2**20
fullNFFT = True

# plot_freqs = (55000.0, 65000.0)
plot_freqs = (20000, 90000)

invert_zorder = True
invert_colors = False

#window = mlab.window_hanning
window = mlab.window_none

# ylim = (1e-21, 1e-14)
ylim = (5e-4, 1.1e-1)
# ylim = ()

xlim = (20000.0, 90000.0)
# xlim = ()

xticks = [25000, 50000, 75000]
# xticks = None


annotate_frame = True
annotate_indices = [4, 5]
annotate_string = 'Change spin axis: XY -> XZ'
annotate_size = 16


###########################################################

bu.make_all_pardirs(os.path.join(movie_frames_path, 'test.py'))

cmap = 'plasma'





allfiles, lengths = bu.find_all_fnames(dirname, sort_time=True)
files = allfiles[file_inds[0]:file_inds[1]:file_step]

colors = bu.get_colormap(len(files), cmap=cmap)
if invert_colors:
    colors = colors[::-1]

zorders = np.arange(len(files)) + 5
if invert_zorder:
    zorders = zorders[::-1]



print(f"Processing {len(files)} files...")
for fil_ind, fil in enumerate(files):

    fig_savename = os.path.join(movie_frames_path, f'frame_{fil_ind:04d}.png')

    dfig, daxarr = plt.subplots(len(data_axes),sharex=True,sharey=False, \
                                figsize=(8,4*len(data_axes)), dpi=100)
    if len(data_axes) == 1:
        daxarr = [daxarr]

    # color = colors[fil_ind]
    # zorder = zorders[fil_ind]

    color = 'C0'
    zorder = 1
    
    # Display percent completion
    bu.progress_bar(fil_ind, len(files))

    # Load data
    obj = bu.hsDat(fil, load=True)

    fsamp = obj.attribs['fsamp']
    nsamp = obj.attribs['nsamp']
    t = obj.attribs['time']

    freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)

    if not fullNFFT:
        NFFT = userNFFT
    else:
        NFFT = nsamp


    for axind, ax in enumerate(data_axes):

        psd, freqs = mlab.psd(obj.dat[:,axind], Fs=obj.attribs['fsamp'], \
                                NFFT=NFFT, window=window)
        asd = np.sqrt(psd)

        plot_inds = (freqs > plot_freqs[0]) * (freqs < plot_freqs[1])


        if len(labels):
            daxarr[axind].loglog(freqs[plot_inds], asd[plot_inds], \
                                 label=labels[fil_ind], color=color, \
                                 zorder=zorder)
        else:
            daxarr[axind].loglog(freqs[plot_inds], asd[plot_inds], \
                                 color=color, \
                                 zorder=zorder)

        daxarr[axind].set_ylabel('$\\sqrt{\\mathrm{PSD}}$')
        if ax == data_axes[-1]:
            daxarr[axind].set_xlabel('Frequency [Hz]')

    if len(axes_labels):
        for labelind, label in enumerate(axes_labels):
            daxarr[labelind].set_title(label)

    if len(labels):
        daxarr[0].legend(fontsize=10)
    if len(xlim):
        daxarr[0].set_xlim(xlim[0], xlim[1])
    if len(ylim):
        daxarr[0].set_ylim(ylim[0], ylim[1])

    last_ind = int(len(data_axes)-1)
    if xticks is not None:
        daxarr[last_ind].set_xticks(xticks)
        daxarr[last_ind].set_xticklabels(map(int, xticks))
        daxarr[last_ind].set_xticks([], minor=True)

    if annotate_frame and fil_ind in annotate_indices:
        plt.text(0.03, 0.95, annotate_string, fontsize=annotate_size, \
                 fontweight='bold', color='r', \
                 ha='left', va='top', transform=daxarr[0].transAxes)

    plt.tight_layout()

    dfig.savefig(fig_savename)

    if show_first_frame and fil_ind == 0:
        plt.show()

    plt.close(dfig)

