import sys, re, os

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
plt.rcParams.update({'font.size': 18})

import bead_util as bu

import warnings
warnings.filterwarnings("ignore")


file_paths = ['/data/new_trap_processed/mod_grav/20200320_mle_vs_time.p', \
              '/data/new_trap_processed/mod_grav/20230330_mle_vs_time.p']

markers = ['s', '*']
alphas = [0.2, 1.0]

full_mles = [{12.0: [-6.23e7, 1.2e7], 18.0: [8.48e7, 2.4e7], \
              21.0: [7.91e8, 2.3e7], 33.0: [3.84e8, 1.8e7], \
              36.0: [7.54e7, 1.5e7], 39.0: [5.07e8, 3.2e7]}, \
             {12.0: [8.23e8, 2.5e8], 18.0: [2.2e8, 3.8e7], \
              21.0: [2.58e7, 5.5e8], 33.0: [8.82e7, 1.4e8], \
              36.0: [-3.33e6, 3.6e7], 39.0: [-2.99e9, 3.6e8]}]

fig_base = '/home/cblakemore/plots/20230330/mod_grav/'

savefig = False

two_panel = False
mle_inset = False
mle_panel = True

plot_freqs = [12.0, 18.0, 21.0, 33.0, 36.0, 39.0]

###############################################################
###############################################################
###############################################################


for pathind, file_path in enumerate(file_paths):

    marker = markers[pathind]
    alpha = alphas[pathind]
    full_mle = full_mles[pathind]

    mle_dat = pickle.load( open(file_path, 'rb') )

    freqs = mle_dat['naive_freqs']
    colors = bu.get_colormap(2*len(plot_freqs)+1, cmap='plasma')

    naive = mle_dat['naive']

    print(file_path)
    print(naive.shape)

    ax_dict = {0: 'X', 1: 'Y', 2: 'Z'}
    for i in [0,1,2]:
        if (mle_panel or mle_inset) and (i != 2):
            continue

        if two_panel and pathind == 0:
            fig, axarr = plt.subplots(2,1,figsize=(9.6,7.0),sharex=True, \
                                      constrained_layout=True)
            ms = 7
        elif mle_inset and pathind == 0:
            fig, ax = plt.subplots(1,1,figsize=(9.6,6.0), \
                                   constrained_layout=True)
            axarr = [ax]
            ms = 10
        elif mle_panel and pathind == 0:
            fig, axarr = plt.subplots(1,2,figsize=(9.6,6.0),\
                                      gridspec_kw={'width_ratios':[12,1]}, \
                                      constrained_layout=True)
            ms = 7

        k = 0
        for j, freq in enumerate(freqs):
            if freq not in plot_freqs:
                continue
            color = colors[2*k]
            label = '{:0.1f} Hz'.format(freq)
            axarr[0].errorbar(naive[i,j,0,:], \
                              naive[i,j,1,:]/1e8, \
                              yerr=naive[i,j,2,:]/1e8, \
                              fmt=marker, label=label, color=color, \
                              alpha=alpha, zorder=3, ms=ms)
            k += 1


        xlim = (0, naive[0,-1,0,-1]+0.5*(naive[0,0,0,1]-naive[0,0,0,0]))
        axarr[0].set_xlim(*xlim)
        axarr[0].axhline(0, ls='--', lw=2, color='k', alpha=0.6, zorder=1)
        if two_panel or mle_panel:
            axarr[1].axhline(0, ls='--', lw=2, color='k', alpha=0.6, zorder=1)
        axarr[0].legend(loc='upper left', fontsize=12, ncol=2)

        axarr[0].set_xlabel('Time [hr]')
        axarr[0].set_ylabel('$\\hat{\\alpha} ~ / ~ 10^8$')
        if not mle_panel:
            fig.tight_layout()
        fig_name = '20230330_mle-vs-time_{:s}-axis.svg'.format(ax_dict[i])

        if mle_inset and i == 2:
            inset_ax = plt.axes([0,0,1,1])
            ip = InsetPosition(axarr[0], [0.55, 0.15, 0.35, 0.12])
            inset_ax.set_axes_locator(ip)

            inset_ax.get_yaxis().set_visible(False)
            inset_ax.tick_params(labelsize=14)
            inset_ax.set_title('MLE for full data set', fontsize=14)

            k = 0
            for j, freq in enumerate(freqs):
                color = colors[2*j]
                inset_ax.errorbar([full_mle[freq][0]/1.0e8], [k+1], \
                                  yerr=None, xerr=[full_mle[freq][1]/1.0e8], \
                                  fmt='s', color=color, alpha=1.0, zorder=3, ms=5)
                k += 1

            inset_ax.set_ylim(-1, 7)
            inset_ax.set_xlabel('$\\hat{\\alpha} ~ / ~ 10^8$', fontsize=14)
            fig_name = fig_name.replace('.svg', '_inset.svg')

        if mle_panel and i == 2:
            ylim0 = axarr[0].get_ylim()
            xlim0 = axarr[0].get_xlim()
            ylim1 = [ylim0[0]/3.0, ylim0[1]/3.0]
            axarr[1].set_ylim(*ylim1)
            # axarr[1].set_ylim(ylim0[0], ylim0[1])
            axarr[1].set_xlim(-1, 7)

            axarr[1].yaxis.set_label_position("right")
            axarr[1].yaxis.tick_right()
            axarr[1].get_xaxis().set_visible(False)

            axarr[1].set_ylabel('$\\hat{\\alpha} ~ / ~ 10^8$ - full data set')        

            k = 0
            for j, freq in enumerate(freqs):
                if freq not in plot_freqs:
                    continue
                color = colors[2*k]
                axarr[1].errorbar([k+1], [full_mle[freq][0]/1.0e8], \
                                  yerr=[full_mle[freq][1]/1.0e8], xerr=None, \
                                  fmt=marker, color=color, alpha=alpha, \
                                  zorder=3, ms=5)
                k += 1

            fig.tight_layout()
            axarr[1].set_ylabel('$\\hat{\\alpha} ~ / ~ 10^8$ - full data set', \
                                labelpad=15)
            plt.subplots_adjust(right=0.85)

            fig_name = fig_name.replace('.svg', '_mle-panel.svg')

axarr[0].add_patch(patches.ConnectionPatch( \
    xyA=(xlim0[1], ylim1[1]), xyB=(-1, ylim1[1]), \
    coordsA='data', coordsB='data',\
    axesA=axarr[0], axesB=axarr[1], \
    shrinkA=6.0, shrinkB=2.0, \
    linestyle='dashed'))

axarr[0].add_patch(patches.ConnectionPatch( \
    xyA=(xlim0[1], ylim1[0]), xyB=(-1, ylim1[0]), \
    coordsA='data', coordsB='data', \
    axesA=axarr[0], axesB=axarr[1], \
    shrinkA=6.0, shrinkB=2.0, \
    linestyle='dashed'))

if savefig:
    fig.savefig(os.path.join(fig_base, fig_name))

plt.show()