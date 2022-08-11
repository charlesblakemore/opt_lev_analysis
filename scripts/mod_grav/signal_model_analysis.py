import sys, re, os

import dill as pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import grav_util_3 as gu
import bead_util as bu
import configuration as config

plt.rcParams.update({'font.size': 16})

import warnings
warnings.filterwarnings("ignore")

theory_base = '/home/cblakemore/opt_lev_analysis/gravity_sim/results'
sim = '7_6um-gbead_1um-unit-cells_close/'
sim = '7_6um-gbead_1um-unit-cells_master/'
# sim = '15um-gbead_1um-unit-cells_close/'

theory_data_dir = os.path.join(theory_base, sim)

yuklambda = 10.0e-6

seps = [7.5]
# seps = [9.0, 11.0, 13.0, 15.0, 18.0, 21.0]
# seps = [2.5, 5.0, 7.5, 10.0]#, 12.5]

# heights = [6.0]
heights = np.linspace(-20.0, 20.0, 51)
# heights = np.linspace(-10.0, 10.0, 7)
# heights = np.linspace(-5.0, 5.0, 7)

posvec = np.linspace(-249.5, 249.5, 501)

gfuncs_class = gu.GravFuncs(theory_data_dir)
rbead = gfuncs_class.rbead * 1e6

# Factor to adjust density
# fac = 1550.0 / 1850.0
fac = 1.0

fig_title = 'Yukawa-modified gravity: \n $d_{{\\rm MS}}={:0.2f}~\\mu$m'.format(2.0*rbead) \
                + ', $\\alpha = 1$, $\\lambda = {:0.1f}~\\mu$m'.format(yuklambda*1e6)


#####################################################################
#####################################################################
####################################################################
lambind = np.argmin(np.abs(gfuncs_class.lambdas - yuklambda))

sep_fig, sep_axarr = plt.subplots(3, 1, sharex=True, sharey=True, \
                                        figsize=(7,7))
height_fig, height_axarr = plt.subplots(3, 1, sharex=True, sharey=True, \
                                        figsize=(8,8))

modamp_fig, modamp_ax = plt.subplots(1,1)


minsep = np.min(seps)

colors = bu.get_colormap(len(seps), cmap='plasma')
for sepind, sep in enumerate(seps):

    ones = np.ones_like(posvec)
    pts = np.stack((sep*ones + rbead, posvec, 5.0*ones), axis=-1)

    for resp in [0,1,2]:

        yukforce = gfuncs_class.yukfuncs[resp][lambind](pts*1.0e-6)

        sep_axarr[resp].plot(posvec, fac*yukforce, color=colors[sepind], \
                                label='$\\Delta x = {:0.1f}~\\mu$m'.format(sep))

ax_dict = {0: 'X', 1: 'Y', 2: 'Z'}
for resp in [0,1,2]:
    sep_axarr[resp].set_ylabel('{:s} Force [N]'.format(ax_dict[resp]))
sep_axarr[-1].set_xlabel('Position Along Density Modulation [$\\mu$m]')
sep_axarr[0].set_xlim(np.min(posvec), np.max(posvec))

sep_axarr[1].legend(fontsize=14,ncol=2)
sep_axarr[0].set_title(fig_title, fontsize=16)
sep_axarr[2].text(-175, -2.5e-24, '$\\Delta z = {:0.1f}~\\mu$m'.format(5.0), \
                  ha='center', va='center', fontdict={'size':14})
sep_fig.tight_layout()



modamps = [[], [], []]

colors = bu.get_colormap(len(heights), cmap='coolwarm')
for heightind, height in enumerate(heights):

    ones = np.ones_like(posvec)
    pts = np.stack((minsep*ones + rbead, posvec, height*ones), axis=-1)

    for resp in [0,1,2]:

        yukforce = gfuncs_class.yukfuncs[resp][lambind](pts*1.0e-6)

        height_axarr[resp].plot(posvec, fac*yukforce, color=colors[heightind], \
                                label='$\\Delta z = {:0.1f}$ um'.format(height))

        inds = np.abs(posvec) <= 50.0
        modamps[resp].append(np.max(yukforce[inds]) - np.min(yukforce[inds]))

ax_dict = {0: 'X', 1: 'Y', 2: 'Z'}
for resp in [0,1,2]:
    height_axarr[resp].set_ylabel('{:s} Force [N]'.format(ax_dict[resp]))
height_axarr[-1].set_xlabel('Position Along Density Modulation [um]')
height_axarr[0].set_xlim(np.min(posvec), np.max(posvec))

height_axarr[1].legend(fontsize=14,ncol=2)
height_axarr[0]
height_fig.tight_layout()




for resp in [0,1,2]:
    modamp_ax.plot(heights, modamps[resp], label='{:s}'.format(ax_dict[resp]))

modamp_ax.set_title('Modulation vs. Z with $\\Delta x = {:0.1f}~\\mu$m'.format(minsep))
modamp_ax.set_xlabel('Bead Height Relative to Attractor Center [um]')
modamp_ax.set_ylabel('Amplitude of Modulation [N]')
modamp_ax.legend(loc=0)
modamp_fig.tight_layout()





plt.show()






