import sys, re, os

import dill as pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import grav_util_3 as gu
import bead_util as bu
import configuration as config

plt.rcParams.update({'font.size': 14})

import warnings
warnings.filterwarnings("ignore")

ncore = 30
# ncore = 1



theory_data_dir = '/data/old_trap/grav_sim_data/2um_spacing_data/'

yuklambda = 100.0e-6

# seps = [9.0]
seps = [9.0, 11.0, 13.0, 15.0, 18.0, 21.0]

# heights = [6.0]
heights = np.linspace(-10.0, 10.0, 21)

posvec = np.linspace(-250.0, 250.0, 501)

gfuncs_class = gu.GravFuncs(theory_data_dir)




#####################################################################
#####################################################################
####################################################################
lambind = np.argmin(np.abs(gfuncs_class.lambdas - yuklambda))

sep_fig, sep_axarr = plt.subplots(3, 1, sharex=True, sharey=True, \
                                        figsize=(8,8))
height_fig, height_axarr = plt.subplots(3, 1, sharex=True, sharey=True, \
                                        figsize=(8,8))

minsep = np.min(seps)

colors = bu.get_color_map(len(seps), cmap='plasma')
for sepind, sep in enumerate(seps):

    ones = np.ones_like(posvec)
    pts = np.stack((sep*ones, posvec, 6.0*ones), axis=-1)

    for resp in [0,1,2]:

        yukforce = gfuncs_class.yukfuncs[resp][lambind](pts*1.0e-6)

        sep_axarr[resp].plot(posvec, yukforce, color=colors[sepind], \
                                label='$\\Delta x = {:0.1f}$ um'.format(sep))

ax_dict = {0: 'X', 1: 'Y', 2: 'Z'}
for resp in [0,1,2]:
    sep_axarr[resp].set_ylabel('{:s} Force [N]'.format(ax_dict[resp]))
sep_axarr[-1].set_xlabel('Position Along Density Modulation [um]')

sep_axarr[1].legend(fontsize=10,ncol=2)
sep_axarr[0].set_title('Yukawa-modified gravity for $\\alpha = 1$, $\\lambda = {:0.1f}$ um'\
                                .format(yuklambda*1e6), fontsize=14)
sep_fig.tight_layout()





colors = bu.get_color_map(len(heights), cmap='coolwarm')
for heightind, height in enumerate(heights):

    ones = np.ones_like(posvec)
    pts = np.stack((minsep*ones, posvec, height*ones), axis=-1)

    for resp in [0,1,2]:

        yukforce = gfuncs_class.yukfuncs[resp][lambind](pts*1.0e-6)

        height_axarr[resp].plot(posvec, yukforce, color=colors[heightind], \
                                label='$\\Delta z = {:0.1f}$ um'.format(height))

ax_dict = {0: 'X', 1: 'Y', 2: 'Z'}
for resp in [0,1,2]:
    height_axarr[resp].set_ylabel('{:s} Force [N]'.format(ax_dict[resp]))
height_axarr[-1].set_xlabel('Position Along Density Modulation [um]')

height_axarr[1].legend(fontsize=10,ncol=2)
height_axarr[0]
height_fig.tight_layout()






plt.show()






