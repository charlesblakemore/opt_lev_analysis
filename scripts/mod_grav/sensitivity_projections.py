import os, sys
import dill as pickle

import tracemalloc

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import grav_util as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")

theory_base = '/home/cblakemore/opt_lev_analysis/gravity_sim/results'
sim = '7_6um-gbead_1um-unit-cells_master/'
# sim = '15um-gbead_1um-unit-cells_close_morelamb/'

theory_data_dir = os.path.join(theory_base, sim)

tracemalloc.start()
gfuncs_class = gu.GravFuncs(theory_data_dir)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.start()


lambdas = gfuncs_class.lambdas
rbead = gfuncs_class.rbead * 1e6


sep = 3.0         # um
noise = 7.0e-20   # N/rt(Hz)
int_time = 3e6    # s

save_base = '/home/cblakemore/opt_lev_analysis/scripts/sense_plot/projections/'
proj_name = 'attractorv2_rbead{:0.1f}um_sep{:0.1f}um_noise{:1.0e}NrtHz_int{:1.0e}s'\
				.format(rbead, sep, noise, int_time)
proj_name = proj_name.replace('.', '_')
save_filename = os.path.join(save_base, proj_name+'.txt')
print(save_filename)

############################################################################
############################################################################
############################################################################

posvec = np.linspace(-50.0, 50.0, 100)
ones = np.ones_like(posvec)
pts = np.stack(((sep+rbead)*ones, posvec, 0.0*ones), axis=-1)

alphas = []
for yukind, yuklambda in enumerate(lambdas):
    yukforce = gfuncs_class.yukfuncs[0][yukind](pts*1.0e-6)
    diff = np.max(yukforce) - np.min(yukforce)

    alpha = noise * (1.0 / np.sqrt(int_time)) / diff
    alphas.append(alpha)

plt.loglog(lambdas, alphas)
plt.show()

outarr = np.array([lambdas, alphas]).T

np.savetxt(save_filename, outarr, delimiter=',')
