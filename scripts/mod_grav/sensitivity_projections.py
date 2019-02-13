import dill as pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import grav_util_3 as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")

theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'


grav_func_dict = gu.build_mod_grav_funcs(theory_data_dir)
lambdas = grav_func_dict['lambdas']



sep = 10e-6     # m
noise = 1.0e-18   # N/rt(Hz)
int_time = 1e5    # s


posvec = np.linspace(-40.0e-6, 40.0e-6, 100)
ones = np.ones_like(posvec)
pts = np.stack((sep*ones, posvec, 0.0*ones), axis=-1)





alphas = []
for yukind, yuklambda in enumerate(lambdas):
    yukforce = grav_func_dict['yukfuncs'][0][yukind](pts)
    diff = np.max(yukforce) - np.min(yukforce)

    alpha = noise * (1.0 / np.sqrt(int_time)) / diff
    alphas.append(alpha)

plt.loglog(lambdas, alphas)
plt.show()

outarr = np.array([lambdas, alphas]).T

np.savetxt('/home/charles/opt_lev_analysis/scripts/sense_plot/projections/attractorv2_sep10um_noise1e-18NrtHz_int1e5s.txt', outarr, delimiter=',')
