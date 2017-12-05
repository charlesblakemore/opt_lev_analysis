import sys, time

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opti

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config


savepath = '/sensitivities/20171106data_95cl_alpha_lambda.npy'
figtitle = 'Sensitivity: Patterned Attractor'

lambdas, alphas = np.load(savepath)


### Load limits to plot against

limitdata_path = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160928_datathief_nodecca2.txt'
#limitdata_path = '/home/charles/limit_nodecca2.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160914_datathief.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'



fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)

ax.loglog(lambdas, alphas, linewidth=2, label='95% CL')
ax.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
ax.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
ax.grid()

ax.set_xlabel('$\lambda$ [m]')
ax.set_ylabel('$\\alpha$')

ax.legend(numpoints=1, fontsize=9)

ax.set_title(figtitle)

plt.tight_layout(w_pad=1.2, h_pad=1.2, pad=1.2)

plt.show()
