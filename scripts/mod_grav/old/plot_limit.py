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
figtitle = 'Sensitivity: Multi-Dimensional Fit'

lims_to_plot = [['/sensitivities/20171106data_95cl_alpha_lambda.npy', 'All'], \
                ['/sensitivities/20171106data_95cl_alpha_lambda_closepoints.npy', 'Close'], \
                ['/sensitivities/20171106data_95cl_alpha_lambda_farpoints.npy', 'Far'], \
                ['/sensitivities/20170903data_95cl_alpha_lambda.npy', 'All'], \
                ['/sensitivities/20170903data_95cl_alpha_lambda_closepoints.npy', 'Close'], \
                ['/sensitivities/20170903data_95cl_alpha_lambda_farpoints.npy', 'Far']]

lambdas, alphas = np.load(savepath)


### Load limits to plot against

#limitdata_path = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/limitdata_20160928_datathief_nodecca2.txt'
limitdata_path = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/no_decca2_limit.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

#limitdata_path2 = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/limitdata_20160914_datathief.txt'
limitdata_path2 = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/decca2_limit.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'



fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)

for lim in lims_to_plot:
    lambdas, alphas = np.load(lim[0])
    lab = lim[1]
    if '20171106' in lim[0]:
        color = 'C0'
        lab = 'Patt ' + lab
    if '20170903' in lim[0]:
        color = 'C1'
        lab = 'No Patt ' + lab
    
    if 'closepoints' in lim[0]:
        style = '-.'
    elif 'farpoints' in lim[0]:
        style = ':'
    else:
        style = '-'

    ax.loglog(lambdas, alphas, style, linewidth=2, label=lab, color=color)

ax.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
ax.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
ax.grid()

ax.set_xlabel('$\lambda$ [m]')
ax.set_ylabel('$\\alpha$')

ax.legend(numpoints=1, fontsize=9)

ax.set_title(figtitle)

plt.tight_layout(w_pad=1.2, h_pad=1.2, pad=1.2)

plt.show()
