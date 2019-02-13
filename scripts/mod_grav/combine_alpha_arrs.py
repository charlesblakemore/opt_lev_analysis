import dill as pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import grav_util_3 as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")



arr_paths = [#'/processed_data/alpha_arrs/20180625_X50-75um_Z15-25um_17Hz_6harm-full.arr', \
             #'/processed_data/alpha_arrs/20180625_X50-75um_Z15-25um_17Hz_elec-term_6harm-full.arr', \
             \
             '/processed_data/alpha_arrs/20180704_shield_6harm-full.arr', \
             '/processed_data/alpha_arrs/20180704_shield2_6harm-full.arr', \
             '/processed_data/alpha_arrs/20180704_shield3_6harm-full.arr', \
             '/processed_data/alpha_arrs/20180704_shield4_6harm-full.arr', \
             ]

date = '20180704'

fit_type = 'Gaussian'






#################################################

if fit_type == 'Gaussian':
    data_ind = 2
    err_ind = 4
if fit_type == 'Planar':
    data_ind = 0
    err_ind = 1


weighted_alphas = []
weighted_errs = []

for ind, path in enumerate(arr_paths):

    print ind, path

    alpha_arr = np.load( open(path, 'rb') )
    lambdas = np.load( open(path[:-4] + '_lambdas.arr', 'rb') )

    Ncomp = alpha_arr.shape[-2]
    comp_colors = bu.get_color_map(Ncomp, cmap='viridis')


    ### Compute the weighted mean of the X and Y projections of the data onto signal template
    ### and the uncertainty of the weighted mean
    ###
    ### The resulting object is indexed as [lambda_ind, component_ind]
    ###
    alpha_w = np.sum(alpha_arr[:,0:2,:,data_ind]*alpha_arr[:,0:2,:,err_ind]**(-2), axis=1) / \
                  np.sum(alpha_arr[:,0:2,:,err_ind]**(-2), axis=1)

    alpha_err_w = 1.0 / np.sqrt( np.sum(alpha_arr[:,0:2,:,err_ind]**(-2), axis=1) )


    weighted_alphas.append(alpha_w)
    weighted_errs.append(alpha_err_w)

    errs_x = np.zeros_like(alpha_arr[:,0,0,0])
    N = 0
    for comp_ind in range(Ncomp - 1):
        errs_x += alpha_w[:,comp_ind+1]**2
        N += 1
    errs_x = np.sqrt(errs_x / N)



    plt.figure(1)
    plt.title(date + ': Result of %s Fitting' % fit_type, fontsize=16)

    #plt.loglog(lambdas, np.abs(alpha_w[:,0]), lw=2, \
    #           label='Dataset: %i' % ind, color='C'+str(ind))

    #plt.loglog(lambdas, alpha_err_w[:,0], '--', lw=2, \
    #           label='Gauss Err', color='C'+str(ind))   




plt.loglog(gu.limitdata[:,0], gu.limitdata[:,1], '-.',# label=gu.limitlab, \
           linewidth=3, color='k', alpha=0.5)
plt.loglog(gu.limitdata2[:,0], gu.limitdata2[:,1], '-.',# label=gu.limitlab2, \
           linewidth=3, color='k')



all_alpha_arr = np.stack(weighted_alphas)
all_err_arr = np.stack(weighted_errs)

full_alpha_w = np.sum(all_alpha_arr[:,:,:] * all_err_arr[:,:,:]**(-2), axis=0) / \
               np.sum(all_err_arr[:,:,:]**(-2), axis=0)
full_err_w = 1.0 / np.sqrt( np.sum(all_err_arr[:,:,:]**(-2), axis=0) )

plt.loglog(lambdas, np.abs(full_alpha_w[:,0]), lw=3, \
               label='Weighted Mean', color='r')
plt.loglog(lambdas, np.abs(full_err_w[:,0]), lw=3, ls='--', \
               label='Error on Weighted Mean', color='r')

plt.xlabel('Length Scale: $\lambda$ [m]')
plt.ylabel('Strength: |$\\alpha$| [arb]')
plt.xlim(1e-7, 1e-3)
plt.ylim(1e4, 1e14)
plt.legend(loc=3)
plt.grid()
    
plt.show()



