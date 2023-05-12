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

data_dirs = [#'/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz', \
             #'/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz_elec-term', \
             #\
             #'/data/20180704/bead1/grav_data/shield', \
             #'/data/20180704/bead1/grav_data/shield_1s_1h', \
             #'/data/20180704/bead1/grav_data/shield2', \
             #'/data/20180704/bead1/grav_data/shield3', \
             #'/data/20180704/bead1/grav_data/shield4', \
             '/data/20180704/no_bead/grav_data/shield', \
             #\
             #'/data/20180808/bead4/grav_data/shield1'
             ]

fit_type = 'Gaussian'
#fit_type = 'Planar'

p0_bead_dict = {'20180625': [19.0,40.0,20.0], \
                '20180704': [18.7,40.0,20.0], \
                '20180808': [18,40.0,23.0] \
                }

load_agg = True

harms = [1,2,3,4,5,6]

#opt_ext = 'TEST'
opt_ext = '_6harm-full'



if fit_type == 'Gaussian':
    data_ind = 2
    err_ind = 4
if fit_type == 'Planar':
    data_ind = 0
    err_ind = 1


for ddir in data_dirs:
    print()

    parts = ddir.split('/')
    date = parts[2]
    p0_bead = p0_bead_dict[date]
    
    nobead = ('no_bead' in parts) or ('nobead' in parts) or ('no-bead' in parts)
    if nobead:
        opt_ext += '_NO-BEAD'

    agg_path = '/processed_data/aggdat/' + date + '_' + parts[-1] + opt_ext + '.agg'
    alpha_arr_path = '/processed_data/alpha_arrs/' + date + '_' + parts[-1] + opt_ext + '.arr'
    lambda_path = alpha_arr_path[:-4] + '_lambdas.arr'

    if load_agg:

        print(agg_path)

        agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms)
        agg_dat.load(agg_path)

        agg_dat.reload_grav_funcs()

        #agg_dat.fit_alpha_xyz_vs_alldim(weight_planar=False, plot=False, plot_hists=True)
        alpha_arr = agg_dat.alpha_xyz_best_fit
        lambdas = agg_dat.lambdas

        np.save(open(alpha_arr_path, 'wb'), alpha_arr)
        np.save(open(lambda_path, 'wb'), agg_dat.lambdas)

    else:
        alpha_arr = np.load(open(alpha_arr_path, 'rb'))
        lambdas = np.load(open(lambda_path, 'rb'))


    Ncomp = alpha_arr.shape[-2]
    comp_colors = bu.get_colormap(Ncomp, cmap='viridis')

    alpha_w = np.sum(alpha_arr[:,0:2,:,data_ind]*alpha_arr[:,0:2,:,err_ind]**(-2), axis=1) / \
                  np.sum(alpha_arr[:,0:2,:,err_ind]**(-2), axis=1)

    #alpha_w = np.sum(alpha_arr[:,0:2,:,2], axis=1) * 0.5

    errs_x = np.zeros_like(alpha_arr[:,0,0,0])
    N = 0
    for ind in range(Ncomp - 1):
        errs_x += alpha_w[:,ind+1]**2
        N += 1
    errs_x = np.sqrt(errs_x / N)

    sigma_alpha_w = 1.0 / np.sqrt( np.sum(alpha_arr[:,:2,:,3]**(-2), axis=1) )
    N_w = np.sum(alpha_arr[:,:2,:,7], axis=1)
    
    plt.figure(1)
    if nobead:
        plt.title(date + '_' + 'no-bead' + ': Result of %s Fitting' % fit_type, fontsize=16)
    else:
        plt.title(date + '_' + parts[-1] + ': Result of %s Fitting' % fit_type, fontsize=16)

    plt.loglog(lambdas, np.abs(alpha_w[:,0]), lw=4, \
               label='Template basis vector')

    plt.loglog(lambdas, errs_x, '--', lw=2, \
               label='Quadrature sum of other vectors')       

    plt.loglog(gu.limitdata[:,0], gu.limitdata[:,1], '--', label=gu.limitlab, \
               linewidth=3, color='r')
    plt.loglog(gu.limitdata2[:,0], gu.limitdata2[:,1], '--', label=gu.limitlab2, \
               linewidth=3, color='k')
    plt.xlabel('Length Scale: $\lambda$ [m]')
    plt.ylabel('Strength: |$\\alpha$| [arb]')
    plt.xlim(1e-7, 1e-3)
    plt.ylim(1e4, 1e14)
    plt.legend()
    plt.grid()
    plt.show()


    
    for ind in range(Ncomp):
        fig2 = plt.figure(2)
        plt.title("%s fit for Basis Vector: %i" % (fit_type, ind))

        plt.loglog(lambdas, np.abs(alpha_arr[:,0,ind,data_ind]), \
                   color=comp_colors[ind], ls='--', label='$\\alpha_x$')
        plt.loglog(lambdas, np.abs(alpha_arr[:,0,ind,err_ind]), \
                   color=comp_colors[ind], ls='--', label='$\sigma_{\\alpha_x}$', \
                   alpha=0.5)
        plt.loglog(lambdas, np.abs(alpha_w[:,ind]), \
                   color=comp_colors[ind], ls='-', lw=3, label='Weighted mean')
        plt.loglog(lambdas, np.abs(alpha_arr[:,1,ind,data_ind]), \
                   color=comp_colors[ind], ls='-.', label='$\\alpha_y$')
        plt.loglog(lambdas, np.abs(alpha_arr[:,1,ind,err_ind]), \
                   color=comp_colors[ind], ls='-.', label='$\sigma_{\\alpha_y}$', \
                   alpha=0.5)
        plt.xlabel('Length Scale: $\lambda$ [m]')
        plt.ylabel('Strength: |$\\alpha$| [arb]')
        plt.xlim(1e-6, 1e-3)
        plt.ylim(1e6, 1e15)
        plt.legend()
        plt.grid()

        fig_title = '/home/charles/plots/' + date + '/' + parts[-1] + '/' \
                    + date + '_' + parts[-1] + '_%s-fit_comp%i.png' % (fit_type, ind)

        fig2.savefig(fig_title)
        plt.close(fig2)
        
        #plt.show()
    
    #for fig_num in [1,2,3]:
    #    plt.figure(fig_num)
    #    plt.xlabel('Length Scale: $\lambda$ [m]')
    #    plt.ylabel('Strength: |$\\alpha$| [arb]')
    #    plt.legend()
    #    plt.grid()
    #plt.show()
