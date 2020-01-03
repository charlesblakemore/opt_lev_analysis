import dill as pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import grav_util_3 as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")



theory_data_dir = '/data/old_trap/grav_sim_data/2um_spacing_data/'

data_dirs = ['/data/old_trap/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz', \
             '/data/old_trap/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz_elec-term', \
             #\
             '/data/old_trap/20180704/bead1/grav_data/shield', \
             '/data/old_trap/20180704/bead1/grav_data/shield_1s_1h', \
             #'/data/old_trap/20180704/bead1/grav_data/shield2', \
             #'/data/old_trap/20180704/bead1/grav_data/shield3', \
             #'/data/old_trap/20180704/bead1/grav_data/shield4', \
             #'/data/old_trap/20180704/no_bead/grav_data/shield', \
             #\
             #'/data/old_trap/20180808/bead4/grav_data/shield1'
             ]


data_dirs = ['/data/new_trap/20191204/Bead1/Shaking/Shaking370/']
new_trap = True


Nfiles = 100

redo_alphafit = False
save = False
plot_end_result = True

plot_harms = False
plot_basis = True
plot_alpha_xyz = True

save_hists = False

p0_bead_dict = {'20180625': [19.0,40.0,20.0], \
                '20180704': [18.7,40.0,20.0], \
                '20180808': [18.0,40.0,20.0] \
                }


p0_bead_dict = {'20191204': [385.0, 200.0, 29.0], \
                }

new_trap = True

harms = [1,2,3,4,5,6]

#opt_ext = 'TEST'
opt_ext = '_6harm-full'


for ddir in data_dirs:
    # Skip the ones I've already calculated
    #if ddir == data_dirs[0]:
    #    continue
    print()

    paths = gu.build_paths(ddir, opt_ext)
    agg_path = paths['agg_path']
    p0_bead = p0_bead_dict[paths['date']]

    if not redo_alphafit:
        datafiles, lengths = bu.find_all_fnames(ddir, ext=config.extensions['data'])[:Nfiles]

        agg_dat = gu.AggregateData(datafiles, p0_bead=p0_bead, harms=harms, reload_dat=True, \
                                   plot_harm_extraction=plot_harms, new_trap=new_trap, \
                                   step_cal_drive_freq=151.0)

        agg_dat.load_grav_funcs(theory_data_dir)

        if save:
            agg_dat.save(agg_path)
        
        #agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms)
        #agg_dat.load(agg_path)

        agg_dat.bin_rough_stage_positions()
        #agg_dat.average_resp_by_coordinate()

        agg_dat.plot_force_plane(resp=0, fig_ind=1, show=False)
        agg_dat.plot_force_plane(resp=1, fig_ind=2, show=False)
        agg_dat.plot_force_plane(resp=2, fig_ind=3, show=True)

        agg_dat.find_alpha_xyz_from_templates(plot=plot_alpha_xyz, plot_basis=plot_basis)

        if save:
            agg_dat.save(agg_path)


        agg_dat.fit_alpha_xyz_vs_alldim()

        if save:
            agg_dat.save(agg_path)




    else:
        agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms)
        agg_dat.load(agg_path)

        agg_dat.bin_rough_stage_positions()
        #agg_dat.average_resp_by_coordinate()

        agg_dat.find_alpha_xyz_from_templates(plot=False, plot_basis=False)

        if save:
            agg_dat.save(agg_path)

        #agg_dat.plot_force_plane(resp=2)
        agg_dat.reload_grav_funcs()
        #agg_dat.plot_alpha_xyz_dict(resp=2, lambind=35)

        hist_prefix = date + '_' + parts[-1] + opt_ext + '_'
        hist_info = {'date': date, 'prefix': hist_prefix}

        agg_dat.fit_alpha_xyz_vs_alldim(weight_planar=True, plot=plot_alpha_xyz, \
                                        save_hists=save_hists, hist_info=hist_info)
        alpha_arr = agg_dat.alpha_xyz_best_fit

        if save:
            agg_dat.save(agg_path)




        if plot:
            alpha_w = np.sum(alpha_arr[:,0:2,:,0]*alpha_arr[:,0:2,:,1]**(-2), axis=1) / \
                      np.sum(alpha_arr[:,0:2,:,1]**(-2), axis=1)

            errs_x = np.zeros_like(alpha_arr[:,0,0,0])
            N = 0
            for ind in range(2 * np.sum(agg_dat.ginds) - 1):
                errs_x += alpha_w[:,ind+1]**2
                N += 1

            errs_x = np.sqrt(errs_x / N)

            #plt.loglog(agg_dat.lambdas, np.abs(alpha_arr[:,0,0,1]), lw=2, label='Gaussian')
            #plt.loglog(agg_dat.lambdas, np.abs(alpha_arr[:,0,0,2]), lw=2, label='Cauchy')

            plt.title('Result of Planar Fitting', fontsize=16)

            plt.loglog(agg_dat.lambdas, np.abs(alpha_w[:,0]), lw=4, \
                       label='Template basis vector')

            plt.loglog(agg_dat.lambdas, errs_x, '--', lw=2, \
                       label='Quadrature sum of other vectors')       

            plt.loglog(gu.limitdata[:,0], gu.limitdata[:,1], '--', label=gu.limitlab, \
                       linewidth=3, color='r')
            plt.loglog(gu.limitdata2[:,0], gu.limitdata2[:,1], '--', label=gu.limitlab2, \
                       linewidth=3, color='k')
            plt.xlabel('Length Scale: $\lambda$ [m]')
            plt.ylabel('Strength: |$\\alpha$| [arb]')
            plt.legend()
            plt.grid()
            plt.show()
        


