import sys, re, os

import dill as pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

import grav_util_3 as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")


# ncore = 30
ncore = 1



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


# data_dirs = ['/data/new_trap/20191204/Bead1/Shaking/Shaking370/']
# data_dirs = ['/data/new_trap/20200107/Bead3/Shaking/Shaking380/']
# data_dirs = ['/data/new_trap/20200113/Bead1/Shaking/Shaking377/']
data_dirs = [#'/data/new_trap/20200210/Bead2/Shaking/Shaking382/', \
             '/data/new_trap/20200210/Bead2/Shaking/Shaking384/']
new_trap = True


tf_interp = True

#substr = ''
# substr = 'Shaking0' # for 20200210/.../...382/
substr = 'Shaking3'  # for 20200210/.../...384/

Nfiles = 16000
# Nfiles = 5

# reprocess = True
# save = True
reprocess = False
save = False
plot_end_result = False

# redo_alpha_fit = True
redo_alpha_fit = False

plot_harms = False
plot_basis = False
plot_alpha_xyz = False
plot_bad_alphas = True

save_hists = False

# Position of bead relative to attractor AT FULLEST EXTENT OF ATTRACTOR
# Thus for new trap data where the bead is usually within the range of 
# x-axis, this number can be negative
p0_bead_dict = {'20180625': [19.0,40.0,20.0], \
                '20180704': [18.7,40.0,20.0], \
                '20180808': [18.0,40.0,20.0] \
                }

p0_bead_dict = {# '20191204': [385.0, 200.0, 29.0], \
                '20191204': [-115.0, 200.0, 29.0], \
                # '20200107': [-111, 190.0, 17.0], \
                '20200107': [-111, 190.0, 25.9], \
                # '20200113': [-114.6, 184.5, 16.0], \
                '20200113': [-114.6, 184.5, 10.0], \
                # '20200210': [382.0 - 500.0, 184.5, 52.0], \
                '20200210': [392.0 - 500.0, 184.5, 57.67], \
                }

# harms = [6]
harms = [3,4,5,6]
# harms = [3,4,5,6,7,8,9,10]

#opt_ext = 'TEST'
opt_ext = '_harms-'
for harm in harms:
    opt_ext += str(int(harm))
opt_ext += '_deltaz-6um_first-{:d}'.format(Nfiles)
if len(substr):
    opt_ext += '_{:s}'.format(substr)


for ddir in data_dirs:
    # Skip the ones I've already calculated
    #if ddir == data_dirs[0]:
    #    continue
    print()

    paths = gu.build_paths(ddir, opt_ext, new_trap=new_trap)
    agg_path = paths['agg_path']
    p0_bead = p0_bead_dict[paths['date']]

    if save:
        bu.make_all_pardirs(agg_path)


    if reprocess:
        datafiles, lengths = bu.find_all_fnames(ddir, ext=config.extensions['data'], \
                                                substr=substr)
        datafiles = datafiles[:Nfiles]

        agg_dat = gu.AggregateData(datafiles, p0_bead=p0_bead, harms=harms, reload_dat=True, \
                                   plot_harm_extraction=plot_harms, new_trap=new_trap, \
                                   step_cal_drive_freq=151.0, ncore=ncore, noisebins=10, \
                                   tf_interp=tf_interp)

        agg_dat.load_grav_funcs(theory_data_dir)

        if save:
            agg_dat.save(agg_path)

        agg_dat.bin_rough_stage_positions()
        #agg_dat.average_resp_by_coordinate()

        # agg_dat.plot_force_plane(resp=0, fig_ind=1, show=False)
        # agg_dat.plot_force_plane(resp=1, fig_ind=2, show=False)
        # agg_dat.plot_force_plane(resp=2, fig_ind=3, show=True)

        agg_dat.find_alpha_xyz_from_templates(plot=plot_alpha_xyz, plot_basis=plot_basis, \
                                                ncore=ncore)#, add_fake_data=True, fake_alpha=1e9)

        if save:
            agg_dat.save(agg_path)


        # agg_dat.fit_alpha_xyz_vs_alldim()
        agg_dat.fit_alpha_xyz_onepos_simple(resp=2)

        if save:
            agg_dat.save(agg_path)

        agg_dat.plot_sensitivity()




    else:
        agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms, new_trap=new_trap)
        agg_dat.load(agg_path)

        agg_dat.bin_rough_stage_positions()
        #agg_dat.average_resp_by_coordinate()

        if redo_alpha_fit:   
            agg_dat.find_alpha_xyz_from_templates(plot=plot_alpha_xyz, plot_basis=plot_basis, \
                                                    ncore=ncore, plot_bad_alphas=plot_bad_alphas, \
                                                    # add_fake_data=True, fake_alpha=1e9, \
                                                    )

        agg_dat.fit_alpha_xyz_onepos_simple(resp=2, plot=True, show=True)
        # agg_dat.fit_alpha_xyz_onepos(resp=2)
        agg_dat.plot_sensitivity()

        # filenums = [5, 50, 500, 5000]
        filenums = list(map(int, np.geomspace(10,14000,15)))
        # plot_alphas = list(np.logspace(-3, 0, 20))
        plot_alphas = list(np.exp(np.linspace(-3, 0, len(filenums))))
        print('{:d} total...'.format(len(filenums)))

        show = False
        colors = bu.get_color_map(len(filenums), cmap='BuPu')
        for i, j in enumerate(filenums):
            print(i, end=', ')
            # print(ind)
            if j == filenums[-1]:
                show = True
            agg_dat.fit_alpha_xyz_onepos_simple(resp=2, last_file=j, plot=True, \
                                                show=show, plot_color=colors[i], \
                                                plot_label='{:d} files'.format(j)) #, \
                                                # plot_alpha=plot_alphas[i])
        print()

        #agg_dat.plot_force_plane(resp=0, fig_ind=1, show=False)
        #agg_dat.plot_force_plane(resp=1, fig_ind=2, show=False)
        #agg_dat.plot_force_plane(resp=2, fig_ind=3, show=True)

        # agg_dat.find_alpha_xyz_from_templates(plot=plot_alpha_xyz, plot_basis=plot_basis, \
        #                                         ncore=ncore)
        # agg_dat.plot_alpha_xyz_dict(k=0)
        # agg_dat.plot_alpha_xyz_dict(k=1)
        # agg_dat.plot_alpha_xyz_dict(k=2)
        # agg_dat.plot_alpha_xyz_dict(lambind=10)
        # agg_dat.plot_alpha_xyz_dict(lambind=50)

        # if save:
        #     agg_dat.save(agg_path)

        # #agg_dat.plot_force_plane(resp=2)
        # agg_dat.reload_grav_funcs()
        # #agg_dat.plot_alpha_xyz_dict(resp=2, lambind=35)

        # date = paths['date']
        # hist_prefix = date + '_' + paths['name'] + opt_ext + '_'
        # hist_info = {'date': date, 'prefix': hist_prefix}

        # agg_dat.fit_alpha_xyz_vs_alldim(weight_planar=True, plot=plot_alpha_xyz, \
        #                                 save_hists=save_hists, hist_info=hist_info)
        # alpha_arr = agg_dat.alpha_xyz_best_fit

        # if save:
        #     agg_dat.save(agg_path)




    # if plot_end_result:
    #     alpha_w = np.sum(alpha_arr[:,0:2,:,0]*alpha_arr[:,0:2,:,1]**(-2), axis=1) / \
    #               np.sum(alpha_arr[:,0:2,:,1]**(-2), axis=1)

    #     errs_x = np.zeros_like(alpha_arr[:,0,0,0])
    #     N = 0
    #     for ind in range(2 * np.sum(agg_dat.ginds) - 1):
    #         errs_x += alpha_w[:,ind+1]**2
    #         N += 1

    #     errs_x = np.sqrt(errs_x / N)

    #     #plt.loglog(agg_dat.lambdas, np.abs(alpha_arr[:,0,0,1]), lw=2, label='Gaussian')
    #     #plt.loglog(agg_dat.lambdas, np.abs(alpha_arr[:,0,0,2]), lw=2, label='Cauchy')

    #     plt.title('Result of Planar Fitting', fontsize=16)

    #     plt.loglog(agg_dat.lambdas, np.abs(alpha_w[:,0]), lw=4, \
    #                label='Template basis vector')

    #     plt.loglog(agg_dat.lambdas, errs_x, '--', lw=2, \
    #                label='Quadrature sum of other vectors')       

    #     plt.loglog(gu.limitdata[:,0], gu.limitdata[:,1], '--', label=gu.limitlab, \
    #                linewidth=3, color='r')
    #     plt.loglog(gu.limitdata2[:,0], gu.limitdata2[:,1], '--', label=gu.limitlab2, \
    #                linewidth=3, color='k')
    #     plt.xlabel('Length Scale: $\lambda$ [m]')
    #     plt.ylabel('Strength: |$\\alpha$| [arb]')
    #     plt.legend()
    #     plt.grid()
    #     plt.tight_layout()
    #     plt.show()
        


