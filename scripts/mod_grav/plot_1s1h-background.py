import dill as pickle

import numpy as np
import scipy.special as special

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
             '/data/20180704/bead1/grav_data/shield_1s_1h', \
             #'/data/20180704/bead1/grav_data/shield2', \
             #'/data/20180704/bead1/grav_data/shield3', \
             #'/data/20180704/bead1/grav_data/shield4', \
             #'/data/20180704/no_bead/grav_data/shield', \
             #\
             #'/data/20180808/bead4/grav_data/shield1'
             ]



# data_dirs = ['/data/new_trap/20191204/Bead1/Shaking/Shaking370/']
data_dirs = ['/data/new_trap/20200107/Bead3/Shaking/Shaking380/']
# data_dirs = ['/data/new_trap/20200113/Bead1/Shaking/Shaking377/']
new_trap = True

#substr = ''
substr = 'Shaking13'

load_agg = True
load_alpha_arr = True

Nfiles = 100000

redo_alphafit = False
save = True
plot_end_result = True

plot = False
plot_harms = True
plot_basis = False
plot_alpha_xyz = False

save_hists = False

p0_bead_dict = {'20180625': [19.0,40.0,20.0], \
                '20180704': [18.7,40.0,20.0], \
                '20180808': [18.0,40.0,20.0] \
                }

p0_bead_dict = {#'20191204': [385.0, 200.0, 29.0], \
                '20191204': [-115.0, 200.0, 29.0], \
                # '20200107': [-111, 190.0, 17.0], \
                '20200107': [-111, 190.0, 26.0], \
                # '20200113': [-114.6, 184.5, 16.0], \
                '20200113': [-114.6, 184.5, 10.0], \
                }

# harms = [3,4,5,6]
harms = [6]

lambind = 70

#opt_ext = 'TEST'
# opt_ext = '_harms-3456_deltaz-6um'
opt_ext = '_harms-6_deltaz-6um_first-350'
if len(substr):
    opt_ext += '_{:s}'.format(substr)



def volume_ndim_ellipsoid(axes):
    ndim = len(axes)
    prefac = np.pi**(float(ndim)/2.0) / special.gamma(0.5*float(ndim)+1)
    return prefac * np.prod(axes)




for ddir in data_dirs:
    # Skip the ones I've already calculated
    #if ddir == data_dirs[0]:
    #    continue
    print()

    paths = gu.build_paths(ddir, opt_ext=opt_ext, new_trap=new_trap)
    p0_bead = p0_bead_dict[paths['date']]

    agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms)
    if load_agg:
        agg_dat.load(paths['agg_path'])
        agg_dat.gfuncs_class.reload_grav_funcs()
        bu.make_all_pardirs(paths['alpha_dict_path'])
        agg_dat.save_alpha_dict(paths['alpha_dict_path'])
        #agg_dat.save_alpha_arr(alpha_arr_path)
    elif load_alpha_arr:
        agg_dat.load_alpha_dict(paths['alpha_dict_path'])
        

    if load_agg and plot:
        agg_dat.plot_force_plane(resp=0, fig_ind=1, show=False)
        agg_dat.plot_force_plane(resp=1, fig_ind=2, show=False)
        agg_dat.plot_force_plane(resp=2, fig_ind=3, show=True)

    #print(agg_dat.alpha_xyz_dict[0.0][370.0].keys())#.keys())
    #input()

    keys0 = list(agg_dat.alpha_xyz_dict.keys())
    keys0.sort()
    keys01 = list(agg_dat.alpha_xyz_dict[keys0[0]].keys())
    keys01.sort()
    keys012 = list(agg_dat.alpha_xyz_dict[keys0[0]][keys01[0]].keys())
    keys012.sort()

    # alpha_arr = agg_dat.alpha_xyz_dict[list(agg_dat.alpha_xyz_dict.keys())[0]]\
    #             [agg_dat.ax1vec[0]][agg_dat.ax2vec[0]]
    alpha_arr = agg_dat.alpha_xyz_dict[keys0[0]][keys01[0]][keys012[0]]

    fig1, axarr1 = plt.subplots(3,1,sharex=True,sharey=True,figsize=(8,8))
    #fig2, axarr2 = plt.subplots(3,1,sharex=True)

    for resp in [0,1,2]:
        background = []
        vol = []
        for fileind, filedat in enumerate(alpha_arr):
            background.append(filedat[lambind][0][resp][0])
            vol.append(volume_ndim_ellipsoid( np.abs(filedat[lambind][0][resp][:]) ))
        xvec = np.arange(len(background)) * 10
        axarr1[resp].plot(xvec, background, label='Signal Axis Projection')
        # axarr1[resp].plot(xvec, np.array(vol)**(1.0/float(len(harms))), alpha=0.4, \
        #                   label='$(V_{12D})^{1/12}$')
        plt.figure()
        plt.hist(background, 20)
        print('Axis {:d} : mean {:0.3g}, std/rt(N) {:0.3g}'\
                .format(resp, np.mean(background), \
                        np.std(background) / np.sqrt(len(background))) )
    axarr1[0].set_ylabel('X Projection $[\\alpha]$')
    axarr1[1].set_ylabel('Y Projection $[\\alpha]$')
    axarr1[2].set_ylabel('Z Projection $[\\alpha]$')
    axarr1[0].legend()
    axarr1[2].set_xlabel('Time [s]')
    fig1.tight_layout()
    plt.show()
            
        
