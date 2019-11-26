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

load_agg = False
load_alpha_arr = True

Nfiles = 100000

redo_alphafit = False
save = True
plot_end_result = True

plot = False
plot_harms = False
plot_basis = False
plot_alpha_xyz = False

save_hists = False

p0_bead_dict = {'20180625': [19.0,40.0,20.0], \
                '20180704': [18.7,40.0,20.0], \
                '20180808': [18.0,40.0,20.0] \
                }

harms = [1,2,3,4,5,6]

#opt_ext = 'TEST'
opt_ext = '_6harm-full'



def volume_ndim_ellipsoid(axes):
    ndim = len(axes)
    prefac = np.pi**(float(ndim)/2.0) / special.gamma(0.5*float(ndim)+1)
    return prefac * np.prod(axes)




for ddir in data_dirs:
    # Skip the ones I've already calculated
    #if ddir == data_dirs[0]:
    #    continue
    print()

    paths = gu.build_paths(ddir, opt_ext=opt_ext)
    p0_bead = p0_bead_dict[paths['date']]

    agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms)
    if load_agg:
        agg_dat.load(paths['agg_path'])
        agg_dat.reload_grav_funcs()
        agg_dat.save_alpha_dict(paths['alpha_dict_path'])
        #agg_dat.save_alpha_arr(alpha_arr_path)
    elif load_alpha_arr:
        agg_dat.load_alpha_dict(paths['alpha_dict_path'])
        

    if load_agg and plot:
        agg_dat.plot_force_plane(resp=0, fig_ind=1, show=False)
        agg_dat.plot_force_plane(resp=1, fig_ind=2, show=False)
        agg_dat.plot_force_plane(resp=2, fig_ind=3, show=True)

    alpha_arr = agg_dat.alpha_xyz_dict[list(agg_dat.alpha_xyz_dict.keys())[0]]\
                [agg_dat.ax1vec[0]][agg_dat.ax2vec[0]]

    fig1, axarr1 = plt.subplots(2,1,sharex=True,sharey=True)
    #fig2, axarr2 = plt.subplots(3,1,sharex=True)

    for resp in [0,1]:
        background = []
        vol = []
        for fileind, filedat in enumerate(alpha_arr):
            background.append(filedat[70][0][resp][0])
            vol.append(volume_ndim_ellipsoid( np.abs(filedat[70][0][resp][:]) ))
        axarr1[resp].plot(background, label='Signal Axis Projection')
        axarr1[resp].plot(np.array(vol)**(1.0/11.0), alpha=0.4, \
                          label='$(V_{12D})^{1/12}$')
    axarr1[0].set_ylabel('X Projection [alpha]')
    axarr1[1].set_ylabel('Y Projection [alpha]')
    axarr1[0].legend()
    axarr1[1].set_xlabel('Time [m]')
    plt.show()
            
        
