import time, sys, os
import dill as pickle

import numpy as np
import scipy.constants as constants
import scipy.interpolate as interp
import scipy.optimize as opti

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import grav_util_3 as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")



theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'

patches_base_path = '/processed_data/comsol_data/patch_potentials/'
patches_name = 'patch_pot_2um_1Vrms_50um-deep-patches'


# Include some legacy grav data to compare to later
data_dirs = [#'/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz', \
             #\
             #'/data/20180704/bead1/grav_data/shield', \
             #\
             #'/data/20180808/bead4/grav_data/shield1' \
             #\
             #'/data/20180827/bead2/500e_data/dipole_v_height_ac', \
             '/data/20180827/bead2/500e_data/dipole_v_height_no_v_ysweep'
             ]

load_files = False

p0_bead_dict = {'20180625': [19.0, 40.0, 20.0], \
                '20180704': [18.7, 40.0, 20.0], \
                '20180808': [18.0, 40.0, 20.0], \
                '20180827': [13.0 ,40.0, 28.6]
                }

opt_ext = '_electrostatics'

harms = [1]
#harms = [1,2,3,4,5]

charge = 430 * constants.elementary_charge * (-1.0)
plot_field_test = True

############################################################
xx = np.load(open(patches_base_path + patches_name + '.xx', 'rb'))
yy = np.load(open(patches_base_path + patches_name + '.yy', 'rb'))
zz = np.load(open(patches_base_path + patches_name + '.zz', 'rb'))

field = np.load(open(patches_base_path + patches_name + '.field', 'rb'))
potential = np.load(open(patches_base_path + patches_name + '.potential', 'rb')) 


pot_func = interp.RegularGridInterpolator((xx, yy, zz), potential, \
                                          bounds_error=False, fill_value=None)

field_func = []
for resp in 0,1,2:
    field_func.append( interp.RegularGridInterpolator((xx, yy, zz), field[resp], \
                                          bounds_error=False, fill_value=None) )


if plot_field_test:
    posvec = np.linspace(-20e-6, 20e-6, 101)
    ones = np.ones_like(posvec)
    xval = 20.0e-6
    yval = 0.0e-6
    zval = 0.0e-6
    eval_pts = np.stack((xval*ones, posvec, zval*ones), axis=-1)
    eval_pts = np.stack((xval*ones, yval*ones, posvec), axis=-1)

    ann_str = 'Sep: %0.2f um, Height: %0.2f um' % (xval*1e6, zval*1e6)

    
    plt.figure()
    plt.plot(posvec*1e6, pot_func(eval_pts))

    plt.figure(figsize=(7,5))
    #plt.title(name)
    plt.plot(posvec*1e6, field_func[0](eval_pts)*charge, label='fx')
    plt.plot(posvec*1e6, field_func[1](eval_pts)*charge, label='fy')
    plt.plot(posvec*1e6, field_func[2](eval_pts)*charge, label='fz')
    plt.legend()
    plt.xlabel('Displacement Along Attractor [um]')
    plt.ylabel('Force on 500e$^-$ [N]')
    plt.annotate(ann_str, xy=(0.2, 0.9), xycoords='axes fraction')
    plt.tight_layout()
    plt.grid()

    plt.show()
    





for ddir in data_dirs:

    paths = gu.build_paths(ddir, opt_ext=opt_ext)

    datafiles = bu.find_all_fnames(ddir)
    p0_bead = p0_bead_dict[paths['date']]

    if load_files:
        agg_dat = gu.AggregateData(datafiles, p0_bead=p0_bead, harms=harms, \
                                   elec_drive=False, elec_ind=0, plot_harm_extraction=False)        

        agg_dat.save(paths['agg_path'])

    else:
        agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms)
        agg_dat.load(paths['agg_path'])

    agg_dat.bin_rough_stage_positions(ax_disc=1.0)
    #agg_dat.plot_force_plane(resp=0, fig_ind=1, show=False)
    #agg_dat.plot_force_plane(resp=1, fig_ind=2, show=False)
    #agg_dat.plot_force_plane(resp=2, fig_ind=3, show=False)

    force_plane_dict = agg_dat.get_vector_force_plane(resp=(0,2), fig_ind=1, \
                                                      plot=True, show=True, \
                                                      sign=[-1.0, -1.0, -1.0])

    pos_dict = agg_dat.make_sep_height_arrs()
    seps = pos_dict['seps']
    heights = pos_dict['heights']
    seps_g, heights_g = np.meshgrid(seps, heights, indexing='ij')




    def F_comsol_func(sep_off, ypos, height_off, charge, resp=0):
        interp_mesh = np.array(np.meshgrid((seps+sep_off)*1e-6, [ypos*1e-6], 
                                           (heights+height_off)*1e-6, indexing='ij'))
        interp_points = np.rollaxis(interp_mesh, 0, 4)
        interp_points = interp_points.reshape((interp_mesh.size // 3, 3))

        res = interp.interpn((xx, yy, zz), field[resp], interp_points, \
                                       bounds_error=False, fill_value=None)

        shaped = np.reshape(res, (len(seps), len(heights)))

        #if resp==2:
        #    charge = -1.0 * charge

        return shaped * charge * constants.elementary_charge
    



    soln = [0.0, 0.0, 0.0, -331.0]


    F_comsol = [np.zeros((len(seps), len(heights))), \
                np.zeros((len(seps), len(heights))), \
                np.zeros((len(seps), len(heights)))]

    for resp in [0,1,2]:
        F_comsol[resp] = F_comsol_func(*soln, resp=resp)



    keyscale = 1.0e-14
    scale_pow = int(np.log10(keyscale))

    scale = keyscale * 4


    fig = plt.figure(4)
    ax = fig.add_subplot(111)

    fig2 = plt.figure(5)
    ax2 = fig2.add_subplot(111)

    qdat = ax.quiver(seps_g+soln[0], heights_g+soln[2], force_plane_dict[0], force_plane_dict[2], \
                     color='k', pivot='mid', label='Data', scale=scale)
    qfit = ax2.quiver(seps_g+soln[0], heights_g+soln[2], F_comsol[0]*0.2, F_comsol[2]*0.2, \
                   color='r', pivot='mid', label='Fit', scale=scale)
    
    ax.quiverkey(qdat, X=0.3, Y=1.05, U=keyscale, \
                 label='$10^{%i}~$N Force' % scale_pow, labelpos='N')
    ax2.quiverkey(qfit, X=0.7, Y=1.05, U=keyscale, \
                 label='$10^{%i}~$N Force' % scale_pow, labelpos='N')
    ax.legend(loc=1)
    ax.set_xlabel('Separation [um]   |   Fx')
    ax.set_ylabel('Height [um]   |   Fz')

    
    ax2.legend(loc=1)
    ax2.set_xlabel('Separation [um]   |   Fx')
    ax2.set_ylabel('Height [um]   |   Fz')

    plt.show()


   
