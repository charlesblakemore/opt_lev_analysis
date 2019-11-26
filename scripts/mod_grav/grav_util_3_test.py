import sys, time, itertools

import dill as pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opti
import scipy.linalg as linalg

import bead_util as bu
import grav_util_3 as gu
import calib_util as cal
import transfer_func_util as tf
import configuration as config

import warnings
warnings.filterwarnings("ignore")

theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'

data_dir = '/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz'
#data_dir = '/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz_elec-term'

#data_dir = '/data/20180704/bead1/grav_data/shield'
#data_dir = '/data/20180704/bead1/grav_data/shield_1s_1h'
#data_dir = '/data/20180704/bead1/grav_data/shield2'
#data_dir = '/data/20180704/bead1/grav_data/shield3'
#data_dir = '/data/20180704/bead1/grav_data/shield4'

datafiles = bu.find_all_fnames(data_dir, ext=config.extensions['data'])


#############################
#############################

p0_bead = [19,0,20]
#harms = []
harms = [1,2,3,4,5,6]


load = False
analyze_subset = True
save = True
N = 162

#opt_ext = ''
opt_ext = '_162files'

if analyze_subset:
    datafiles = datafiles[:N]

parts = data_dir.split('/')
if parts[-1] == '':
    agg_path = '/processed_data/aggdat/' + parts[2] + '_' + parts[-2] + opt_ext + '.agg'
else:
    agg_path = '/processed_data/aggdat/' + parts[2] + '_' + parts[-1] + opt_ext + '.agg'


if load:
    agg_dat = pickle.load(open(agg_path, 'rb'))
    agg_dat.reload_grav_funcs()

    agg_dat.bin_rough_stage_positions()

    ## Analyze alpha vs height/sep
    agg_dat.find_mean_alpha_vs_position()

    agg_dat.save(agg_path)

else:

    ## Load the data
    #agg_dat = gu.AggregateData(datafiles, p0_bead=p0_bead, harms=harms, reload_dat=True)
    #agg_dat.load_grav_funcs(theory_data_dir)

    agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms)
    agg_dat.load(agg_path)

    #if save:
    #    agg_dat.save(agg_path)

    ## Get height/sep grid
    #agg_dat.bin_rough_stage_positions()

    #agg_dat.average_resp_by_coordinate()

    ## Analyze alpha vs height/sep
    #agg_dat.find_alpha_xyz_from_templates_avg(plot=True)
    #agg_dat.find_alpha_xyz_from_templates(plot=False)
    
    #if save:
    #    agg_dat.save(agg_path)


    agg_dat.fit_alpha_xyz_vs_alldim()

    agg_dat.save(agg_path)
    print('Saved that new-new')

    '''
    agg_dat.save(agg_path)

    ## Extract a limit
    if fit_spatial_alpha:
        agg_dat.fit_alpha_vs_alldim()
        if save:
            agg_dat.save(agg_path)

        agg_dat.plot_alpha_dict()
        agg_dat.plot_sensitivity()
    '''
