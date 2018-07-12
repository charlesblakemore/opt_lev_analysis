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
import grav_util_2 as gu
import calib_util as cal
import transfer_func_util as tf
import configuration as config

import warnings
warnings.filterwarnings("ignore")

theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'

#data_dir = '/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz'
#data_dir = '/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz_elec-term'

data_dir = '/data/20180704/bead1/grav_data/shield'

datafiles = bu.find_all_fnames(data_dir, ext=config.extensions['data'])

load = True
analyze_subset = False
fit_spatial_alpha = True
N = 1000

if analyze_subset:
    datafiles = datafiles[:N]
parts = data_dir.split('/')
agg_path = '/processed_data/aggdat/' + parts[2] + '_' + parts[-1]  + '.agg'
#agg_path = '/processed_data/aggdat/size_test_100.agg'

if load:
    agg_dat = pickle.load(open(agg_path, 'rb'))
    agg_dat.reload_grav_funcs()
    agg_dat.plot_alpha_dict(yuklambda=25.0e-6)

    ## Extract a limit
    agg_dat.fit_alpha_vs_alldim()
    agg_dat.plot_sensitivity()

    agg_dat.save(agg_path)

else:

    ## Load the data
    agg_dat = gu.AggregateData(datafiles, p0_bead=[16,0,20])
    agg_dat.load_grav_funcs(theory_data_dir)

    agg_dat.save(agg_path)


    ## Get height/sep grid
    agg_dat.bin_rough_stage_positions()

    ## Analyze alpha vs height/sep
    agg_dat.find_mean_alpha_vs_position()

    agg_dat.save(agg_path)

    ## Extract a limit
    if fit_spatial_alpha:
        agg_dat.fit_alpha_vs_alldim()

        agg_dat.save(agg_path)

        agg_dat.plot_alpha_dict()
        agg_dat.plot_sensitivity()
