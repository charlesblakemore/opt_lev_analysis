import sys, re, os

import dill as pickle

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

import grav_util_3 as gu
import bead_util as bu

import warnings
warnings.filterwarnings("ignore")

### Output path for your alpha-vs-file
output_file = './TEST_FILE.p'

### Value of yukawa lambda to analyze for your output
yuklambda = 1e-3

### Define the location of the simulation output
theory_data_dir = '/home/cblakemore/opt_lev_analysis/gravity_sim/results/7_6um-gbead_1um-unit-cells/'

### Set the directory and the sub-string to select for automtic filename generation.
data_dirs = ['/data/new_trap/20200320/Bead1/Shaking/Shaking378/']
new_trap = True

substr = 'Shaking3'  # for 20200210/.../...384/ and 20200320/.../...378
Nfiles = 10000


### Position of bead relative to the attractor coordinate system
### Z-position gets updated on a file-by-file basis
p0_bead_dict = {'20200320': [392.0, 199.7, 50.0]}


harms = [3,4,5,6,7,8,9,11,12,13,14,15]
n_largest_harms = 5

# axes_to_fit = [0, 1, 2]
axes_to_fit = [2]


### Options for script behavior if you want to do more than save the array
ncore = 1
redo_alpha_fit = False
plot_alpha_xyz = False
plot_basis = False
plot_templates = False

calculate_limit = False

plot_sensitivity = False


#######################################################################
#######################################################################
#######################################################################
#######################################################################



opt_ext = '_harms'
for harm in harms:
    opt_ext += '-' + str(int(harm))
opt_ext += '_first-{:d}'.format(Nfiles)
if len(substr):
    opt_ext += '_{:s}'.format(substr)


for ddir in data_dirs:

    paths = gu.build_paths(ddir, opt_ext, new_trap=new_trap)
    agg_path = paths['agg_path']
    p0_bead = p0_bead_dict[paths['date']]

    ### Open an empty class
    agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms, new_trap=new_trap)

    ### Load the pre-processed AggregateData class
    agg_dat.load(agg_path)
    agg_dat.load_grav_funcs(theory_data_dir)

    # ### A method that helps for grid data, but is unimportant for a single position.
    # ### It still has to be run for the later functions to work
    # agg_dat.bin_rough_stage_positions()

    yuklambdas = agg_dat.gfuncs_class.lambdas
    lambind = np.argmin( np.abs(yuklambdas - yuklambda) )

    ### Shit to access the innards of the class for what you want. It retuns an array of 
    ### the following shape: (Nfile, Nlambda, n_err+1, 3, 2*nharmonics)
    mydict = agg_dat.alpha_xyz_dict
    alpha_xyz_arr = mydict[list(mydict.keys())[0]][agg_dat.ax0vec[0]][agg_dat.ax1vec[0]]

    pickle.dump(alpha_xyz_arr, open(output_file, 'wb'))


    ########################################################
    ### Test plot to demonstrate how to look at the data ###
    ########################################################
    nfiles = alpha_xyz_arr.shape[0]
    inds = np.arange(nfiles)
    err_inds = np.repeat(inds, alpha_xyz_arr.shape[2]-1)

    ### Select z-data, taking the 0th index along the final axis, which
    ### corresponds to the projection along the template vector direction.
    ### Other indices are projections along the other basis vectors
    in_band_alphas = alpha_xyz_arr[:,lambind,0,2,0]
    out_of_band_alphas = alpha_xyz_arr[:,lambind,1:,2,0].flatten()

    ### Plot it!
    plt.plot(inds, in_band_alphas, ls='', ms=6, marker='o', label='In-band')
    plt.plot(err_inds, out_of_band_alphas, ls='', ms=2, marker='o', label='Out-of-band')
    plt.xlabel('File Index')
    plt.ylabel('Alpha')
    plt.legend()
    plt.tight_layout()
    plt.show()




    ########################################################
    #### Other stuff, if you want play around a little  ####
    ########################################################

    ### If you want to redo the fit, this will do it
    if redo_alpha_fit:
        agg_dat.find_alpha_xyz_from_templates(plot=plot_alpha_xyz, plot_basis=plot_basis, \
                                                ncore=ncore, plot_templates=plot_templates, \
                                                n_largest_harms=n_largest_harms)

    ### Get a limit if you want
    if calculate_limit:
        agg_dat.fit_alpha_xyz_onepos_simple(resp=axes_to_fit, verbose=False)

        ### Plot it if you want
        if plot_sensitivity:
            agg_dat.plot_sensitivity()