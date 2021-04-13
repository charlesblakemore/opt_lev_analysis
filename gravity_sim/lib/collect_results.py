import sys, copy, os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import dill as pickle

import bead_util as bu


parent = str( Path(os.path.abspath(__file__)).parents[1] )

raw_path = os.path.join(parent, 'raw_results')
out_path = os.path.join(parent, 'results')

# out_subdir = '4_7um-bead_1um-unit-cells/'
# out_subdir = '4_6um-gbead_1um-unit-cells/'
# out_subdir = '4_6um-gbead_1um-unit-cells_close/'
out_subdir = '7_6um-gbead_1um-unit-cells_z8um-attractor_master/'
# out_subdir = '5um-gbead_1um-unit-cells_master/'
out_path = os.path.join(out_path, out_subdir)

### HAVE TO EDIT THIS FUNCTION TO PARSE SIMULATION OUTPUT
### THAT HAS MULTIPLE BEAD RADII. TRUE VALUE MEANS IT WILL
### BE INCLUDED
def rbead_cond(rbead):
    if rbead > 5.0e-6:
        return False 
    elif rbead > 3.0e-6:
        return True
    else:
        return True

test_filename = os.path.join(out_path, 'test.p')
bu.make_all_pardirs(test_filename)

raw_filenames, _ = bu.find_all_fnames(raw_path, ext='.p', skip_subdirectories=True)

### Loop over all the simulation outut files and extract the 
### simulation parameters used in that file (rbead, sep, height, etc)
seps = []
heights = []
posvec = []
nfiles = len(raw_filenames)
for fil_ind, fil in enumerate(raw_filenames):
    ### Display percent completion
    bu.progress_bar(fil_ind, nfiles, suffix='finding seps/heights')

    sim_out = pickle.load( open(fil, 'rb') )
    keys = list(sim_out.keys())

    ### Avoids the dictionary key that's a string
    for key in keys:
        if type(key) == str:
            continue
        else:
            rbead_key = key

    if not rbead_cond(rbead_key):
        continue
    else:
        rbead = rbead_key

    ### Should probably check if there is more than one key
    cseps = list(sim_out[rbead].keys())
    sep = cseps[0]
    cheights = list(sim_out[rbead][sep].keys())
    height = cheights[0]

    ### Define the arrays that are consant for all simulation 
    ### parameters, i.e. the yukawa lambdas and bead positions
    if not len(posvec):
        posvec = sim_out['posvec']
        lambdas = list(sim_out[rbead][sep][height].keys())
        attractor_params = sim_out['attractor_params']
        rhobead = sim_out['rhobead']
    else:
        assert np.sum(posvec - sim_out['posvec']) == 0.0

    seps.append(sep)
    heights.append(height)



### Select unique values of simulation parameters and construct
### sorted arrays of those values
lambdas = np.sort(np.array(lambdas))

seps = np.sort(np.unique(seps))
heights = np.sort(np.unique(heights))

grid_check = np.zeros((len(seps), len(heights)))

### Build up the 3D array of Newtonian and Yukawa-modified forces
### for each of the positions simulated
Goutarr = np.zeros((len(seps), len(posvec), len(heights), 3))
yukoutarr = np.zeros((len(lambdas), len(seps), len(posvec), len(heights), 3))

for fil_ind, fil in enumerate(raw_filenames):

    bu.progress_bar(fil_ind, nfiles, suffix='collecting sim data')

    sim_out = pickle.load( open(fil, 'rb') )
    keys = list(sim_out.keys())

    ### Avoids the dictionary key that's a string
    for key in keys:
        if type(key) == str:
            continue
        else:
            rbead_key = key

    if not rbead_cond(rbead_key):
        continue
    else:
        rbead = rbead_key

    cseps = list(sim_out[rbead].keys())
    sep = cseps[0]
    cheights = list(sim_out[rbead][sep].keys())
    height = cheights[0]

    dat = sim_out[rbead][sep][height]

    sepind = np.argmin( np.abs(seps - sep) )
    heightind = np.argmin( np.abs(heights - height) )
    grid_check[sepind, heightind] += 1.0

    for ind in [0,1,2]:
        Goutarr[sepind,:,heightind,ind] = dat[lambdas[0]][ind]
        for lambind, lamb in enumerate(lambdas):
            yukoutarr[lambind,sepind,:,heightind,ind] = dat[lamb][ind+3]

print(rbead)
print("Done!")
print()

missing_values = np.sum(grid_check != 1.0)
if missing_values:
    sep_inds, height_inds = np.where(grid_check != 1.0)
    for sep_ind, height_ind in zip(sep_inds, height_inds):
        sep = seps[sep_ind]
        height = heights[height_ind]
        print('Missing simdata for: RBEAD = {:0.3g}, SEP = {:0.3g}, HEIGHT = {:0.3g}'\
                .format(rbead, sep, height))
    print("Saving sparse data anyway. You will probably have to fix shit...")

else:
    print("Saving all that good, good data")

try:
    pickle.dump(attractor_params, open(os.path.join(out_path, 'attractor_params.p'), 'wb'))
    np.save(os.path.join(out_path, 'rbead_rhobead.npy'), [rbead, rhobead])
    # np.save(os.path.join(out_path, 'rbead.npy'), [rbead])
    np.save(os.path.join(out_path, 'lambdas.npy'), lambdas)
    np.save(os.path.join(out_path, 'yukdata.npy'), yukoutarr)
    np.save(os.path.join(out_path, 'Gravdata.npy'), Goutarr)
    np.save(os.path.join(out_path, 'xpos.npy'), seps + rbead)
    np.save(os.path.join(out_path, 'ypos.npy'), posvec)
    np.save(os.path.join(out_path, 'zpos.npy'), heights)

except Exception:
    print("Couldn't save the data.")



