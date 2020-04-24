import sys, copy, os

import numpy as np
import matplotlib.pyplot as plt

import dill as pickle

import bead_util as bu


raw_path = os.path.abspath('../raw_results/')
out_path = os.path.abspath('../results/')

out_subdir = '7_6um-gbead_1um-unit-cells'
out_path = os.path.join(out_path, out_subdir)

### HAVE TO EDIT THIS FUNCTION TO PARSE SIMULATION OUTPUT
### THAT HAS MULTIPLE BEAD RADII
def rbead_cond(rbead):
    if rbead < 2.5e-6:
        return False 
    else:
        return True

test_filename = os.path.join(out_path, 'test.p')
bu.make_all_pardirs(test_filename)

raw_filenames, _ = bu.find_all_fnames(raw_path, ext='.p')

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
            rbead = key
            break
    if not rbead_cond(rbead):
        continue

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
    else:
        assert np.sum(posvec - sim_out['posvec']) == 0.0

    seps.append(sep)
    heights.append(height)



### Select unique values of simulation parameters and construct
### sorted arrays of those values
lambdas = np.array(lambdas)

seps = np.unique(seps)
heights = np.unique(heights)

lambdas = np.sort(lambdas)
seps = np.sort(seps)
heights = np.sort(heights)


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
            rbead = key
            break
    if not rbead_cond(rbead):
        continue

    cseps = list(sim_out[rbead].keys())
    sep = cseps[0]
    cheights = list(sim_out[rbead][sep].keys())
    height = cheights[0]

    dat = sim_out[rbead][sep][height]

    sepind = np.argmin( np.abs(seps - sep) )
    heightind = np.argmin( np.abs(heights - height) )

    for ind in [0,1,2]:
        Goutarr[sepind,:,heightind,ind] = dat[lambdas[0]][ind]
        for lambind, lamb in enumerate(lambdas):
            yukoutarr[lambind,sepind,:,heightind,ind] = dat[lamb][ind+3]

print("Done!")
print("Saving all that good, good data")

np.save(os.path.join(out_path, 'rbead.npy'), [rbead])
np.save(os.path.join(out_path, 'lambdas.npy'), lambdas)
np.save(os.path.join(out_path, 'yukdata.npy'), yukoutarr)
np.save(os.path.join(out_path, 'Gravdata.npy'), Goutarr)
np.save(os.path.join(out_path, 'xpos.npy'), seps)
np.save(os.path.join(out_path, 'ypos.npy'), posvec)
np.save(os.path.join(out_path, 'zpos.npy'), heights)



