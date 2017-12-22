import os, fnmatch, sys, glob

import dill as pickle

import scipy.interpolate as interp
import scipy.signal as signal

from obspy.signal.detrend import polynomial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from mpl_toolkits.mplot3d import Axes3D

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config


parent_path = '/force_v_pos/20171106_11picopos_2/'
pico_prefix = '20171106_force_v_pos_dic_300Hz-tophat_100mHz-notch_10harm'

manifold_paths = glob.glob(parent_path + pico_prefix + '*')

manifold_paths.sort(key = bu.find_str)

pico_dist = 75    # um, estimated distance traveled between picomotor positions
first_pos = -300  # um, position of the center of the cantilever

detrend = True
order = 1

plot_lim = (-1e-15, 1e-15)

# THIS SCRIPT IS GONNA DO DERPY STITCHING


tot_manifold = {}

# Load all the manifolds
manifolds = []
for ind, path in enumerate(manifold_paths):
    manifolds.append( pickle.load( open(manifold_paths[ind], 'rb') ) )

mean_pos_vec = []
for i in range(len(manifolds)):
    mean_pos_vec.append( first_pos + i * pico_dist )


# Assume all manifolds have same heights and separations
test_man = manifolds[0]
xvec = test_man.keys()
zvec = test_man[xvec[0]].keys()

xvec = np.sort(np.array(xvec))
zvec = np.sort(np.array(zvec))

#print xvec
#print zvec

for xpos in xvec:
    tot_manifold[xpos] = {}

    for zpos in zvec:
        tot_manifold[xpos][zpos] = [[],[],[]]

        fig, axarr = plt.subplots(3,1,sharex=True,sharey=True,figsize=(8,6),dpi=200)

        for resp in [0,1,2]:
            bins = np.array([])
            dat = np.array([])
            errs = np.array([])

            for manind, manifold in enumerate(manifolds):
                #print manind
                #print mean_pos_vec[manind]

                newbins = manifold[xpos][zpos][resp][0]
                newbins = newbins - np.mean(newbins) + mean_pos_vec[manind]
                
                if manind != 0:
                    old_oinds = bins > np.min(newbins)
                    new_oinds = newbins < np.max(bins)

                bins = np.concatenate((bins, newbins))

                newdat = manifold[xpos][zpos][resp][1] #+ 1e-15 * manind

                if manind != 0:
                    offset = np.mean(dat[old_oinds]) - np.mean(newdat[new_oinds])
                    if np.isnan(offset):
                        offset = 0
                    newdat += offset

                if detrend:
                    newdat = polynomial(newdat, order=order, plot=False)
                    #newdat = signal.detrend(newdat)
                dat = np.concatenate((dat, newdat)) 

                newerrs = manifold[xpos][zpos][resp][2]
                errs = np.concatenate((errs, newerrs))

            sort_inds = np.argsort(bins)
            
            bins_sort = bins[sort_inds]
            dat_sort = dat[sort_inds]
            errs_sort = errs[sort_inds]

            axarr[resp].plot(bins_sort, dat_sort, '.', markersize=2)
            axarr[resp].set_ylabel('Force [N]', fontsize=10)
            axarr[resp].set_ylim(plot_lim[0], plot_lim[1])

            tot_manifold[xpos][zpos][resp] = [bins_sort, dat_sort, errs_sort]
        
        axarr[2].set_xlabel('Cantilever Postion [um]', fontsize=10)
        plt.tight_layout()
        plt.show()
                
