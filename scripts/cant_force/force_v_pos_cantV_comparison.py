import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config



dir1 = '/data/20171106/bead1/response_vs_dcvoltage_unfilt/'
maxfiles = 10000 # Many more than necessary
ax1_lab = 'x'
ax2_lab = 'z'
nbins = 100 # Final number of force v pos bins
lpf = 100   # Hz

plot = True
ax1_toplot = 75
ax2_toplot = 0


###########################################################

def get_force_curve_dictionary(files, ax1='x', ax2='z', ax1val=0, ax2val=0, \
                               diag=False):
    '''Loops over a list of file names, loads each file, diagonalizes,
       computes force v position and then closes then discards the 
       raw data to avoid filling memory. Returns the result as a dictionary 
       with cantilever biases as keys

       INPUTS: files, list of files names to extract data
               ax1, first axis in output array
               ax2, second axis in output array
               ax1val, value to keep (or closest)
               ax2val, value to keep
               diag, boolean specifying whether to diagonalize

       OUTPUTS: outdic, ouput dictionary with the following indexing
                        outdic[ax1pos][ax2pos][resp(0,1,2)][bins(0) or dat(1)]
                        ax1pos and ax2pos are dictionary keys, resp and bins/dat
                        are array indices (native python lists)
                diagoutdic, if diag=True second dictionary with diagonalized data
    '''
    force_curves = {}
    if diag:
        diag_force_curves = {}

    old_per = 0
    print "Percent complete: "
    for fil_ind, fil in enumerate(files):
        # Display percent completion
        per = int(100. * float(fil_ind) / float(len(files)) )
        if per > old_per:
            print old_per,
            sys.stdout.flush()
            old_per = per

        # Load data
        df = bu.DataFile()
        df.load(fil)

        df.calibrate_stage_position()
        
        cantbias = df.electrode_settings['dc_settings'][0]
        ax1pos = df.stage_settings[ax1 + ' DC']
        ax2pos = df.stage_settings[ax2 + ' DC']

        if diag:
            df.diagonalize(maxfreq=lpf)

        df.get_force_v_pos(verbose=False, nbins=nbins)

        if cantbias not in force_curves.keys():
            force_curves[cantbias] = {}
            if diag:
                diag_force_curves[cantbias] = {}

        if ax1pos not in force_curves[cantbias].keys():
            force_curves[cantbias][ax1pos] = {}
            if diag:
                diag_force_curves[cantbias][ax1pos] = {}

        if ax2pos not in force_curves[cantbias][ax1pos].keys():
            force_curves[cantbias][ax1pos][ax2pos] = [[], [], []]
            if diag:
                diag_force_curves[cantbias][ax1pos][ax2pos] = [[], [], []]

            for resp in [0,1,2]:
                force_curves[cantbias][ax1pos][ax2pos][resp] = \
                    [df.binned_data[resp][0], \
                     df.binned_data[resp][1] * df.conv_facs[resp]]

                if diag:
                    diag_force_curves[cantbias][ax1pos][ax2pos][resp] = \
                        [df.diag_binned_data[resp][0], \
                         df.diag_binned_data[resp][1]]

        else:
            for resp in [0,1,2]:
                old_bins = force_curves[cantbias][ax1pos][ax2pos][resp][0]
                old_dat = force_curves[cantbias][ax1pos][ax2pos][resp][1]
                new_bins = np.hstack((old_bins, df.binned_data[resp][0]))
                new_dat = np.hstack((old_dat, df.binned_data[resp][1] * df.conv_facs[resp]))
                
                sort_inds = np.argsort(new_bins)

                force_curves[cantbias][ax1pos][ax2pos][resp] = \
                            [new_bins[sort_inds], new_dat[sort_inds]]

                if diag:
                    old_diag_bins = diag_force_curves[cantbias][ax1pos][ax2pos][resp][0]
                    old_diag_dat = diag_force_curves[cantbias][ax1pos][ax2pos][resp][1]
                    new_diag_bins = np.hstack((old_diag_bins, df.diag_binned_data[resp][0]))
                    new_diag_dat = np.hstack((old_diag_dat, df.diag_binned_data[resp][1]))

                    diag_sort_inds = np.argsort(new_diag_bins)

                    diag_force_curves[cantbias][ax1pos][ax2pos][resp] = \
                                [new_diag_bins[diag_sort_inds], new_diag_dat[diag_sort_inds]]


    cantV_keys = force_curves.keys()
    ax1_keys = force_curves[cantV_keys[0]].keys()
    ax2_keys = force_curves[cantV_keys[0]][ax1_keys[0]].keys()

    ax1_k = ax1_keys[ np.argmin( np.abs( ax1val - np.array(ax1_keys) ) ) ]
    ax2_k = ax2_keys[ np.argmin( np.abs( ax2val - np.array(ax2_keys) ) ) ]
