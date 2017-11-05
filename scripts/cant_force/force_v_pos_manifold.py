import os, fnmatch

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config



dir1 = '/data/20170903/bead1/grav_data/manysep_h0-20um/'
plot_title = ''





###########################################################

def find_all_fnames(dirname, sort=True):
    '''Finals all the filenames matching a particular extension
       type in the directory and its subdirectories .

       INPUTS: dirname, Nfreqx3x3 complex-valued matrix from the
                     transfer_func_util.make_tf_array() function

       OUTPUTS: none, generates new class attribute.'''

    files = []
    for root, dirnames, filenames in os.walk(dir1):
        for filename in fnmatch.filter(filenames, '*' + config.extensions['data']):
            files.append(os.path.join(root, filename))
    if sort:
        # Sort files based on final index
        files.sort(key = bu.find_str)

    return files




def get_force_curve_dictionary(files, ax1='x', ax2='z', fullax1=True, fullax2=True, \
                               ax1val=0, ax2val=0, spacing=1e-6, diag=False):
    '''Loops over a list of file names, loads each file, diagonalizes,
       computes force v position and then closes then discards the 
       raw data to avoid filling memory. Returns the result as a nested
       dictionary with the first level of keys the ax1 positions and the second
       level of keys the ax2 positions

       INPUTS: files, list of files names to extract data
               ax1, first axis in output array
               ax2, second axis in output array
               fullax1, boolean specifying to loop over all values of ax1
               fullax2, boolean specifying to loop over all values of ax2
               ax1val, if fullax1 -> value to keep
               ax2val, if fullax2 -> value to keep

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
    for fil_ind, fil in enumerate(files):
        # Display percent completion
        per = int(100. * float(fil_ind) / float(len(files)) )
        if per > old_per:
            print old_per
            old_per = per

        # Load data
        df = bu.DataFile()
        df.load(fil)

        df.calibrate_stage_position()
        
        ax1pos = df.stage_settings[ax1 + ' DC']
        ax2pos = df.stage_settings[ax2 + ' DC']

        # If subselection is desired, do that now
        if not fullax1:
            dif1 = np.abs(ax1pos - ax1val)
            if dif1 > spacing:
                continue
        if not fullax2:
            dif2 = np.abs(ax2pos - ax2val)
            if dif2 > spacing:
                continue

        if diag:
            df.diagonalize(verbose=False)

        df.get_force_v_pos(verbose=False)

        # Add the current data to the output dictionary
        if ax1pos not in force_curves.keys():
            force_curves[ax1pos] = {}
        if ax2pos not in force_curves[ax1pos].keys():
            # if height and sep not found, adds them to the directory
            force_curves[ax1pos][ax2pos] = [[], [], []]
            if diag:
                diag_force_curves[ax1pos][ax2pos] = [[], [], []]

            for resp in [0,1,2]:
                force_curves[ax1pos][ax2pos][resp] = \
                        [df.binned_data[resp][0], df.binned_data[resp][1]]
                if diag:
                    diag_force_curves[ax1pos][ax2pos][resp] = \
                           [df.diag_binned_data[resp][0], df.diag_binned_data[resp][1]]
        else:
            for resp in [0,1,2]:
                # if this combination of height and sep have already been recorded,
                # this correctly concatenates and sorts data from multiple files
                old_bins = force_curves[ax1pos][ax2pos][resp][0]
                old_dat = force_curves[ax1pos][ax2pos][resp][1]
                new_bins = np.hstack(old_bins, df.binned_data[resp][0])
                new_dat = np.hstack(old_dat, df.binned_data[resp][1])
                
                sort_inds = np.argsort(new_bins)

                force_curves[ax1pos][ax2pos][resp] = \
                            [new_bins[sort_inds], new_dat[sort_inds]]

                if diag:
                    old_diag_bins = diag_force_curves[ax1pos][ax2pos][resp][0]
                    old_diag_dat = diag_force_curves[ax1pos][ax2pos][resp][1]
                    new_diag_bins = np.hstack(old_diag_bins, df.diag_binned_data[resp][0])
                    new_diag_dat = np.hstack(old_diag_dat, df.diag_binned_data[resp][1])

                    diag_sort_inds = np.argsor(new_diag_bins)

                    diag_force_curves[ax1pos][ax2pos][resp] = \
                                [new_diag_bins[diag_sort_inds], new_diag_dat[diag_sort_inds]]

    if diag:
        return force_curves, diag_force_curves
    else:
        return force_curves


fig, axarr = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(6,10), dpi=150)

for chan in [0,1,2]:
    axarr[chan,0].plot(df.binned_data[chan][0], df.binned_data[chan][1]*df.conv_facs[chan])
    axarr[chan,1].plot(df.diag_binned_data[chan][0], df.diag_binned_data[chan][1])

plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0.5)

if plot_title:
    plt.suptitle(plot_title, fontsize=20)
    plt.subplots_adjust(top=0.9)

plt.show()
