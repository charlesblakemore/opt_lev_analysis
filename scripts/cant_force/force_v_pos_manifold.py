import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from mpl_toolkits.mplot3d import Axes3D

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config



dir1 = '/data/20170903/bead1/grav_data/manysep_h0-20um/'
#dir1 = '/data/20171106/bead1/grav_data_1/'

maxfiles = 10000 # Many more than necessary
ax1_lab = 'x'
ax2_lab = 'z'
nbins = 100

save_path1 = '/force_v_pos/20170903_force_v_pos_dic.p'
save_path2 = '/force_v_pos/20170903_diagforce_v_pos_dic.p'

save = True #True
load = False #True
load_path = '/force_v_pos/20170903_force_v_pos_dic.p'

plot = True
resp_to_plot = 0
ax2_toplot = 0

plot_title = ''

testind = 0
test_posvec = [[],[],[]]
test_arr = [[],[],[]]
diag_test_posvec = [[],[],[]]
diag_test_arr = [[],[],[]]

###########################################################



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
               ax1val, if not fullax1 -> value to keep
               ax2val, if not fullax2 -> value to keep
               spacing, spacing around ax1val or ax2val to keep
               diag, boolean specifying whether to diagonalize

       OUTPUTS: outdic, ouput dictionary with the following indexing
                        outdic[ax1pos][ax2pos][resp(0,1,2)][bins(0) or dat(1)]
                        ax1pos and ax2pos are dictionary keys, resp and bins/dat
                        are array indices (native python lists)
                diagoutdic, if diag=True second dictionary with diagonalized data
    '''

    if len(files) == 0:
        print "No Files Found!!"
        return

    force_curves = {}
    if diag:
        diag_force_curves = {}
    old_per = 0
    print "Processing %i files" % len(files)
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
            df.diagonalize(maxfreq=100)

        df.get_force_v_pos(verbose=False, nbins=nbins)

        # Add the current data to the output dictionary
        if ax1pos not in force_curves.keys():
            force_curves[ax1pos] = {}
            if diag:
                diag_force_curves[ax1pos] = {}
        if ax2pos not in force_curves[ax1pos].keys():
            # if height and sep not found, adds them to the directory
            force_curves[ax1pos][ax2pos] = [[], [], []]
            if diag:
                diag_force_curves[ax1pos][ax2pos] = [[], [], []]

            for resp in [0,1,2]:
                force_curves[ax1pos][ax2pos][resp] = \
                        [df.binned_data[resp][0], \
                         df.binned_data[resp][1] * df.conv_facs[resp]]
                if diag:
                    diag_force_curves[ax1pos][ax2pos][resp] = \
                           [df.diag_binned_data[resp][0], \
                            df.diag_binned_data[resp][1]]
        else:
            for resp in [0,1,2]:
                # if this combination of height and sep have already been recorded,
                # this correctly concatenates and sorts data from multiple files
                old_bins = force_curves[ax1pos][ax2pos][resp][0]
                old_dat = force_curves[ax1pos][ax2pos][resp][1]
                new_bins = np.hstack((old_bins, df.binned_data[resp][0]))
                new_dat = np.hstack((old_dat, df.binned_data[resp][1] * df.conv_facs[resp]))
                
                sort_inds = np.argsort(new_bins)

                force_curves[ax1pos][ax2pos][resp] = \
                            [new_bins[sort_inds], new_dat[sort_inds]]

                if diag:
                    old_diag_bins = diag_force_curves[ax1pos][ax2pos][resp][0]
                    old_diag_dat = diag_force_curves[ax1pos][ax2pos][resp][1]
                    new_diag_bins = np.hstack((old_diag_bins, df.diag_binned_data[resp][0]))
                    new_diag_dat = np.hstack((old_diag_dat, df.diag_binned_data[resp][1]))

                    diag_sort_inds = np.argsort(new_diag_bins)

                    diag_force_curves[ax1pos][ax2pos][resp] = \
                                [new_diag_bins[diag_sort_inds], new_diag_dat[diag_sort_inds]]

    ax1_keys = force_curves.keys()
    ax2_keys = force_curves[ax1_keys[0]].keys()

    print 
    print 'Averaging files and building standard deviations'
    sys.stdout.flush()

    #max_ax1 = np.max( ax1_keys )
    test_ax1 = 35
    max_ax1 = ax1_keys[np.argmin( np.abs( test_ax1 - np.array(ax1_keys)) )]
    ax2pos = ax2_keys[np.argmin( np.abs(ax2_toplot - np.array(ax2_keys)) )]

    for ax1_k in ax1_keys:
        for ax2_k in ax2_keys:
            for resp in [0,1,2]:

                old_bins = force_curves[ax1_k][ax2_k][resp][0]
                old_dat = force_curves[ax1_k][ax2_k][resp][1]

                if ax1_k == max_ax1:
                    if ax2_k == ax2pos:
                        test_posvec[resp] = old_bins
                        test_arr[resp] = old_dat

                dat_func = interp.interp1d(old_bins, old_dat, kind='cubic')

                new_bins = np.linspace(np.min(old_bins)+1e-9, np.max(old_bins)-1e-9, nbins)
                new_dat = dat_func(new_bins)
                new_errs = np.zeros_like(new_dat)

                bin_sp = new_bins[1] - new_bins[0]
                for binind, binval in enumerate(new_bins):
                    inds = np.abs( old_bins - binval ) < bin_sp
                    new_errs[binind] = np.std( old_dat[inds] )

                force_curves[ax1_k][ax2_k][resp] = [new_bins, new_dat, new_errs]

                if diag:
                    old_diag_bins = diag_force_curves[ax1_k][ax2_k][resp][0]
                    old_diag_dat = diag_force_curves[ax1_k][ax2_k][resp][1]

                    if ax1_k == max_ax1:
                        if ax2_k == ax2pos:
                            diag_test_posvec[resp] = old_diag_bins
                            diag_test_arr[resp] = old_diag_dat

                    diag_dat_func = interp.interp1d(old_diag_bins, old_diag_dat, kind='cubic')

                    new_diag_bins = np.linspace(np.min(old_diag_bins)+1e-9, \
                                                np.max(old_diag_bins)-1e-9, nbins)
                    new_diag_dat = dat_func(new_diag_bins)
                    new_diag_errs = np.zeros_like(new_diag_dat)

                    diag_bin_sp = new_diag_bins[1] - new_diag_bins[0]
                    for binind, binval in enumerate(new_diag_bins):
                        diaginds = np.abs( old_diag_bins - binval ) < diag_bin_sp
                        new_diag_errs[binind] = np.std( old_diag_dat[diaginds] )

                    diag_force_curves[ax1_k][ax2_k][resp] = [new_diag_bins, new_diag_dat, new_diag_errs]
                    
                    
    

    if diag:
        return force_curves, diag_force_curves
    else:
        return force_curves




if not load:
    files = bu.find_all_fnames(dir1)
    files = files[:maxfiles]

    if len(files) == 0:
        print 'No Files Found!!'
        quit()
    else:
        force_dic, diag_force_dic = \
            get_force_curve_dictionary(files, ax1=ax1_lab, ax2=ax2_lab, spacing=1e-6, diag=True)

        if save:
            pickle.dump(force_dic, open(save_path1, 'wb') )
            pickle.dump(diag_force_dic, open(save_path2, 'wb') )


fig, axarr = plt.subplots(3,1,sharex=True,sharey=True)
for resp in [0,1,2]:
    axarr[resp].plot(test_posvec[resp], test_arr[resp])
plt.show()


#np.save('./test_posvec2.npy', np.array(test_posvec))
#np.save('./test_arr2.npy', np.array(test_arr))
#np.save('./diag_test_posvec2.npy', np.array(diag_test_posvec))
#np.save('./diag_test_arr2.npy', np.array(diag_test_arr))



if load:
    force_dic = pickle.load( open(load_path, 'rb') )

if plot:

    ax1 = force_dic.keys()
    ax2 = force_dic[ax1[0]].keys()

    sample_yvec = force_dic[ax1[0]][ax2[0]][resp_to_plot][0]
    numbins = len(sample_yvec)

    ax1 = np.sort( np.array(ax1) )
    ax2 = np.sort( np.array(ax2) )

    ax2pos = ax2[np.argmin( np.abs(ax2_toplot - ax2) )]

    fgrid = np.zeros((len(ax1), numbins))
    for ax11_ind, ax11 in enumerate(ax1):
        fgrid[ax11_ind,:] = force_dic[ax11][ax2pos][resp_to_plot][1]
        

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax1 = (np.max(ax1) - ax1) + 10

    xgrid, ygrid = np.meshgrid(sample_yvec, ax1)

    #ax.plot_wireframe(sepgrid, xgrid, out)
    surf = ax.plot_surface(xgrid, ygrid, fgrid, \
                       rcount=20, ccount=100,\
                       cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surf, aspect=20)

    plt.show()
