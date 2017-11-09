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

dir1 = '/data/20171106/bead1/dipole_v_height_10Vfilt_2/'
maxfiles = 10000 # Many more than necessary
ax1_lab = 'z'
nbins = 20

plot_title = ''


###########################################################








def get_force_curve_dictionary(files, cantind=0, ax1='z', fullax1=True, \
                               ax1val=0,  spacing=1e-6, diag=False):
    '''Loops over a list of file names, loads each file, diagonalizes,
       computes force v position and then closes then discards the 
       raw data to avoid filling memory. Returns the result as a nested
       dictionary with the first level of keys the cantilever biases and
       the second level of keys the height

       INPUTS: files, list of files names to extract data
               cantind, cantilever electrode index
               ax1, axis with different DC positions, usually the height
               fullax1, boolean specifying to loop over all values of ax1
               ax1val, if not fullax1 -> value to keep

       OUTPUTS: outdic, ouput dictionary with the following indexing
                        outdic[cantbias][ax1pos][resp(0,1,2)][bins(0) or dat(1)]
                        cantbias and ax2pos are dictionary keys, resp and bins/dat
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
            print old_per,
            sys.stdout.flush()
            old_per = per

        # Load data
        df = bu.DataFile()
        df.load(fil)

        df.calibrate_stage_position()
    
        cantbias = df.electrode_settings['dc_settings'][0]
        ax1pos = df.stage_settings[ax1 + ' DC']

        # If subselection is desired, do that now
        if not fullax1:
            dif1 = np.abs(ax1pos - ax1val)
            if dif1 > spacing:
                continue

        if diag:
            df.diagonalize(date='20171106', maxfreq=100)

        df.get_force_v_pos(verbose=False, nbins=nbins)

        # Add the current data to the output dictionary
        if cantbias not in force_curves.keys():
            force_curves[cantbias] = {}
            if diag:
                diag_force_curves[cantbias] = {}
        if ax1pos not in force_curves[cantbias].keys():
            # if height and sep not found, adds them to the directory
            force_curves[cantbias][ax1pos] = [[], [], []]
            if diag:
                diag_force_curves[cantbias][ax1pos] = [[], [], []]

            for resp in [0,1,2]:
                force_curves[cantbias][ax1pos][resp] = \
                        [df.binned_data[resp][0], \
                         df.binned_data[resp][1] * df.conv_facs[resp]]
                if diag:
                    diag_force_curves[cantbias][ax1pos][resp] = \
                           [df.diag_binned_data[resp][0], \
                            df.diag_binned_data[resp][1]]
        else:
            for resp in [0,1,2]:
                # if this combination of height and sep have already been recorded,
                # this correctly concatenates and sorts data from multiple files
                old_bins = force_curves[cantbias][ax1pos][resp][0]
                old_dat = force_curves[cantbias][ax1pos][resp][1]
                new_bins = np.hstack((old_bins, df.binned_data[resp][0]))
                new_dat = np.hstack((old_dat, df.binned_data[resp][1] * df.conv_facs[resp]))
                
                sort_inds = np.argsort(new_bins)

                force_curves[cantbias][ax1pos][resp] = \
                            [new_bins[sort_inds], new_dat[sort_inds]]

                if diag:
                    old_diag_bins = diag_force_curves[cantbias][ax1pos][resp][0]
                    old_diag_dat = diag_force_curves[cantbias][ax1pos][resp][1]
                    new_diag_bins = np.hstack((old_diag_bins, df.diag_binned_data[resp][0]))
                    new_diag_dat = np.hstack((old_diag_dat, df.diag_binned_data[resp][1]))

                    diag_sort_inds = np.argsort(new_diag_bins)

                    diag_force_curves[cantbias][ax1pos][resp] = \
                                [new_diag_bins[diag_sort_inds], new_diag_dat[diag_sort_inds]]

    cantV_keys = force_curves.keys()
    ax1_keys = force_curves[cantV_keys[0]].keys()

    print 
    print 'Averaging files and building standard deviations'
    sys.stdout.flush()

    for cantV_k in cantV_keys:
        for ax1_k in ax1_keys:
            for resp in [0,1,2]:

                old_bins = force_curves[cantV_k][ax1_k][resp][0]
                old_dat = force_curves[cantV_k][ax1_k][resp][1]

                dat_func = interp.interp1d(old_bins, old_dat, kind='cubic')

                new_bins = np.linspace(np.min(old_bins)+1e-9, np.max(old_bins)-1e-9, nbins)
                new_dat = dat_func(new_bins)
                new_errs = np.zeros_like(new_dat)

                bin_sp = new_bins[1] - new_bins[0]
                for binind, binval in enumerate(new_bins):
                    inds = np.abs( old_bins - binval ) < bin_sp
                    new_errs[binind] = np.std( old_dat[inds] )

                force_curves[cantV_k][ax1_k][resp] = [new_bins, new_dat, new_errs]

                if diag:
                    old_diag_bins = diag_force_curves[cantV_k][ax1_k][resp][0]
                    old_diag_dat = diag_force_curves[cantV_k][ax1_k][resp][1]
                    diag_dat_func = interp.interp1d(old_diag_bins, old_diag_dat, kind='cubic')

                    new_diag_bins = np.linspace(np.min(old_diag_bins)+1e-9, \
                                                np.max(old_diag_bins)-1e-9, nbins)
                    new_diag_dat = dat_func(new_diag_bins)
                    new_diag_errs = np.zeros_like(new_diag_dat)

                    diag_bin_sp = new_diag_bins[1] - new_diag_bins[0]
                    for binind, binval in enumerate(new_diag_bins):
                        diaginds = np.abs( old_diag_bins - binval ) < diag_bin_sp
                        new_diag_errs[binind] = np.std( old_diag_dat[diaginds] )

                    diag_force_curves[cantV_k][ax1_k][resp] = \
                                        [new_diag_bins, new_diag_dat, new_diag_errs]
                    
                    
    

    if diag:
        return force_curves, diag_force_curves
    else:
        return force_curves



datafiles = bu.find_all_fnames(dir1, ext=config.extensions['data'])

force_dic, diag_force_dic = \
        get_force_curve_dictionary(datafiles, ax1=ax1_lab, diag=True)

cantV = force_dic.keys()
cantV.sort()

figs = []
axarrs = []

for biasind, bias in enumerate(cantV):
    fig, axarr = plt.subplots(3,2,sharex=True,sharey=True,figsize=(6,8),dpi=150)

    figs.append(fig)
    axarrs.append(axarr)

    stage_settings = force_dic[bias].keys()
    stage_settings.sort()

    for posind, pos in enumerate(stage_settings):
        color = 'C' + str(posind)
        lab = str(pos) + ' um'

        for resp in [0,1,2]:
            bins = force_dic[bias][pos][resp][0]
            dat = force_dic[bias][pos][resp][1]
            errs = force_dic[bias][pos][resp][2]

            diag_bins = diag_force_dic[bias][pos][resp][0]
            diag_dat = diag_force_dic[bias][pos][resp][1]
            diag_errs = diag_force_dic[bias][pos][resp][2]
        
            dat = (dat - dat[0]) * 1.0e15
            diag_dat = (diag_dat - diag_dat[0]) * 1.0e15

            axarrs[biasind][resp,0].errorbar(bins, dat, errs, \
                                             fmt='-', marker='.', \
                                             ms=7, color = color, label=lab, \
                                             alpha=0.9)
            axarrs[biasind][resp,1].errorbar(diag_bins, diag_dat, diag_errs, \
                                             fmt='-', marker='.', \
                                             ms=7, color = color, label=lab, \
                                             alpha=0.9)

for arrind, arr in enumerate(axarrs):

    voltage = cantV[arrind]
    title = 'Dipole vs Height for %i V (filtered)' % int(voltage)
    arr[0,0].set_title('Raw Data', fontsize=12)
    arr[0,1].set_title('Diagonalized Data', fontsize=12)
    arr[2,0].set_xlabel('Distance From Cantilever [um]', fontsize=12)
    arr[2,1].set_xlabel('Distance From Cantilever [um]', fontsize=12)

    for resp in [0,1,2]:
        arr[resp,0].set_ylabel('Force [fN]')
        plt.setp(arr[resp,0].get_xticklabels(), fontsize=10, visible=True)
        plt.setp(arr[resp,0].get_yticklabels(), fontsize=10, visible=True)
        plt.setp(arr[resp,1].get_xticklabels(), fontsize=10, visible=True)
        plt.setp(arr[resp,1].get_yticklabels(), fontsize=10, visible=True)
        

    arr[0,1].legend(loc=0, numpoints=1, fontsize=6, ncol=2)
    plt.tight_layout()
    figs[arrind].suptitle(title, fontsize=14)
    figs[arrind].subplots_adjust(top=0.9)

#plt.tight_layout()
plt.show()
