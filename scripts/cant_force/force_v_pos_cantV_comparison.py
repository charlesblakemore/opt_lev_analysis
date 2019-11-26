import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import configuration as config



#dir1 = '/data/20171106/bead1/response_vs_dcvoltage_10Vunfilt/'
dir1 = '/data/20171106/bead1/response_vs_acvoltage_5V/'
maxfiles = 10000 # Many more than necessary
ax1_lab = 'x'
ax2_lab = 'z'
nbins = 40 # Final number of force v pos bins
lpf = 100   # Hz

plot = True
ax1_toplot = 75
ax2_toplot = 0

save = False
load = False


###########################################################

load_path = '/force_v_pos/20171106_many_cantV.p'
save_path = load_path

diag_load_path = '/force_v_pos/20171106_many_cantV_diag.p'
diag_save_path = diag_load_path


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
    print("Percent complete: ")
    for fil_ind, fil in enumerate(files):
        # Display percent completion
        per = int(100. * float(fil_ind) / float(len(files)) )
        if per > old_per:
            print(old_per, end=' ')
            sys.stdout.flush()
            old_per = per

        # Load data
        df = bu.DataFile()
        df.load(fil)

        df.calibrate_stage_position()
        
        df.high_pass_filter(fc=5)

        #cantbias = df.electrode_settings['dc_settings'][0]

        cantbias = df.electrode_settings['amplitudes'][0]

        ax1pos = df.stage_settings[ax1 + ' DC']
        ax2pos = df.stage_settings[ax2 + ' DC']

        if diag:
            df.diagonalize(maxfreq=lpf)

        df.get_force_v_pos(verbose=False, nbins=nbins, nharmonics=10, width=0)

        if cantbias not in list(force_curves.keys()):
            force_curves[cantbias] = {}
            if diag:
                diag_force_curves[cantbias] = {}

        if ax1pos not in list(force_curves[cantbias].keys()):
            force_curves[cantbias][ax1pos] = {}
            if diag:
                diag_force_curves[cantbias][ax1pos] = {}

        if ax2pos not in list(force_curves[cantbias][ax1pos].keys()):
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


    cantV_keys = list(force_curves.keys())
    ax1_keys = list(force_curves[cantV_keys[0]].keys())
    ax2_keys = list(force_curves[cantV_keys[0]][ax1_keys[0]].keys())

    for cantV in cantV_keys:
        for ax1 in ax1_keys:
            for ax2 in ax2_keys:
                for resp in [0,1,2]:
                    old_bins = force_curves[cantV][ax1][ax2][resp][0]
                    old_dat = force_curves[cantV][ax1][ax2][resp][1]

                    dat_func = interp.interp1d(old_bins, old_dat, kind='cubic')

                    new_bins = np.linspace(np.min(old_bins)+1e-9, 
                                           np.max(old_bins)-1e-9, nbins)
                    new_dat = dat_func(new_bins)
                    new_errs = np.zeros_like(new_dat)

                    bin_sp = new_bins[1] - new_bins[0]
                    for binind, binval in enumerate(new_bins):
                        inds = np.abs( old_bins - binval ) < bin_sp
                        new_errs[binind] = np.std( old_dat[inds] )

                    force_curves[cantV][ax1][ax2][resp] = \
                                            [new_bins, new_dat, new_errs]

                    if diag:                        
                        old_diag_bins = diag_force_curves[cantV][ax1][ax2][resp][0]
                        old_diag_dat = diag_force_curves[cantV][ax1][ax2][resp][1]

                        diag_dat_func = interp.interp1d(old_diag_bins, \
                                                        old_diag_dat, kind='cubic')

                        new_diag_bins = np.linspace(np.min(old_diag_bins)+1e-9, 
                                           np.max(old_diag_bins)-1e-9, nbins)
                        new_diag_dat = diag_dat_func(new_diag_bins)
                        new_diag_errs = np.zeros_like(new_diag_dat)

                        diag_bin_sp = new_diag_bins[1] - new_diag_bins[0]
                        for binind, binval in enumerate(new_diag_bins):
                            inds = np.abs( old_diag_bins - binval ) < diag_bin_sp
                            new_diag_errs[binind] = np.std( old_diag_dat[inds] )

                        diag_force_curves[cantV][ax1][ax2][resp] = \
                                [new_diag_bins, new_diag_dat, new_diag_errs]


    if diag:
        return force_curves, diag_force_curves
    else:
        return force_curves



def select_allbias_onepos(force_dic, ax1pos, ax2pos):
    '''Selects all the data at a particular cantilever position and
       compares force curves with different biases applied

       INPUTS: force_dic, list of files names to extract data
               ax1pos, ax1 value to keep (or closest)
               ax2pos, ax2       "     "

       OUTPUTS: outdic, ouput dictionary with the following indexing
                        outdic[cantV][resp(0,1,2)][bins(0) or dat(1) or errs(2)]
    '''


    cantVvec = list(force_dic.keys())
    ax1vec = list(force_dic[cantVvec[0]].keys())
    ax2vec = list(force_dic[cantVvec[0]][ax1vec[0]].keys())

    ax1pos = ax1vec[ np.argmin( np.abs(np.array(ax1vec) - ax1pos) ) ]
    ax2pos = ax2vec[ np.argmin( np.abs(np.array(ax2vec) - ax2pos) ) ]

    new_dic = {}
    for cantV in cantVvec:
        new_dic[cantV] = force_dic[cantV][ax1pos][ax2pos]
    
    return new_dic



if not load:

    files = bu.find_all_fnames(dir1)
    files = files[:maxfiles]
    force_dic, diag_force_dic = \
        get_force_curve_dictionary(files, ax1=ax1_lab, ax2=ax2_lab, diag=True)

    if save:
        pickle.dump(force_dic, open(save_path, 'wb') )
        pickle.dump(diag_force_dic, open(diag_save_path, 'wb') )


if load:
    force_dic = pickle.load( open(load_path, 'rb') )
    diag_force_dic = pickle.load( open(diag_load_path, 'rb') )



force_dic = select_allbias_onepos(force_dic, ax1_toplot, ax2_toplot)
diag_force_dic = select_allbias_onepos(diag_force_dic, ax1_toplot, ax2_toplot)



if plot:

    cantVvec = list(force_dic.keys())  
    cantVvec.sort()

    fig, axarr = plt.subplots(3,2,sharex=True,sharey=True,figsize=(6,8),dpi=150)

    colors = bu.get_color_map(len(cantVvec))

    for cantind, cantV in enumerate(cantVvec):
        color = colors[cantind]
        lab = '%0.2f V' % cantV

        for resp in [0,1,2]:
            bins = force_dic[cantV][resp][0]
            dat = force_dic[cantV][resp][1]
            errs = force_dic[cantV][resp][2]

            diag_bins = diag_force_dic[cantV][resp][0]
            diag_dat = diag_force_dic[cantV][resp][1]
            diag_errs = diag_force_dic[cantV][resp][2]
        
            offset = 0
            #offset = dat[0]
            doffset = 0
            #doffset = diag_dat[0]

            dat = (dat - offset) * 1.0e15
            diag_dat = (diag_dat - doffset) * 1.0e15

            axarr[resp,0].errorbar(bins, dat, errs, \
                                    fmt='-', marker='.', \
                                    ms=7, color = color, label=lab, \
                                    alpha=0.9)
            axarr[resp,1].errorbar(diag_bins, diag_dat, diag_errs, \
                                    fmt='-', marker='.', \
                                    ms=7, color = color, label=lab, \
                                    alpha=0.9)


    title = 'Grav Response vs Voltage'
    axarr[0,0].set_title('Raw Data', fontsize=12)
    axarr[0,1].set_title('Diagonalized Data', fontsize=12)
    axarr[2,0].set_xlabel('Distance Along Cantilever [um]')
    axarr[2,1].set_xlabel('Distance Along Cantilever [um]')

    for resp in [0,1,2]:
        axarr[resp,0].set_ylabel('Force [fN]')
        plt.setp(axarr[resp,0].get_xticklabels(), fontsize=10, visible=True)
        plt.setp(axarr[resp,0].get_yticklabels(), fontsize=10, visible=True)
        plt.setp(axarr[resp,1].get_xticklabels(), fontsize=10, visible=True)
        plt.setp(axarr[resp,1].get_yticklabels(), fontsize=10, visible=True)
        

    axarr[1,1].legend(loc=0, numpoints=1, fontsize=6, ncol=2)
    plt.tight_layout()
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.9, bottom=0.1)


    plt.show()
