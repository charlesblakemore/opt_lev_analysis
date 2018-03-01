import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp
import scipy.signal as signal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from mpl_toolkits.mplot3d import Axes3D

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config


### Specify path with data, or folder with many picomotor subfolders

dir1 = '/data/20180220/bead1/gravity_data/grav_data_noshield/'
#dir1 = '/data/20180215/bead1/grav_data_withshield/'

parts = dir1.split('/')
save_path1 = '/force_v_pos/%s/%s/%s_force_v_pos_dic.p' % (parts[2], parts[3], parts[-2])
save_path2 = '/force_v_pos/%s/%s/%s_diagforce_v_pos_dic.p' % (parts[2], parts[3], parts[-2])
bu.make_all_pardirs(save_path1)
bu.make_all_pardirs(save_path2)

save = False #True
load = True
load_path = '/force_v_pos/%s/%s/%s_force_v_pos_dic.p' % (parts[2], parts[3], parts[-2])



# If True, this will trigger the script to use the following inputs
# and process many directories simultaneously
picomotors = False
picodir = '/data/20171106/bead1/grav_data_10picopos/'
date = picodir.split('/')[2]
numdirs = bu.count_dirs(picodir)
parent_savepath = '/force_v_pos/' + date + '_' + str(numdirs) + 'picopos_2/'
if not os.path.exists(parent_savepath):
    print 'Making directory: ', parent_savepath
    os.makedirs(parent_savepath)



optional_ext = '' #'300Hz-tophat_100mHz-notch_10harm'




### Specify other inputs

maxfiles = 10000   # Many more than necessary
ax1_lab = 'x'
ax2_lab = 'z'
nbins = 100        # Bins per full cantilever throw
lpf = 300          # top-hat filter cutoff
nharmonics = 10    # harmonics of cantilever drive to include in spatial binning
width = 0.1        # Notch filter width in Hz

apply_butter = True   # Whether to apply a butterworth filter to data
butter_freq = 100
butter_order = 3


plot = True
resp_to_plot = 0
ax2_toplot = 7

smooth_manifold = True   # Whether to smooth the final manifold
tukey_alpha = 0.1        # alpha param for tukey window in manifold smoothing

fakedrive = True
fakefreq = 29
fakeamp = 40


### These arrays are for testing various aspects of averaging files
### together and some smoothing/resampling
testind = 0
test_posvec = [[],[],[]]
test_posvec_int = [[],[],[]]
test_posvec_final = [[],[],[]]
test_arr = [[],[],[]]
test_arr_int = [[],[],[]]
test_arr_final = [[],[],[]]
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

    ### Do inital looping over files to concatenate data at the same
    ### heights and separations
    force_curves = {}
    if diag:
        diag_force_curves = {}
    old_per = 0
    print
    print os.path.dirname(files[0])
    print "Processing %i files" % len(files)
    print "Percent complete: "
    for fil_ind, fil in enumerate(files):

        bu.progress_bar(fil_ind, len(files))

        # Display percent completion
        #per = int(100. * float(fil_ind) / float(len(files)) )
        #if per > old_per:
        #    print old_per,
        #    sys.stdout.flush()
        #    old_per = per

        # Load data
        df = bu.DataFile()
        df.load(fil)

        df.calibrate_stage_position()
        
        # Pick out height and separation
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
            df.diagonalize(maxfreq=lpf)

        df.get_force_v_pos(verbose=False, nbins=nbins, nharmonics=nharmonics, \
                           width=width, fakedrive=fakedrive, fakefreq=fakefreq, fakeamp=fakeamp)

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
    test_ax1 = 38
    max_ax1 = ax1_keys[np.argmin( np.abs( test_ax1 - np.array(ax1_keys)) )]
    ax2pos = ax2_keys[np.argmin( np.abs(ax2_toplot - np.array(ax2_keys)) )]

    ax1_keys.sort()
    ax2_keys.sort()

    for ax1_k in ax1_keys:
        for ax2_k in ax2_keys:
            for resp in [0,1,2]:

                old_bins = force_curves[ax1_k][ax2_k][resp][0]
                old_dat = force_curves[ax1_k][ax2_k][resp][1]

                new_bins = np.linspace(np.min(old_bins)+1e-9, np.max(old_bins)-1e-9, nbins)

                bin_sp = new_bins[1] - new_bins[0]

                int_bins = []
                int_dat = []

                num_files = int(np.sum( np.abs(old_bins - old_bins[0]) <= 0.2 * bin_sp ))
                #num_files = 3
                #print num_files

                #for binval in old_bins[::num_files]:
                #    inds = np.abs(old_bins - binval) <= 0.2 * bin_sp
                #    avg_bin = np.mean(old_bins[inds])
                #    if avg_bin not in int_bins:
                #        int_bins.append(avg_bin)
                #        int_dat.append(np.mean(old_dat[inds]))

                #dat_func = interp.interp1d(old_bins, old_dat, kind='cubic', bounds_error=False,\
                #                           fill_value='extrapolate')

                #new_dat = dat_func(new_bins)
                #new_errs = np.zeros_like(new_dat)

                new_dat = np.zeros_like(new_bins)
                new_errs = np.zeros_like(new_bins)
                for binind, binval in enumerate(new_bins):
                    inds = np.abs(old_bins - binval) <= 0.5*bin_sp
                    new_dat[binind] = np.mean(old_dat[inds])
                    new_errs[binind] = np.std(old_dat[inds])

                if ax1_k == max_ax1:
                    if ax2_k == ax2pos:
                        test_posvec[resp] = old_bins
                        test_posvec_int[resp] = int_bins
                        test_posvec_final[resp] = new_bins
                        test_arr[resp] = old_dat
                        test_arr_int[resp] = int_dat
                        test_arr_final[resp] = new_dat

                force_curves[ax1_k][ax2_k][resp] = [new_bins, new_dat, new_errs]

                if diag:
                    old_diag_bins = diag_force_curves[ax1_k][ax2_k][resp][0]
                    old_diag_dat = diag_force_curves[ax1_k][ax2_k][resp][1]

                    if ax1_k == max_ax1:
                        if ax2_k == ax2pos:
                            diag_test_posvec[resp] = old_diag_bins
                            diag_test_arr[resp] = old_diag_dat


                    new_diag_bins = np.linspace(np.min(old_diag_bins)+1e-9, \
                                                np.max(old_diag_bins)-1e-9, nbins)

                    diag_bin_sp = new_diag_bins[1] - new_diag_bins[0]

                    int_diag_bins = []
                    int_diag_dat = []

                    # num_files = int( np.sum( np.abs(old_diag_bins - old_diag_bins[0]) \
                    #                          <= 0.2 * diag_bin_sp ) )

                    # for binval in old_diag_bins[::num_files]:
                    #     inds = np.abs(old_diag_bins - binval) <= 0.2 * diag_bin_sp
                    #     int_diag_bins.append(np.mean(old_diag_bins[inds]))
                    #     int_diag_dat.append(np.mean(old_diag_dat[inds]))

                    # diag_dat_func = interp.interp1d(int_diag_bins, int_diag_dat, kind='cubic', \
                    #                                bounds_error=False, fill_value='extrapolate')

                    # new_diag_dat = diag_dat_func(new_diag_bins)
                    # new_diag_errs = np.zeros_like(new_diag_dat)

                    # diag_bin_sp = new_diag_bins[1] - new_diag_bins[0]
                    # for binind, binval in enumerate(new_diag_bins):
                    #     diaginds = np.abs( old_diag_bins - binval ) < diag_bin_sp
                    #     new_diag_errs[binind] = np.std( old_diag_dat[diaginds] )


                    new_diag_dat = np.zeros_like(new_diag_bins)
                    new_diag_errs = np.zeros_like(new_diag_bins)
                    for binind, binval in enumerate(new_diag_bins):
                        inds = np.abs(old_diag_bins - binval) <= 0.5*diag_bin_sp
                        new_diag_dat[binind] = np.mean(old_diag_dat[inds])
                        new_diag_errs[binind] = np.std(old_diag_dat[inds])

                    diag_force_curves[ax1_k][ax2_k][resp] = \
                                            [new_diag_bins, new_diag_dat, new_diag_errs]
                    
                    
    

    if diag:
        return force_curves, diag_force_curves
    else:
        return force_curves




if not load:

    if not picomotors:
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
    
    elif picomotors:
        for i in range(numdirs):
            path = picodir + '/pico_p' + str(i) + '/'
            files = bu.find_all_fnames(path)
            files = files[:maxfiles]

            save_path1 = parent_savepath + date + '_force_v_pos_dic_' + \
                             optional_ext + '_p' +str(i) + '.p'
            save_path2 = parent_savepath + date + '_diagforce_v_pos_dic_' + \
                             optional_ext + '_p' + str(i) + '.p'

            if len(files) == 0:
                print 'No Files Found in: ', path
                quit()

            else:
                force_dic, diag_force_dic = \
                            get_force_curve_dictionary(files, ax1=ax1_lab, ax2=ax2_lab, \
                                                       spacing=1e-6, diag=True)
                if save:
                    pickle.dump(force_dic, open(save_path1, 'wb') )
                    pickle.dump(diag_force_dic, open(save_path2, 'wb') )



fig, axarr = plt.subplots(3,2,sharex=True,sharey=True)
for resp in [0,1,2]:
    axarr[resp,0].plot(test_posvec[resp], test_arr[resp])
    axarr[resp,0].plot(test_posvec_int[resp], test_arr_int[resp])
    axarr[resp,1].plot(test_posvec_final[resp], test_arr_final[resp])
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

    xgrid, ygrid = np.meshgrid(sample_yvec, ax1)
    surf = ax.plot_surface(xgrid, ygrid, fgrid, \
                           rcount=20, ccount=100,\
                           cmap=cm.coolwarm, linewidth=0, antialiased=True)

    fig.colorbar(surf, aspect=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.suptitle('Raw Data')



    xgrid2, ygrid2 = np.mgrid[np.min(sample_yvec):np.max(sample_yvec):200j, \
                              np.min(ax1):np.max(ax1):200j]

    xwin = signal.tukey(len(sample_yvec), alpha=tukey_alpha)
    ywin = signal.tukey(len(ax1), alpha=tukey_alpha)

    gridwin = np.einsum('i,j->ji', xwin, ywin)

    if smooth_manifold:
        s = None
    else:
        s = 0
    tck = interp.bisplrep(xgrid, ygrid, fgrid*gridwin, s=s)
    fgrid_fine = interp.bisplev(xgrid2[:,0], ygrid2[0,:], tck)

    #ax.plot_wireframe(sepgrid, xgrid, out)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')

    surf2 = ax2.plot_surface(xgrid2, ygrid2, fgrid_fine, rstride=1, cstride=1, \
                             cmap=cm.coolwarm, linewidth=0, antialiased=True)

    fig2.colorbar(surf2, aspect=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    if smooth_manifold:
        plt.suptitle('Oversampled/Smoothed Data')
    else:
        plt.suptitle('Oversampled Data')


    plt.show()
