###########################
# Script to analyze the microsphere's dipole response to a fixed
# voltage on the cantilver which is driven toward and away from
# the bead
###########################

import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti
from scipy.optimize import minimize_scalar as minimize

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from mpl_toolkits.mplot3d import Axes3D

import bead_util as bu
import configuration as config

###########################################################



dir1 = '/data/20180618/bead1/dipole_vs_height/10V_70um_17Hz_2'
start_file = 0
maxfiles = 6000
ax1_lab = 'z'
nbins = 30
tophatf = 300  # Top-hat filter frequency used in diagonalization

plot_title = ''

tfdate = '' #'20180215'

fit_xdat = True
fit_zdat = True
closest_sep = 20
#closest_sep = 60

diag = True
###########################################################

def dipole_force(x, a, b, c, x0=0):
    return a*(1.0/np.abs(x-x0))**2 + b*(1.0/np.abs(x-x0)) + c

def parabola(x, a, b, c):
    return a*(x**2) + b*x + c

def get_force_curve_dictionary(files, cantind=0, ax1='z', fullax1=True, \
                               ax1val=0,  spacing=1e-6, diag=False, fit_xdat=False, \
                               fit_zdat=False, plottf=False):
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
        bu.progress_bar(fil_ind, len(files))

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
            if fil_ind == 0 and plottf:
                df.diagonalize(date=tfdate, maxfreq=tophatf, plot=True)
            else:
                df.diagonalize(date=tfdate, maxfreq=tophatf)

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

                #plt.plot(new_bins[sort_inds], new_dat[sort_inds])
                #plt.show()

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

    if fit_xdat:
        xdat = {'fit': dipole_force}
        diag_xdat = {'fit': dipole_force}
    if fit_zdat:
        zdat = {'fit': dipole_force}
        diag_zdat = {'fit': dipole_force}

    for cantV_k in cantV_keys:
        if fit_zdat:
            if cantV_k not in zdat:
                zdat[cantV_k] = {}
                if diag:
                    diag_zdat[cantV_k] = {}
        if fit_xdat:
            if cantV_k not in xdat:
                xdat[cantV_k] = {}
                if diag:
                    diag_xdat[cantV_k] = {}

        for ax1_k in ax1_keys:
            for resp in [0,1,2]:

                old_bins = force_curves[cantV_k][ax1_k][resp][0]
                old_dat = force_curves[cantV_k][ax1_k][resp][1]

                #dat_func = interp.interp1d(old_bins, old_dat, kind='cubic')

                new_bins = np.linspace(np.min(old_bins)+1e-9, np.max(old_bins)-1e-9, nbins)
                new_dat = np.zeros_like(new_bins)
                new_errs = np.zeros_like(new_bins)

                bin_sp = new_bins[1] - new_bins[0]
                for binind, binval in enumerate(new_bins):
                    inds = np.abs( old_bins - binval ) < bin_sp
                    new_dat[binind] = np.mean( old_dat[inds] )
                    new_errs[binind] = np.std( old_dat[inds] )

                force_curves[cantV_k][ax1_k][resp] = [new_bins, new_dat, new_errs]

                if fit_xdat and resp == 0:
                    x0 = np.max(new_bins) + closest_sep
                    p0 = [np.max(new_dat)/closest_sep**2, 0, 0]
                    fitfun = lambda x,a,b,c: xdat['fit'](x,a,b,c,x0=x0)
                    popt, pcov = opti.curve_fit(fitfun, new_bins, new_dat)
                    val = fitfun(np.max(new_bins), popt[0], popt[1], 0)

                    #print resp
                    #print fitfun(-200, *popt) - popt[2]
                    #print fitfun(50, *popt) - popt[2]
                    #plt.plot(new_bins, new_dat, label='Dat')
                    #plt.plot(new_bins, fitfun(new_bins, *popt), label='Fit')
                    #plt.legend()
                    #plt.show()

                    xdat[cantV_k][ax1_k] = (popt, val)

                if fit_zdat and resp == 2:
                    x0 = np.max(new_bins) + closest_sep
                    p0 = [np.max(new_dat)/closest_sep**2, 0, 0]
                    fitfun = lambda x,a,b,c: zdat['fit'](x,a,b,c,x0=x0)
                    popt, pcov = opti.curve_fit(fitfun, new_bins, new_dat)
                    val = fitfun(np.max(new_bins), popt[0], popt[1], 0)

                    zdat[cantV_k][ax1_k] = (popt, val)

                if diag:
                    old_diag_bins = diag_force_curves[cantV_k][ax1_k][resp][0]
                    old_diag_dat = diag_force_curves[cantV_k][ax1_k][resp][1]
                    
                    #diag_dat_func = interp.interp1d(old_diag_bins, old_diag_dat, kind='cubic')

                    new_diag_bins = np.linspace(np.min(old_diag_bins)+1e-9, \
                                                np.max(old_diag_bins)-1e-9, nbins)
                    new_diag_dat = np.zeros_like(new_diag_bins)
                    new_diag_errs = np.zeros_like(new_diag_bins)

                    diag_bin_sp = new_diag_bins[1] - new_diag_bins[0]
                    for binind, binval in enumerate(new_diag_bins):
                        diaginds = np.abs( old_diag_bins - binval ) < diag_bin_sp
                        new_diag_errs[binind] = np.std( old_diag_dat[diaginds] )
                        new_diag_dat[binind] = np.mean( old_diag_dat[diaginds] )

                    diag_force_curves[cantV_k][ax1_k][resp] = \
                                        [new_diag_bins, new_diag_dat, new_diag_errs]

                    if fit_xdat and resp == 0:
                        x0 = np.max(new_diag_bins) + closest_sep
                        p0 = [np.max(new_diag_dat)/closest_sep**2, 0, 0]
                        fitfun = lambda x,a,b,c: diag_xdat['fit'](x,a,b,c,x0=x0)
                        popt, pcov = opti.curve_fit(fitfun, new_diag_bins, new_diag_dat)
                        val = fitfun(np.max(new_diag_bins), popt[0], popt[1], 0)

                        diag_xdat[cantV_k][ax1_k] = (popt, val)

                    if fit_zdat and resp == 2:
                        x0 = np.max(new_diag_bins) + closest_sep
                        p0 = [np.max(new_diag_dat)/closest_sep**2, 0, 0]
                        fitfun = lambda x,a,b,c: diag_zdat['fit'](x,a,b,c,x0=x0)
                        popt, pcov = opti.curve_fit(fitfun, new_diag_bins, new_diag_dat)
                        val = fitfun(np.max(new_diag_bins), popt[0], popt[1], 0)

                        diag_zdat[cantV_k][ax1_k] = (popt, val)

                
    fits = {}
    if fit_xdat:
        if diag:
            fits['x'] = (xdat, diag_xdat)
        else:
            fits['x'] = (xdat)
    if fit_zdat:
        if diag:
            fits['z'] = (zdat, diag_zdat)
        else:
            fits['z'] = (zdat)

    
    if diag:
        return force_curves, diag_force_curves, fits
    else:
        return force_curves, fits





datafiles = bu.find_all_fnames(dir1, ext=config.extensions['data'])
datafiles = datafiles[start_file:start_file+maxfiles]

force_dic, diag_force_dic, fits= \
            get_force_curve_dictionary(datafiles, ax1=ax1_lab, diag=diag, \
                                       fit_xdat=fit_xdat, fit_zdat=fit_zdat, plottf=False)

if fit_xdat:
    xdat = fits['x'][0]
    if diag:
        diag_xdat = fits['x'][1]

if fit_zdat:
    zdat = fits['z'][0]
    if diag:
        diag_zdat = fits['z'][1]


cantV = force_dic.keys()
cantV.sort()

figs = []
axarrs = []
xfigs = []
xaxarrs = []
zfigs = []
zaxarrs = []

for biasind, bias in enumerate(cantV):
    fig, axarr = plt.subplots(3,2,sharex=True,sharey=True,figsize=(6,8),dpi=150)
    figs.append(fig)
    axarrs.append(axarr)

    if fit_xdat:
        xfig, xaxarr = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,3),dpi=150)
        xfigs.append(xfig)
        xaxarrs.append(xaxarr)
        xfits = []
        diag_xfits = []

    if fit_zdat:
        zfig, zaxarr = plt.subplots(1,2,sharex=True,sharey=True,figsize=(5,3),dpi=150)
        zfigs.append(zfig)
        zaxarrs.append(zaxarr)
        zfits = []
        diag_zfits = []

    stage_settings = force_dic[bias].keys()
    stage_settings.sort()
    stage_settings = np.array(stage_settings)

    nsettings = len(stage_settings)
    if nsettings < 10:
        colors = ['C' + str(i) for i in range(nsettings)]
    else:
        colors = bu.get_color_map(nsettings, cmap='jet')

    for posind, pos in enumerate(stage_settings):
        color = colors[posind]
        lab = str(pos) + ' um'

        if fit_xdat:
            xfits.append(xdat[bias][pos][1])
            diag_xfits.append(diag_xdat[bias][pos][1])

        if fit_zdat:
            zfits.append(zdat[bias][pos][1])
            diag_zfits.append(diag_zdat[bias][pos][1])

        for resp in [0,1,2]:
            bins = force_dic[bias][pos][resp][0]
            dat = force_dic[bias][pos][resp][1]
            errs = force_dic[bias][pos][resp][2]

            diag_bins = diag_force_dic[bias][pos][resp][0]
            diag_dat = diag_force_dic[bias][pos][resp][1]
            diag_errs = diag_force_dic[bias][pos][resp][2]
        
            if resp == 0 and fit_xdat:
                off = np.mean(dat[0]) + xdat[bias][pos][0][2]
                doff = np.mean(diag_dat[0]) + diag_xdat[bias][pos][0][2]
            elif resp == 2 and fit_zdat:
                off = np.mean(dat[0]) + zdat[bias][pos][0][2]
                doff = np.mean(diag_dat[0]) + diag_zdat[bias][pos][0][2]
            else:
                off = dat[0]
                doff = diag_dat[0]

            dat = (dat - off) * 1.0e15
            diag_dat = (diag_dat - doff) * 1.0e15

            axarrs[biasind][resp,0].errorbar(bins, dat, errs, \
                                             fmt='-', marker='.', \
                                             ms=7, color = color, label=lab, \
                                             alpha=0.9)
            axarrs[biasind][resp,1].errorbar(diag_bins, diag_dat, diag_errs, \
                                             fmt='-', marker='.', \
                                             ms=7, color = color, label=lab, \
                                             alpha=0.9)

    if fit_xdat:
        xfits = np.array(xfits)
        diag_xfits = np.array(diag_xfits)

        maxind = np.argmax(np.abs(xfits))
        diag_maxind = np.argmax(np.abs(diag_xfits))

        gpts = np.abs(stage_settings - stage_settings[maxind]) < 15
        diaggpts = np.abs(stage_settings - stage_settings[diag_maxind]) < 15

        popt, pcov = opti.curve_fit(parabola, stage_settings[gpts], xfits[gpts])
        popt_diag, pcov_diag = opti.curve_fit(parabola, stage_settings[diaggpts], diag_xfits[diaggpts])

        beadheight = 'Height: %0.3g um' % (-0.5*popt[1]/popt[0])
        diag_beadheight = 'Height: %0.3g um' % (-0.5*popt_diag[1]/popt_diag[0])

        xaxarrs[biasind][0].errorbar(stage_settings, xfits, marker='.', ms=7)
        xaxarrs[biasind][1].errorbar(stage_settings, diag_xfits, marker='.', ms=7)

        xaxarrs[biasind][0].plot(stage_settings, parabola(stage_settings, *popt), '--', color='r')
        xaxarrs[biasind][0].plot(stage_settings[gpts], parabola(stage_settings[gpts], *popt), \
                                 color='r', lw=2)

        xaxarrs[biasind][1].plot(stage_settings, \
                                 parabola(stage_settings, *popt_diag), '--', color='r')
        xaxarrs[biasind][1].plot(stage_settings[diaggpts], \
                                 parabola(stage_settings[diaggpts], *popt_diag), \
                                 color='r', lw=2)

        xaxarrs[biasind][0].text(0.3*np.max(stage_settings), np.min(xfits), beadheight) 
        xaxarrs[biasind][1].text(0.3*np.max(stage_settings), np.min(diag_xfits), diag_beadheight) 

        xaxarrs[biasind][0].set_xlabel('Cantilever Height [um]')
        xaxarrs[biasind][1].set_xlabel('Cantielver Height [um]')
        xaxarrs[biasind][0].set_ylabel('Peak X-force From Fit')

        xaxarrs[biasind][0].set_title('Raw Data', fontsize=12)
        xaxarrs[biasind][1].set_title('Diagonalized Data', fontsize=12)

        plt.tight_layout()


    if fit_zdat:
        zfits = np.array(zfits)
        diag_zfits = np.array(diag_zfits)

        fitfun = lambda x,a,b: a*x + b

        crossind = np.argmin(np.abs(zfits))
        diag_crossind = np.argmin(np.abs(diag_zfits))

        gpts = np.abs(stage_settings - stage_settings[crossind]) < 15
        diaggpts = np.abs(stage_settings - stage_settings[diag_crossind]) < 15

        popt, pcov = opti.curve_fit(fitfun, stage_settings[gpts], zfits[gpts])
        popt_diag, pcov_diag = opti.curve_fit(fitfun, stage_settings[diaggpts], diag_zfits[diaggpts])

        beadheight = 'Height: %0.3g um' % (-1.0*popt[1]/popt[0])
        diag_beadheight = 'Height: %0.3g um' % (-1.0*popt_diag[1]/popt_diag[0])

        zaxarrs[biasind][0].errorbar(stage_settings, zfits, marker='.', ms=7)
        zaxarrs[biasind][1].errorbar(stage_settings, diag_zfits, marker='.', ms=7)

        zaxarrs[biasind][0].plot(stage_settings, fitfun(stage_settings, *popt), '--', color='r')
        zaxarrs[biasind][0].plot(stage_settings[gpts], fitfun(stage_settings[gpts], *popt), \
                                 color='r', lw=2)

        zaxarrs[biasind][1].plot(stage_settings, \
                                 fitfun(stage_settings, *popt_diag), '--', color='r')
        zaxarrs[biasind][1].plot(stage_settings[diaggpts], \
                                 fitfun(stage_settings[diaggpts], *popt_diag), \
                                 color='r', lw=2)

        zaxarrs[biasind][0].text(np.min(stage_settings), np.max(zfits), beadheight) 
        zaxarrs[biasind][1].text(np.min(stage_settings), np.max(diag_zfits), diag_beadheight) 

        zaxarrs[biasind][0].set_xlabel('Cantilever Height [um]')
        zaxarrs[biasind][1].set_xlabel('Cantielver Height [um]')
        zaxarrs[biasind][0].set_ylabel('Peak Z-force From Fit')

        zaxarrs[biasind][0].set_title('Raw Data', fontsize=12)
        zaxarrs[biasind][1].set_title('Diagonalized Data', fontsize=12)

        plt.tight_layout()

for arrind, arr in enumerate(axarrs):

    voltage = cantV[arrind]
    title = 'Dipole vs Height for %i V' % int(voltage)
    arr[0,0].set_title('Raw Data', fontsize=12)
    arr[0,1].set_title('Diagonalized Data', fontsize=12)
    arr[2,0].set_xlabel('Distance From Cantilever [um]', fontsize=10)
    arr[2,1].set_xlabel('Distance From Cantilever [um]', fontsize=10)

    rdict = {0: 'X', 1: 'Y', 2: 'Z'}
    for resp in [0,1,2]:
        arr[resp,0].set_ylabel(rdict[resp] + ' Force [fN]')
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
