import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

dir1 = '/data/20180605/bead1/overnight'
maxfiles = 200

use_dir = False

#allfiles = [r'C:\Data\20180611\bead6\1_4mbar_zcool_init.h5',\
#            r'C:\Data\20180611\bead6\1_4mbar_zcool_low1.h5', \
#            r'C:\Data\20180611\bead6\1_4mbar_zcool_low2.h5']

allfiles = [r'C:\Data\20180611\bead6\1_4mbar_xyzcool.h5',\
            r'C:\Data\20180611\bead6\turbombar_xyzcool_pumped.h5', \
            r'C:\Data\20180611\bead6\post_pump3\turbombar_xyzcool_pumped_0.h5']

labels = []
labels = ['1.4 mbar', 'Post-Pump', 'Much Later']
matplotlib_colors = False

data_axes = [0,1,2]
other_axes = []
#other_axes = [5,7]

drive_ax = 1

step10 = False #True
invert_order = False

#### HACKED SHITTTT
savefigs = False
title_pre = '/home/charles/plots/20180105_precession/test1_100V_muchlater3'


#ylim = (1e-21, 1e-14)
#ylim = (1e-7, 1e-1)
ylim = ()

lpf = 2500   # Hz

file_inds = (0, 1800)

userNFFT = 2**12
diag = False

fullNFFT = False

#window = mlab.window_hanning
window = mlab.window_none

###########################################################


outvec1 = []
outvec2 = []


def plot_many_spectra(files, data_axes=[0,1,2], cant_axes=[], elec_axes=[], other_axes=[], \
                      diag=True, colormap='jet', sort='time', file_inds=(0,10000)):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then plots the amplitude spectral density of any number of data
       or cantilever/electrode drive signals

       INPUTS: files, list of files names to extract data
               data_axes, list of pos_data axes to plot
               cant_axes, list of cant_data axes to plot
               elec_axes, list of electrode_data axes to plot
               diag, boolean specifying whether to diagonalize

       OUTPUTS: none, plots stuff
    '''

    if diag:
        dfig, daxarr = plt.subplots(len(data_axes),2,sharex=True,sharey=True, \
                                    figsize=(8,8))
    else:
        dfig, daxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=False, \
                                    figsize=(8,8))

    if len(cant_axes):
        cfig, caxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True)
        if len(cant_axes) == 1:
            caxarr = [caxarr]
    if len(elec_axes):
        efig, eaxarr = plt.subplots(len(elec_axes),1,sharex=True,sharey=True)
        if len(elec_axes) == 1:
            eaxarr = [eaxarr]
    if len(other_axes):
        ofig, oaxarr = plt.subplots(len(other_axes),1,sharex=True,sharey=True)
        if len(other_axes) == 1:
            oaxarr = [oaxarr]


    files = [(os.stat(path), path) for path in files]
    files = [(stat.st_ctime, path) for stat, path in files]
    files.sort(key = lambda x: (x[0]))
    files = [obj[1] for obj in files]

    files = files[file_inds[0]:file_inds[1]]
    if step10:
        files = files[::10]
    if invert_order:
        files = files[::-1]

    colors = bu.get_color_map(len(files), cmap=colormap)
    colors = ['r', 'g', 'b']
    if matplotlib_colors:
        colors = ['C' + str(i) for i in range(10)]

    old_per = 0
    print "Processing %i files..." % len(files)
    for fil_ind, fil in enumerate(files):
        color = colors[fil_ind]
        
        # Display percent completion
        bu.progress_bar(fil_ind, len(files))

        # Load data
        df = bu.DataFile()
        df.load(fil)

        if df.badfile:
            continue

        if len(other_axes):
            df.load_other_data()

        df.calibrate_stage_position()
        
        df.high_pass_filter(fc=1)
        #df.detrend_poly()

        #plt.figure()
        #plt.plot(df.pos_data[0])
        #plt.show()

        freqs = np.fft.rfftfreq(len(df.pos_data[0]), d=1.0/df.fsamp)

        df.diagonalize(maxfreq=lpf, interpolate=False)

        if fil_ind == 0:
            drivepsd = np.abs(np.fft.rfft(df.cant_data[drive_ax]))
            driveind = np.argmax(drivepsd[1:]) + 1
            drive_freq = freqs[driveind]

        for axind, ax in enumerate(data_axes):

            try:
                fac = df.conv_facs[ax]
            except:
                fac = 1.0
            if fullNFFT:
                NFFT = len(df.pos_data[ax])
            else:
                NFFT = userNFFT

            outvec1.append(df.pos_data[ax] * fac)
            if ax == 0:
                freqs = np.fft.rfftfreq(len(df.pos_data[ax]), d=1.0/df.fsamp)
                outvec2.append(freqs)
            fft = np.fft.rfft(df.pos_data[ax]*fac)
            outvec2.append(fft)

            psd, freqs = mlab.psd(df.pos_data[ax], Fs=df.fsamp, \
                                  NFFT=NFFT, window=window)

            if diag:
                dpsd, dfreqs = mlab.psd(df.diag_pos_data[ax], Fs=df.fsamp, \
                                        NFFT=NFFT, window=window)
                daxarr[axind,0].loglog(freqs, np.sqrt(psd) * fac, color=color)
                daxarr[axind,0].grid(alpha=0.5)
                daxarr[axind,1].loglog(freqs, np.sqrt(dpsd), color=color)
                daxarr[axind,1].grid(alpha=0.5)
                daxarr[axind,0].set_ylabel('sqrt(PSD) [N/rt(Hz)]', fontsize=10)
                if ax == data_axes[-1]:
                    daxarr[axind,0].set_xlabel('Frequency [Hz]', fontsize=10)
                    daxarr[axind,1].set_xlabel('Frequency [Hz]', fontsize=10)
            else:
                if len(labels):
                    try:
                        lab = labels[fil_ind]
                    except:
                        lab = ''
                else:
                    lab = ''
                daxarr[axind].loglog(freqs, np.sqrt(psd) * fac, \
                                     label=lab, color=color)
                daxarr[axind].grid(alpha=0.5)
                daxarr[axind].set_ylabel('sqrt(PSD) [N/rt(Hz)]', fontsize=10)
                if ax == data_axes[-1]:
                    daxarr[axind].set_xlabel('Frequency [Hz]', fontsize=10)

        if len(cant_axes):
            for axind, ax in enumerate(cant_axes):
                psd, freqs = mlab.psd(df.cant_data[ax], Fs=df.fsamp, \
                                      NFFT=NFFT, window=window)
                caxarr[axind].loglog(freqs, np.sqrt(psd), color=color )

        if len(elec_axes):
            for axind, ax in enumerate(elec_axes):
                psd, freqs = mlab.psd(df.electrode_data[ax], Fs=df.fsamp, \
                                      NFFT=NFFT, window=window)
                eaxarr[axind].loglog(freqs, np.sqrt(psd), color=color ) 

        if len(other_axes):
            for axind, ax in enumerate(other_axes):
                ax = ax - 3
                psd, freqs = mlab.psd(df.other_data[ax], Fs=df.fsamp, \
                                      NFFT=NFFT, window=window)
                oaxarr[axind].loglog(freqs, np.sqrt(psd), color=color )


    daxarr[0].legend(loc=0)
    daxarr[0].set_xlim(0.5, 25000)
    if len(ylim):
        daxarr[0].set_ylim(ylim[0], ylim[1])
    plt.tight_layout()


    if savefigs:
        plt.savefig(title_pre + '.png')

        daxarr[0].set_xlim(2000, 25000)
        plt.tight_layout()

        plt.savefig(title_pre + '_zoomhf.png')

        daxarr[0].set_xlim(1, 80)
        plt.tight_layout()

        plt.savefig(title_pre + '_zoomlf.png')

        daxarr[0].set_xlim(0.5, 25000)
    
    if not savefigs:
        plt.show()


if use_dir:
    allfiles = bu.find_all_fnames(dir1)

allfiles = allfiles[:maxfiles]

plot_many_spectra(allfiles, file_inds=file_inds, diag=diag, \
                  data_axes=data_axes, other_axes=other_axes)
