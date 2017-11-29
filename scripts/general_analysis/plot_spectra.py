import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config



dir1 = '/data/20171106/bead1/response_vs_acvoltage_ext/'
maxfiles = 10000 # Many more than necessary
lpf = 800   # Hz

NFFT = 2**13


###########################################################



def plot_many_spectra(files, data_axes=[0,1,2], cant_axes=[], elec_axes=[], \
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
        dfig, daxarr = plt.subplots(len(data_axes),2,sharex=True,sharey=True)
    else:
        dfig, daxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True)


    if len(cant_axes):
        cfig, caxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True)
    if len(elec_axes):
        efig, eaxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True)


    files = [(os.stat(path), path) for path in files]
    files = [(stat.st_ctime, path) for stat, path in files]
    files.sort(key = lambda x: (x[0]))
    files = [obj[1] for obj in files]

    files = files[file_inds[0]:file_inds[1]]

    colors = bu.get_color_map(len(files), cmap=colormap)

    old_per = 0
    print "Percent complete: "
    for fil_ind, fil in enumerate(files):
        color = colors[fil_ind]
        
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
        
        df.high_pass_filter(fc=5)

        freqs = np.fft.rfftfreq(len(df.pos_data[0]), d=1.0/df.fsamp)

        if diag:
            df.diagonalize(maxfreq=lpf, interpolate=True)

        for axind, ax in enumerate(data_axes):
            psd, freqs = mlab.psd(df.pos_data[ax], Fs=df.fsamp, NFFT=NFFT)
            if diag:
                fac = df.conv_facs[ax]
                dpsd, dfreqs = mlab.psd(df.diag_pos_data[ax], Fs=df.fsamp, NFFT=NFFT)
                daxarr[axind,0].loglog(freqs, np.sqrt(psd) * fac, color=color)
                daxarr[axind,1].loglog(freqs, np.sqrt(dpsd), color=color)
            else:
                daxarr[axind].loglog(freqs, np.sqrt(psd), color=color)

        if len(cant_axes):
            for axind, ax in enumerate(cant_axes):
                psd, freqs = mlab.psd(fd.cant_data[ax], Fs=df.fsamp, NFFT=NFFT)
                caxarr[axind].loglog(freqs, np.sqrt(psd) )

        if len(elec_axes):
            for axind, ax in enumerate(elec_axes):
                psd, freqs = mlab.psd(fd.electrode_data[ax], Fs=df.fsamp, NFFT=NFFT)
                eaxarr[axind].loglog(freqs, np.sqrt(psd) ) 

    plt.show()



allfiles = bu.find_all_fnames(dir1)

plot_many_spectra(allfiles, file_inds=(0,20))
