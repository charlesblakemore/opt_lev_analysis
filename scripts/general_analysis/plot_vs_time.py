import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config


# dir1 = '/data/20180927/bead1/lower1'
# dir1 = '/data/new_trap/20230127/fiberPhaseNoiseEstimate_2/'
dir1 = '/data/new_trap/20230123/Bead0/Shaking/0/'

step10 = False
invert_order = False

data_axes = [0,1,2]

maxfiles = 1000 # Many more than necessary
lpf = 2500   # Hz

# file_inds = (0, 500)
file_inds = (4, 11)

userNFFT = 2**12
diag = False


fullNFFT = False

sharex = True
sharey = False

###########################################################



def plot_vs_time(files, data_axes=[0,1,2], cant_axes=[], elec_axes=[], \
                      diag=True, colormap='inferno', \
                      sort='time', file_inds=(0,10000)):
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
        dfig, daxarr = plt.subplots(len(data_axes),2,\
                                    sharex=sharex,sharey=sharey, \
                                    figsize=(8,8))
    else:
        dfig, daxarr = plt.subplots(len(data_axes),1,\
                                    sharex=sharex,sharey=sharey, \
                                    figsize=(8,8))


    if len(cant_axes):
        cfig, caxarr = plt.subplots(len(data_axes),1,\
                                    sharex=sharex,sharey=True)
    if len(elec_axes):
        efig, eaxarr = plt.subplots(len(data_axes),1,\
                                    sharex=sharex,sharey=True)


    files = [(os.stat(path), path) for path in files]
    files = [(stat.st_ctime, path) for stat, path in files]
    files.sort(key = lambda x: (x[0]))
    files = [obj[1] for obj in files]

    files = files[file_inds[0]:file_inds[1]]
    if step10:
        files = files[::10]
    if invert_order:
        files = files[::-1]

    colors = bu.get_colormap(len(files), cmap=colormap)

    old_per = 0
    print("Processing %i files..." % len(files))
    for fil_ind, fil in enumerate(files):

        color = colors[fil_ind]

        # Load data
        df = bu.DataFile()
        if 'new_trap' in fil:
            df.load_new(fil)
        else:
            df.load(fil)

        df.calibrate_stage_position()
        
        df.calibrate_phase()

        #df.high_pass_filter(fc=1)
        #df.detrend_poly()

        # plt.figure()
        # # #plt.plot((df.daqmx_time-df.daqmx_time[0])*1e-9, \
        # # #         (df.pos_data[2]-np.mean(df.pos_data[2])) * (2.0**3 / 100.0))
        # # plt.plot((df.pos_time-df.pos_time[0])*1e-9, \
        # #          (df.phase[4] - df.phase[4][0]))
        # plt.plot((df.pos_time-df.pos_time[0])*1e-9)

        # # print(type(df.pos_data))

        # # for i in range(len(df.pos_data)):
        # #     print(df.pos_data[i].shape)
        # # for i in range(len(df.amp)):
        # #     print(df.amp[i].shape)
        # # for i in range(len(df.phase)):
        # #     print(df.phase[i].shape)

        # plt.show()
        # input()
        continue


        # df.diagonalize(maxfreq=lpf)
        df.conv_facs = [1.0, 1.0, 1.0]

        loaded_other = False
        for ax in data_axes:
            if ax > 2 and not loaded_other:
                df.load_other_data()

        for axind, ax in enumerate(data_axes):
        
            if ax <= 2:
                data = df.pos_data[ax]
                fac = df.conv_facs[ax]
            if ax > 2:
                data = df.other_data[ax-3]
                fac = 1.0

            t = np.arange(len(data)) * (1.0 / df.fsamp)

            if diag:

                daxarr[axind,0].plot(t, data * fac, color=color)
                daxarr[axind,0].grid(alpha=0.5)
                daxarr[axind,1].plot(t, data, color = color)
                daxarr[axind,1].grid(alpha=0.5)
                daxarr[axind,0].set_ylabel('[N]', fontsize=10)
                if ax == data_axes[-1]:
                    daxarr[axind,0].set_xlabel('t [s]', fontsize=10)
                    daxarr[axind,1].set_xlabel('t [s]', fontsize=10)
            else:
                daxarr[axind].plot(t, data * fac, color=color)
                daxarr[axind].grid(alpha=0.5)
                daxarr[axind].set_ylabel('[N]', fontsize=10)
                if ax == data_axes[-1]:
                    daxarr[axind].set_xlabel('t [s]', fontsize=10)

        if len(cant_axes):
            for axind, ax in enumerate(cant_axes):
                
                t = np.arange(len(df.cant_data[ax])) * (1.0 / df.fsamp)
                caxarr[axind].plot(t, df.cant_data[ax], color=color )

        if len(elec_axes):
            for axind, ax in enumerate(elec_axes): 

                t = np.arange(len(df.electrode_data[ax])) * (1.0 / df.fsamp)
                eaxarr[axind].plot(t, df.electrode_data[ax], color=color )

    #daxarr[0].set_xlim(0.5, 25000)
    #daxarr[0].set_ylim(1e-21, 1e-14)
    plt.tight_layout()


    plt.show()



allfiles, _ = bu.find_all_fnames(dir1, sort_by_index=True)

plot_vs_time(allfiles, data_axes=data_axes, file_inds=file_inds, diag=diag)
