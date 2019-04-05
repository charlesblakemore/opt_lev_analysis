import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config
import transfer_func_util as tf


dir1 = '/data/20180927/bead1/tf_20180928/elec1'
q_bead = -18.0 * (1.602e-19)


step10 = False
invert_order = False

data_axes = [0,1,2]

maxfiles = 1000 # Many more than necessary
lpf = 2500   # Hz

file_inds = (0, 500)

userNFFT = 2**12
diag = False


fullNFFT = False

###########################################################


def harmonic_osc(f, d_accel, f0, gamma):
    omega = 2.0 * np.pi * f
    omega0 = 2.0 * np.pi * f0
    return d_accel / ((omega0**2 - omega**2) + 1.0j * gamma * omega)




def weigh_bead(files, colormap='jet', sort='time', file_inds=(0,10000)):
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

    files = [(os.stat(path), path) for path in files]
    files = [(stat.st_ctime, path) for stat, path in files]
    files.sort(key = lambda x: (x[0]))
    files = [obj[1] for obj in files]

    files = files[file_inds[0]:file_inds[1]]
    if step10:
        files = files[::10]
    if invert_order:
        files = files[::-1]

    nfiles = len(files)
    colors = bu.get_color_map(nfiles, cmap=colormap)

    avg_fft = []

    print "Processing %i files..." % nfiles
    for fil_ind, fil in enumerate(files):
        color = colors[fil_ind]

        bu.progress_bar(fil_ind, nfiles)

        # Load data
        df = bu.DataFile()
        df.load(fil)

        df.calibrate_stage_position()
        
        df.calibrate_phase()

        #plt.hist( df.zcal / df.phase[4] )
        #plt.show()

        freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)
        fft = np.fft.rfft(df.zcal) * bu.fft_norm(df.nsamp, df.fsamp) \
              * np.sqrt(freqs[1] - freqs[0])
        
        drive_fft = np.fft.rfft(df.electrode_data[1])

        inds = np.abs(drive_fft) > 1e4
        inds *= (freqs > 2.0) * (freqs < 300.0)

        drive_amp = np.abs( drive_fft[inds][0] * bu.fft_norm(df.nsamp, df.fsamp) \
                            * np.sqrt(freqs[1] - freqs[0]) )

        if not len(avg_fft):
            avg_fft = fft 
            avg_drive_fft = drive_fft
            
            ratio = fft[inds] / drive_fft[inds]
        else:
            avg_fft += fft
            avg_drive_fft += drive_fft

            ratio += fft[inds] / drive_fft[inds]


    avg_fft *= (1.0 / nfiles)
    avg_drive_fft *= (1.0 / nfiles)

    resp = avg_fft[inds] * (1064.0e-9 / 2.0) * (1.0 / (2.0 * np.pi))


    resp_sc = resp * 1e9   # put resp in units of nm


    def amp_sc(f, d_accel, f0, g):
        return np.abs(harmonic_osc(f, d_accel, f0, g)) * 1e9

    def phase_sc(f, d_accel, f0, g):
        return np.angle(harmonic_osc(f, d_accel, f0, g))

    #plt.loglog(freqs[inds], np.abs(resp_sc))
    #plt.loglog(freqs[inds], np.abs(harmonic_osc(freqs[inds], 1e-3, 160, 75e1))*1e9)
    #plt.show()


    popt, pcov = opti.curve_fit(amp_sc, freqs[inds], np.abs(resp_sc), p0=[1e-3, 160, 750])
    #popt2, pcov2 = opti.curve_fit(phase_sc, freqs[inds], np.angle(resp_sc), p0=[1e-3, 160, 750])

    plt.figure()
    plt.loglog(freqs[inds], np.abs(resp))
    plt.loglog(freqs, np.abs(harmonic_osc(freqs, *popt)))

    force = (drive_amp / (4.0e-3)) * q_bead

    print "IMPLIED MASS [ng]: ", np.abs(popt[0]**(-1) * force) * 10**12

    plt.show()





allfiles = bu.find_all_fnames(dir1)

weigh_bead(allfiles)
