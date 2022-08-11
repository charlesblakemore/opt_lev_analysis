import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti
import scipy.constants as constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config
import transfer_func_util as tf




dirs = ['/data/old_trap/20200307/gbead1/tf_20200311/elec3', \
       ]

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




def weigh_bead(files, pcol=0, colormap='plasma', sort='time', file_inds=(0,10000)):
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

    date = re.search(r"\d{8,}", files[0])[0]
    charge_dat = np.load(open('/calibrations/charges/'+date+'.charge', 'rb'))
    q_bead = -1.0 * charge_dat[0] * constants.elementary_charge
    # q_bead = -25.0 * 1.602e-19

    nfiles = len(files)
    colors = bu.get_colormap(nfiles, cmap=colormap)

    avg_fft = []

    print("Processing %i files..." % nfiles)
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

        #print np.mean(df.zcal / df.phase[4]), np.std(df.zcal / df.phase[4])

        freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)
        fft = np.fft.rfft(df.zcal) * bu.fft_norm(df.nsamp, df.fsamp) \
              * np.sqrt(freqs[1] - freqs[0])
        fft2 = np.fft.rfft(df.phase[4]) * bu.fft_norm(df.nsamp, df.fsamp) \
              * np.sqrt(freqs[1] - freqs[0])

        fftd = np.fft.rfft(df.zcal - np.pi*df.phase[4]) * bu.fft_norm(df.nsamp, df.fsamp) \
               * np.sqrt(freqs[1] - freqs[0])

        #plt.plot(np.pi * df.phase[4])
        #plt.plot(df.zcal)

        #plt.figure()
        #plt.loglog(freqs, np.abs(fft))
        #plt.loglog(freqs, np.pi * np.abs(fft2))
        #plt.loglog(freqs, np.abs(fftd))
        #plt.show()
        
        drive_fft = np.fft.rfft(df.electrode_data[1])

        #plt.figure()
        #plt.loglog(freqs, np.abs(drive_fft))
        #plt.show()

        inds = np.abs(drive_fft) > 1e4
        inds *= (freqs > 2.0) * (freqs < 300.0)
        inds = np.arange(len(inds))[inds]

        ninds = inds + 5

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

    fac = bu.fft_norm(df.nsamp, df.fsamp) * np.sqrt(freqs[1] - freqs[0])

    avg_fft *= (1.0 / nfiles)
    avg_drive_fft *= (1.0 / nfiles)

    resp = fft[inds] * (1064.0e-9 / 2.0) * (1.0 / (2.0 * np.pi))
    noise = fft[ninds] * (1064.0e-9 / 2.0) * (1.0 / (2.0 * np.pi))

    drive_noise = np.abs(np.median(avg_drive_fft[ninds] * fac))

    #plt.loglog(freqs[inds], np.abs(resp))
    #plt.loglog(freqs[ninds], np.abs(noise))
    #plt.show()


    resp_sc = resp * 1e9   # put resp in units of nm
    noise_sc = noise * 1e9

    def amp_sc(f, d_accel, f0, g):
        return np.abs(harmonic_osc(f, d_accel, f0, g)) * 1e9

    def phase_sc(f, d_accel, f0, g):
        return np.angle(harmonic_osc(f, d_accel, f0, g))

    #plt.loglog(freqs[inds], np.abs(resp_sc))
    #plt.loglog(freqs[inds], np.abs(harmonic_osc(freqs[inds], 1e-3, 160, 75e1))*1e9)
    #plt.show()

    #plt.loglog(freqs[inds], np.abs(resp_sc))
    #plt.loglog(freqs, amp_sc(freqs, 1e-3, 160, 750))
    #plt.show()

    popt, pcov = opti.curve_fit(amp_sc, freqs[inds], np.abs(resp_sc), sigma=np.abs(noise_sc), \
                                absolute_sigma=True, p0=[1e-3, 160, 750], maxfev=10000)
    #popt2, pcov2 = opti.curve_fit(phase_sc, freqs[inds], np.angle(resp_sc), p0=[1e-3, 160, 750])

    print(popt)
    print(pcov)

    plt.figure()
    plt.errorbar(freqs[inds], np.abs(resp), np.abs(noise), fmt='.', ms=10, lw=2)
    #plt.loglog(freqs[inds], np.abs(noise))
    plt.loglog(freqs, np.abs(harmonic_osc(freqs, *popt)))
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.ylabel('Z Amplitude [m]', fontsize=16)

    force = (drive_amp / (4.0e-3)) * q_bead

    mass = np.abs(popt[0]**(-1) * force) * 10**12
    fit_err = np.sqrt(pcov[0,0] / popt[0])
    charge_err = 0.1
    drive_err = drive_noise / drive_amp

    print(drive_err)

    mass_err = np.sqrt( (fit_err)**2 + (charge_err)**2 + (drive_err)**2  ) * mass

    #print "IMPLIED MASS [ng]: ", mass

    print('%0.3f ng,  %0.2f e^-,  %0.1f V'  % (mass, q_bead * (1.602e-19)**(-1), drive_amp))
    print('%0.6f ng' % (mass_err))
    plt.tight_layout()

    plt.show()




for dir in dirs:

    allfiles = bu.find_all_fnames(dir)

    weigh_bead(allfiles)
