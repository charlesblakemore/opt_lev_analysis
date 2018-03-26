import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

cbead = '/data/20180314/bead1'

#dir1 = cbead + '/grav_data/ydrive_1sep_1height_1V-1300Hz_shieldin_1V-cant'
dir1 = cbead + '/grav_data/ydrive_1sep_1height_2V-2200Hz_shield_0mV-cant'

dir1 = cbead + '/grav_data/ydrive_3sep_1height_2V-2200Hz_shieldin_0V-cant'


#dir1 = '/data/20180314/bead1/grav_data/xdrive_3height_5Vac-1198Hz'
#dir1 = '/data/20180314/bead1/grav_data/xdrive_1height_nofield_shieldin'

#dir1 = '/data/20180308/bead2/grav_data/onepos_long'


track_phase = True
unwrap = True
build_avg = True
label_drive = True
arrow_fac = 5
drive_ax = 1

sub_cant_phase = True
plot_first_drive = True

ax1val = 60   # um
ax2val = None   # um
ax3val = None   # um

#ylim = (1e-21, 1e-14)
#ylim = (1e-7, 1e-1)
ylim = ()

lpf = 2500   # Hz

file_inds = (0, 3000)

userNFFT = 2**12
diag = False


fullNFFT = True

###########################################################



def find_stage_positions(files):
    '''Loops over a list of file names, loads the attributes of each file, 
       then extracts the DC stage position to sort through data.'''

    axvecs = [{}, {}, {}]
    for fil_ind, fil in enumerate(files):

        df = bu.DataFile()
        df.load_only_attribs(fil)

        if df.badfile:
            continue

        df.calibrate_stage_position()

        for axind, axstr in enumerate(['x', 'y', 'z']):
            axpos = df.stage_settings[axstr + ' DC']
            if axpos not in axvecs[axind].keys():
                axvecs[axind][axpos] = []
            axvecs[axind][axpos].append(fil)

    return axvecs



def select_by_position(files, ax1val=None, ax2val=None, ax3val=None):
    '''Loops over DC stage positions in axvecs, finds all files at a 
       at a single DC position for each axis, then finds the intersection
       of these file lists.'''

    axvecs = find_stage_positions(files)

    if ax1val is not None:
        keys = axvecs[0].keys()
        ax1key = keys[np.argmin(np.abs(np.array(keys)-ax1val))]
        ax1fils = axvecs[0][ax1key]
    else:
        ax1fils = []
    if ax2val is not None:
        keys = axvecs[1].keys()
        ax2key = keys[np.argmin(np.abs(np.array(keys)-ax2val))]
        ax2fils = axvecs[1][ax2key]
    else:
        ax2fils = []
    if ax3val is not None:
        keys = axvecs[2].keys()
        ax3key = keys[np.argmin(np.abs(np.array(keys)-ax3val))]
        ax3fils = axvecs[2][ax3key]
    else:
        ax3fils = []

    if (ax1val is None) and (ax2val is None) and (ax3val is None):
        return files
    else:
        return bu.find_common_filnames(ax1fils, ax2fils, ax3fils)




def analyze_background(files, data_axes=[0,1,2], other_axes=[], \
                       ax_labs = {0: 'X', 1: 'Y', 2: 'Z'},\
                       diag=True, colormap='jet', sort='time', \
                       file_inds=(0,10000), unwrap=False, \
                       harms = [2, 3], drive_ax=1):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then plots the amplitude spectral density of any number of data
       or cantilever/electrode drive signals

       INPUTS: files, list of files names to extract data
               data_axes, list of pos_data axes to plot
               ax_labs, dict with labels for plotted axes
               diag, bool specifying whether to diagonalize
               unwrap, bool to unwrap phase of background
               harms, harmonics to label in ASD

       OUTPUTS: none, plots stuff
    '''

    files = bu.sort_files_by_timestamp(files)
    files = files[file_inds[0]:file_inds[1]]
    files = select_by_position(files, ax1val=ax1val, ax2val=ax2val, ax3val=ax3val)

    colors = bu.get_color_map(len(files), cmap=colormap)
    
    avg_psd = [[]] * len(data_axes)
    Npsds = [[]] * len(data_axes)
    avg_facs = [0] * len(data_axes)

    amps = [[], [], []]
    amp_errs = [[], [], []]
    phases = [[], [], []]
    phase_errs = [[], [], []]

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
        
        #df.high_pass_filter(fc=1)
        #df.detrend_poly()

        df.diagonalize(maxfreq=lpf, interpolate=False)

        Nsamp = len(df.pos_data[0])

        if fil_ind == 0:
            if plot_first_drive:
                df.plot_cant_asd(drive_ax)
            driveind = np.argmax(drivepsd[1:]) + 1
            drive_freq = fftfreqs[driveind]
            dt = 1.0/df.fsamp
            tint = dt * len(df.pos_data[0]) + 0.5

        for axind, ax in enumerate(data_axes):

            try:
                fac = df.conv_facs[ax]
            except:
                fac = 1.0
            if fullNFFT:
                NFFT = Nsamp
            else:
                NFFT = userNFFT

            freqs = np.fft.rfftfreqs(Nsamp, d=1.0/df.fsamp)
            bin_sp = freqs[1] - freqs[0]
            
            fft = np.fft.rfft(df.pos_data[ax])
            cantfft = np.fft.rfft(df.cant_data[drive_ax])

            asd = np.abs(fft) * bu.fft_norm(Nsamp, df.fsamp)

            if sub_cant_phase:
                phases[axind].append(np.angle(fft[driveind]) - \
                                     np.angle(cantfft[driveind]))
            else:
                phases[axind].append(np.angle(fft[driveind]))

            errinds = (freqs > freqs[driveind-25]) * (freqs < freqs[driveind+25])
            errinds[driveind] = False

            sig_re = np.mean(asd[errinds] * bin_sp / np.sqrt(2))
            sig_im = np.copy(sig_re)

            normfft = fft * bu.fft_norm(Nsamp, df.fsamp)

            im = np.imag(normfft[driveind]) * fac * np.sqrt(bin_sp)
            re = np.real(normfft[driveind]) * fac * np.sqrt(bin_sp)

            phase_var = np.mean((im**2 * sig_re**2 + re**2 * sig_im**2) / \
                                (re**2 + im**2)**2)
            phase_errs[axind].append(np.sqrt(phase_var))
            
            amps[axind].append(np.sqrt(psd[driveind]*bin_sp)*fac)
            err_est = np.median(np.sqrt(psd[errinds]*bin_sp)*fac)
            amp_errs[axind].append(err_est)

            if not len(avg_psd[axind]):
                avg_psd[axind] = psd
                Npsds[axind] = 1
                avg_facs[axind] = fac
            else:
                avg_psd[axind] += psd
                Npsds[axind] += 1

    avgfig, avgaxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True, \
                                    figsize=(8,8))
    ampfig, ampaxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True, \
                                        figsize=(8,8))
    phasefig, phaseaxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True, \
                                        figsize=(8,8))
    if unwrap:
        phasefig2, phaseaxarr2 = plt.subplots(len(data_axes),1,sharex=True,sharey=True, \
                                              figsize=(8,8))

    for axind, ax in enumerate(data_axes):
        plotasd = np.sqrt(avg_psd[axind]/Npsds[axind])*avg_facs[axind]
        avgaxarr[axind].loglog(freqs, plotasd)
        avgaxarr[axind].grid(alpha=0.5)
        lab = ax_labs[ax] + ' sqrt(PSD) [N/rt(Hz)]'
        avgaxarr[axind].set_ylabel(lab, fontsize=10)
        if ax == data_axes[-1]:
            avgaxarr[axind].set_xlabel('Frequency [Hz]', fontsize=10)

        arrow_tip = (drive_freq, plotasd[np.argmin(np.abs(freqs-drive_freq))]*1.2)
        text = (drive_freq, arrow_tip[1]*arrow_fac)
            
        avgaxarr[axind].annotate('$f_0$', xy=arrow_tip, xytext=text, \
                                 arrowprops=dict(facecolor='red', shrink=0.01, \
                                                 width=3, headwidth=6), \
                                 horizontalalignment='center')

        for harm in harms:
            arrow_tip = (harm * drive_freq, \
                         plotasd[np.argmin(np.abs(freqs-harm*drive_freq))]*1.2)
            text = (harm * drive_freq, arrow_tip[1]*arrow_fac)
            text_str = '%i$f_0$' % harm
            avgaxarr[axind].annotate(text_str, xy=arrow_tip, xytext=text, \
                                     arrowprops=dict(facecolor='red', shrink=0.01, \
                                                     width=3, headwidth=6), \
                                     horizontalalignment='center')
    plt.tight_layout()

    inds = np.array(range(len(phases[0]))) * tint
    for axind, ax in enumerate(data_axes):
        lab = ax_labs[ax] + ' Phase of Fund. [rad]'
        if unwrap:
            phaseaxarr2[axind].errorbar(inds, np.unwrap(phases[axind]), \
                                       phase_errs[axind], \
                                       fmt='-', marker='.', ms=7, capsize=3)
            phaseaxarr2[axind].set_ylabel(lab, fontsize=10)

        phaseaxarr[axind].errorbar(inds, phases[axind], phase_errs[axind], \
                                       fmt='-', marker='.', ms=7, capsize=3)
        
        phaseaxarr[axind].set_ylabel(lab, fontsize=10)
        if ax == data_axes[-1]:
            phaseaxarr[axind].set_xlabel('Time [s]', fontsize=10)
            if unwrap:
                phaseaxarr2[axind].set_xlabel('Time [s]', fontsize=10)
    plt.tight_layout()

    inds = np.array(range(len(amps[0]))) * tint
    for axind, ax in enumerate(data_axes):
        ampaxarr[axind].errorbar(inds, amps[axind], amp_errs[axind], \
                                 fmt='-', marker='.', ms=7, capsize=3)
        lab = ax_labs[ax] + ' Amp. of Fund. [N]'
        ampaxarr[axind].set_ylabel(lab, fontsize=10)
        if ax == data_axes[-1]:
            ampaxarr[axind].set_xlabel('Time [s]', fontsize=10)
    plt.tight_layout()

    plt.show()

allfiles = bu.find_all_fnames(dir1)

analyze_background(allfiles, file_inds=file_inds, diag=diag, \
                   data_axes=[0,1,2], other_axes=[], \
                   unwrap=unwrap, drive_ax=drive_ax)
