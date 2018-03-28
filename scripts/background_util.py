import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config


###########################################################




class Background:
    '''Class to hold information about the background signal for 
    a single height/separation combination. Stores the complex-value
    of the FFT for the response at the fundamental frequency of the 
    cantilever drive, as well as various harmonics.

    Also estimates and tracks errors based on the median of the
    ASD in bins surrounding the fundamental/harmonics.

    Uses bead_util.Datafile() objects to analyze individual files
    then collects aggregate information.'''


    def __init__(self, files):
        self.allfiles = files
        self.axvecs = 'Stage positions (for all files) not loaded'
        self.freqs = 'Freqs not loaded'
        self.ginds = 'Harmonic indices of freqs array not loaded'
        self.amps = 'Background amplitudes not loaded'
        self.phases = 'Background phases not loaded'
        self.amp_errs = 'Background amplitude errors not loaded'
        self.phase_errs = 'Background phase errors not loaded'


    def find_stage_positions(self):
        '''Loops over a list of file names, loads the attributes of each file, 
           then extracts the DC stage position to sort through data.'''

        axvecs = [{}, {}, {}]
        for fil_ind, fil in enumerate(self.allfiles):

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

        self.axvecs = axvecs



    def select_by_position(self, ax0val=None, ax1val=None, ax2val=None):
        '''Loops over DC stage positions in axvecs, finds all files at a 
           at a single DC position for each axis, then finds the intersection
           of these file lists.'''

        if not len(self.allfiles):
            print "No files!"
            return
        else:
            print "Looping over files and selecting by DC stage position..."
    
        if type(self.axvecs) == str:
            self.find_stage_positions()

        if ax0val is not None:
            keys = self.axvecs[0].keys()
            ax1key = keys[np.argmin(np.abs(np.array(keys)-ax0val))]
            ax1fils = self.axvecs[0][ax1key]
        else:
            ax1fils = []
    
        if ax1val is not None:
            keys = self.axvecs[1].keys()
            ax2key = keys[np.argmin(np.abs(np.array(keys)-ax1val))]
            ax2fils = self.axvecs[1][ax2key]
        else:
            ax2fils = []
    
        if ax2val is not None:
            keys = self.axvecs[2].keys()
            ax3key = keys[np.argmin(np.abs(np.array(keys)-ax2val))]
            ax3fils = self.axvecs[2][ax3key]
        else:
            ax3fils = []

        if (ax0val is None) and (ax1val is None) and (ax2val is None):
            self.relevant_files = self.allfiles
        else:
            self.relevant_files = bu.find_common_filnames(ax1fils, ax2fils, ax3fils)




    def analyze_background(self, data_axes=[0,1,2], lpf=2500, \
                           diag=False, colormap='jet', \
                           file_inds=(0,10000), unwrap=False, \
                           harms_to_track = [1, 2, 3], \
                           ext_cant_drive=False, ext_cant_ind=0, \
                           plot_first_drive=False, sub_cant_phase=True):
        '''Loops over a list of file names, loads each file, diagonalizes,
           then plots the amplitude spectral density of any number of data
           or cantilever/electrode drive signals

           INPUTS: files, list of files names to extract data
                   data_axes, list of pos_data axes to plot
                   ax_labs, dict with labels for plotted axes
                   diag, bool specifying whether to diagonalize
                   unwrap, bool to unwrap phase of background
                   harms, harmonics to label in ASD

           OUTPUTS: none, generates class attributes
        '''

        files = bu.sort_files_by_timestamp(self.relevant_files)
        files = files[file_inds[0]:file_inds[1]]

        nfreq = len(harms_to_track)
        nax = len(data_axes)
        nfiles = len(files)
    
        colors = bu.get_color_map(nfiles, cmap=colormap)

        avg_asd = [[]] * nax
        Nasds = [[]] * nax

        amps = np.zeros((nax, nfreq, nfiles))
        amp_errs = np.zeros((nax, nfreq, nfiles))
        phases = np.zeros((nax, nfreq, nfiles))
        phase_errs = np.zeros((nax, nfreq, nfiles))

        print "Processing %i files..." % nfiles
        for fil_ind, fil in enumerate(files):
            color = colors[fil_ind]

            # Display percent completion
            bu.progress_bar(fil_ind, nfiles)

            # Load data
            df = bu.DataFile()
            df.load(fil)
            if df.badfile:
                continue

            df.calibrate_stage_position()

            #df.high_pass_filter(fc=1)
            #df.detrend_poly()

            df.diagonalize(maxfreq=lpf, interpolate=False)

            Nsamp = len(df.pos_data[0])

            if len(harms_to_track):
                harms = harms_to_track
            else:
                harms = [1]

            ginds, driveind, drive_freq, drive_ax = \
                        df.get_boolean_cantfilt(ext_cant_drive=ext_cant_drive, \
                                                ext_cant_ind=ext_cant_ind, \
                                                nharmonics=10, harms=harms, width=0)

            if fil_ind == 0:
                if plot_first_drive:
                    df.plot_cant_asd(drive_ax)
                freqs = np.fft.rfftfreq(Nsamp, d=1.0/df.fsamp)
                bin_sp = freqs[1] - freqs[0]
                dt = 1.0/df.fsamp
                tint = dt * len(df.pos_data[0]) + 0.5

            datfft, diagdatfft, daterr, diagdaterr = \
                         df.get_datffts_and_errs(ginds, drive_freq, plot=False)

            harm_freqs = freqs[ginds]
            for freqind, freq in enumerate(harm_freqs):
                for axind, ax in enumerate(data_axes):
                    phase = np.angle(datfft[axind][freqind])
                    if sub_cant_phase:
                        cantfft = np.fft.rfft(df.cant_data[drive_ax])
                        cantphase = np.angle(cantfft[driveind])
                        phases[axind][freqind][fil_ind] = phase - cantphase
                    else:
                        phases[axind][freqind][fil_ind] = phase

                    sig_re = daterr[axind][freqind] / np.sqrt(2)
                    sig_im = np.copy(sig_re)

                    im = np.imag(datfft[axind][freqind])
                    re = np.real(datfft[axind][freqind])

                    phase_var = np.mean((im**2 * sig_re**2 + re**2 * sig_im**2) / \
                                        (re**2 + im**2)**2)
                    phase_errs[axind][freqind][fil_ind] = np.sqrt(phase_var)

                    amps[axind][freqind][fil_ind] = np.abs(datfft[axind][freqind] * \
                                                           np.sqrt(bin_sp) * \
                                                           bu.fft_norm(Nsamp, df.fsamp))
                    amp_errs[axind][freqind][fil_ind] = daterr[axind][freqind] * \
                                                        np.sqrt(bin_sp) * \
                                                        bu.fft_norm(Nsamp, df.fsamp)

                    asd = np.abs( np.fft.rfft(df.pos_data[ax]) ) * \
                            bu.fft_norm(Nsamp, df.fsamp) * df.conv_facs[ax]
                    if not len(avg_asd[axind]):
                        avg_asd[axind] = asd
                        Nasds[axind] = 1
                    else:
                        avg_asd[axind] += asd
                        Nasds[axind] += 1

        for axind, ax in enumerate(data_axes):
            avg_asd[axind] *= (1.0 / Nasds[axind])

        self.freqs = freqs
        self.ginds = ginds
        self.avg_asd = avg_asd
        self.amps = amps
        self.phases = phases
        self.amp_errs = amp_errs
        self.phase_errs = phase_errs
        self.tint = tint



    def plot_background(self, harms_to_plot=[1,2,3], harms_to_label=[1,2,3], \
                        data_axes=[0,1,2], ax_labs = {0: 'X', 1: 'Y', 2: 'Z'}, \
                        ylim=(), arrow_fac=5, unwrap=False):
        '''Plots the output from the analyze background method

           INPUTS: 

           OUTPUTS: none, plots stuff
        '''

        harm_freqs = self.freqs[self.ginds]
        drive_freq = harm_freqs[0]

        avgfig, avgaxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True, \
                                        figsize=(8,8))

        ampfigs = []
        ampaxarrs = []
        phasefigs = []
        phaseaxarrs = []
        for harm in harms_to_plot:
            ampfig, ampaxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True, \
                                            figsize=(8,8))
            phasefig, phaseaxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True, \
                                                figsize=(8,8))
            ampfigs.append(ampfig)
            ampaxarrs.append(ampaxarr)
            phasefigs.append(phasefig)
            phaseaxarrs.append(phaseaxarr)

        if unwrap:
            phasefigs2 = []
            phaseaxarrs2 = []
            for harm in harms_to_plot:
                phasefig2, phaseaxarr2 = plt.subplots(len(data_axes),1,sharex=True,sharey=True, \
                                                      figsize=(8,8))
                phasefigs2.append(phasefig2)
                phaseaxarrs2.append(phaseaxarr2)

        for axind, ax in enumerate(data_axes):
            avgaxarr[axind].loglog(self.freqs, self.avg_asd[axind])
            avgaxarr[axind].grid(alpha=0.5)
            lab = ax_labs[ax] + ' sqrt(PSD) [N/rt(Hz)]'
            avgaxarr[axind].set_ylabel(lab, fontsize=10)
            if ax == data_axes[-1]:
                avgaxarr[axind].set_xlabel('Frequency [Hz]', fontsize=10)

            arrow_tip = (drive_freq, \
                         self.avg_asd[axind][np.argmin(np.abs(self.freqs-drive_freq))]*1.2)
            text = (drive_freq, arrow_tip[1]*arrow_fac)

            avgaxarr[axind].annotate('$f_0$', xy=arrow_tip, xytext=text, \
                                     arrowprops=dict(facecolor='red', shrink=0.01, \
                                                     width=3, headwidth=6), \
                                     horizontalalignment='center')

            for harm in harms_to_label:
                if harm == 1:
                    continue
                arrow_tip = (harm * drive_freq, \
                             self.avg_asd[axind][np.argmin(np.abs(self.freqs-harm*drive_freq))]*1.2)
                text = (harm * drive_freq, arrow_tip[1]*arrow_fac)
                text_str = '%i$f_0$' % harm
                avgaxarr[axind].annotate(text_str, xy=arrow_tip, xytext=text, \
                                         arrowprops=dict(facecolor='red', shrink=0.01, \
                                                         width=3, headwidth=6), \
                                         horizontalalignment='center')
        plt.tight_layout()

        inds = np.array(range(len(self.phases[0][0]))) * self.tint
        for harmind, harm in enumerate(harms_to_plot):
            if harm == 1:
                title = "Fundamental: $f_0$"
            else:
                title = "Harmonic: %i $f_0$" % harm
            ampaxarrs[harmind][0].set_title(title, fontsize=18)
            phaseaxarrs[harmind][0].set_title(title, fontsize=18)
            if unwrap:
                phaseaxarrs2[harmind][0].set_title(title, fontsize=18)

            for axind, ax in enumerate(data_axes):
                lab = ax_labs[ax] + ' Phase [rad]'
                if unwrap:
                    phaseaxarrs2[harmind][axind].errorbar(inds, \
                                                          np.unwrap(self.phases[axind][harmind]), \
                                                          self.phase_errs[axind][harmind], \
                                                          fmt='-', marker='.', ms=7, capsize=3)
                    phaseaxarrs2[harmind][axind].set_ylabel(lab, fontsize=10)

                neg_inds = np.array(self.phases[axind][harmind]) < -2.5
                #okay_inds = np.array(phases[axind]) > -2.5

                plotphases = np.array(self.phases[axind][harmind]) + 2.0 * np.pi * neg_inds
                phaseaxarrs[harmind][axind].errorbar(inds, plotphases, \
                                                     self.phase_errs[axind][harmind], \
                                                     fmt='-', marker='.', ms=7, capsize=3)

                phaseaxarrs[harmind][axind].set_ylabel(lab, fontsize=10)
                if ax == data_axes[-1]:
                    phaseaxarrs[harmind][axind].set_xlabel('Time [s]', fontsize=10)
                    if unwrap:
                        phaseaxarrs2[harmind][axind].set_xlabel('Time [s]', fontsize=10)
            plt.tight_layout()

            for axind, ax in enumerate(data_axes):
                ampaxarrs[harmind][axind].errorbar(inds, self.amps[axind][harmind], \
                                                   self.amp_errs[axind][harmind], \
                                                   fmt='-', marker='.', ms=7, capsize=3)
                lab = ax_labs[ax] + ' Amp. [N]'
                ampaxarrs[harmind][axind].set_ylabel(lab, fontsize=10)
                if ax == data_axes[-1]:
                    ampaxarrs[harmind][axind].set_xlabel('Time [s]', fontsize=10)
            plt.tight_layout()

        plt.show()


