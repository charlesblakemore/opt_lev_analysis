import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

plt.rcParams.update({'font.size': 14})

# dir1 = '/data/old_trap/20191017/bead1/spinning/junk/shit_test_10'
maxfiles = 500



# dir1 = '/data/old_trap/20191203/daq_tests/nominal'
# dir1 = '/data/old_trap/20191203/daq_tests/new_wiring'

dir1 = '/data/old_trap/20200307/gbead1/junk/elec5/'
dir1 = '/data/old_trap/20200312/beam_profiling/xprof_init/'

dir1 = '/data/old_trap/20200312/beam_profiling/xprof_init/'

dir1 = '/data/old_trap/20230208/bead1/long_monitor/'
dir1 = '/data/old_trap/20230306/bead4/trans_func/20230308/freqcomb_elec3_10V'

dir1 = '/data/old_trap/20230322/bead1/zphase_test'

dir1 = '/data/old_trap/20230327/bead1/zoffset_step_5mbar_down'
dir1 = '/data/old_trap/20230327/bead1/imaging_realignment/lower_hysteresis_point'

dir1 = '/data/old_trap/20230410/bead1/zoffset_sweep/1_5mbar_down'
dir1 = '/data/old_trap/20230410/bead1/pumpdown_2'
dir1 = '/data/old_trap/20230410/bead1/discharge/high_pressure_test'
dir1 = '/data/old_trap/20230410/bead1/discharge/low_pressure_test'
dir1 = '/data/old_trap/20230410/bead1/discharge/low_pressure_test_no_drive'
dir1 = '/data/old_trap/20230410/bead1/discharge/low_pressure_test_no_drive_later'
dir1 = '/data/old_trap/20230410/bead1/discharge/low_pressure_test_no_drive_lessfb'
dir1 = '/data/old_trap/20230410/bead1/spinning/initial_test_slow_DAQ'
dir1 = '/data/old_trap/20230410/bead1/spinning/moderate_spinup_slow_DAQ'
dir1 = '/data/old_trap/20230410/bead1/discharge/spinning_test'

dir1 = '/data/old_trap/20230518/bead1/height_determination'

dir1 = '/data/old_trap/20230531/bead1/zset_step'
dir1 = '/data/old_trap/20230531/bead1/zset_step_with_xdrive'
dir1 = '/data/old_trap/20230531/bead1/spinning/test'


# use_dir = True
use_dir = False

invert_order = False



# allfiles  = [
#              # '/data/old_trap/20230208/bead1/3mbar_nocool.h5',
#              # '/data/old_trap/20230208/bead1/3mbar_nocool_z-5000.h5',
#              # '/data/old_trap/20230208/bead1/3mbar_nocool_with_zsig.h5',
#              '/data/old_trap/20230208/bead1/3mbar_nopowfb_nocool.h5',
#              # '/data/old_trap/20230208/bead1/3mbar_powfb_nocool.h5',
#              # '/data/old_trap/20230208/bead1/3mbar_powfb_zcool_d.h5',
#              '/data/old_trap/20230208/bead1/3mbar_powfb_zcool_pd.h5',
#              # '/data/old_trap/20230208/bead1//long_monitor/3mbar_powfb_zcool_pd_0.h5',
#              '/data/old_trap/20230208/bead1//long_monitor/3mbar_powfb_zcool_pd_92.h5',
#              # '/data/old_trap/20230208/bead1//long_monitor/3mbar_powfb_zcool_pd_93.h5',
#              '/data/old_trap/20230208/bead1//long_monitor/3mbar_powfb_zcool_pd_94.h5',
#             ]


# allfiles  = [
#              # '/data/old_trap/20230221/bead1/powerfb/powerfb.h5',
#              # '/data/old_trap/20230221/bead1/powerfb/powerfb_and_zfb.h5',
#              # '/data/old_trap/20230221/bead1/powerfb/powerfb_and_zfb_2.h5',
#              # '/data/old_trap/20230221/bead1/powerfb/powerfb_and_zfb_3.h5',
#              # '/data/old_trap/20230221/bead1/powerfb/powerfb_and_zfb_4.h5',
#              '/data/old_trap/20230221/bead1/powerfb/powerfb_and_zfb_6.h5',
#              '/data/old_trap/20230221/bead1/powerfb/xfb.h5',
#              # '/data/old_trap/20230221/bead1/powerfb/yfb_4.h5',
#              '/data/old_trap/20230221/bead1/powerfb/yfb_5.h5',
#             ]


# allfiles  = [
#              '/data/old_trap/20230327/bead1/imaging_realignment/initial/5mbar_nocool_0.h5',
#              '/data/old_trap/20230327/bead1/imaging_realignment/initial/5mbar_nocool_1.h5',
#              '/data/old_trap/20230327/bead1/imaging_realignment/adjust1/5mbar_nocool_0.h5',
#              '/data/old_trap/20230327/bead1/imaging_realignment/adjust1/5mbar_nocool_1.h5'
#             ]


# allfiles  = [
#              # '/data/old_trap/20230410/bead1/feedback_tuning_2/1_5mbar_nocool.h5',
#              # '/data/old_trap/20230410/bead1/feedback_tuning_2/1_5mbar_izcool.h5',
#              # '/data/old_trap/20230410/bead1/feedback_tuning_2/1_5mbar_pizcool.h5',
#              # '/data/old_trap/20230410/bead1/feedback_tuning_2/1_5mbar_pidzcool.h5',
#              # '/data/old_trap/20230410/bead1/feedback_tuning_2/1_5mbar_pidzcool_2.h5',
#              # '/data/old_trap/20230410/bead1/feedback_tuning_2/1_5mbar_xzcool.h5',
#              # '/data/old_trap/20230410/bead1/feedback_tuning_2/1_5mbar_yzcool.h5',
#              # '/data/old_trap/20230410/bead1/feedback_tuning_2/1_5mbar_xyzcool.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m0.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m100k.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m200k.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m300k.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m400k.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m500k.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m600k.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m700k.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m700k_2.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m700k_3.h5',
#              # '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m700k_4.h5',
#              '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m700k_morelight.h5',
#              '/data/old_trap/20230410/bead1/lowering/1_5mbar_xyzcool_m800k.h5',
#             ]

# allfiles  = [\
#              # '/data/old_trap/20230531/bead1/feedback_tuning/5mbar_zcool_init.h5', \
#              # '/data/old_trap/20230531/bead1/feedback_tuning/3mbar_zcool_low1.h5', \
#              '/data/old_trap/20230531/bead1/1_5mbar_zcool.h5', \
#              '/data/old_trap/20230531/bead1/1_5mbar_xzcool.h5', \
#              '/data/old_trap/20230531/bead1/1_5mbar_yzcool.h5', \
#              '/data/old_trap/20230531/bead1/1_5mbar_xyzcool.h5', \
#              # '/data/old_trap/20230531/bead1/discharge/test_hp/1_5mbar_xyzcool_xdrive.h5', \
#              # '/data/old_trap/20230531/bead1/discharge/test_hp/1_5mbar_xyzcool_ydrive.h5', \
#              # '/data/old_trap/20230531/bead1/spinup/1_5mbar_xyzcool_init41Hz.h5', \
#              # '/data/old_trap/20230531/bead1/spinup/1_5mbar_xyzcool_init91Hz.h5', \
#              # '/data/old_trap/20230531/bead1/spinup/1_5mbar_xyzcool_spin700Hz_xdrive41Hz.h5', \
#              # '/data/old_trap/20230531/bead1/spinup/1_5mbar_xyzcool_spin700Hz_xdrive61Hz.h5', \
#              # '/data/old_trap/20230531/bead1/spinup/1_5mbar_zcool_spin700Hz_xdrive61Hz.h5', \
#              # '/data/old_trap/20230531/bead1/discharge/test_hp_2/1_5mbar_zcool_xdrive.h5', \
#              # '/data/old_trap/20230531/bead1/discharge/test_hp_2/1_5mbar_zcool_ydrive.h5', \
#             ]

allfiles  = [\
             # '/data/old_trap/20230617/bead1/6mbar_zcool.h5', \
             # '/data/old_trap/20230617/bead1/lowering/6mbar_zcool_init.h5', \
             # '/data/old_trap/20230617/bead1/lowering/6mbar_zcool_m200k.h5', \
             # '/data/old_trap/20230617/bead1/lowering/6mbar_zcool_m500k.h5', \
             # '/data/old_trap/20230617/bead1/lowering/6mbar_zcool_m800k.h5', \
             # '/data/old_trap/20230617/bead1/lowering/6mbar_zcool_m800k_realign.h5', \
             '/data/old_trap/20230617/bead1/feedback/18e-1mbar_zxycool.h5', \
             '/data/old_trap/20230617/bead1/feedback/12e-1mbar_zxycool.h5', \
            ]

# allfiles = ['/data/new_trap/20200210/Bead2/InitialTest/Data56.h5', \
#             ]

new_trap = False


tfdate = '20190619'  # Bangs bead
tfdate = '20200327'  # gbead
tfdate = ''
tf_plot = False
diag = False

# filename_labels = True 
filename_labels = False

#labs = ['1','2', '3']

figsize = (6,7)

data_axes = [0,1,2]

amp_axes = []
# amp_axes = [0, 1, 2, 3, 4]

phase_axes = []
# phase_axes = [0, 1, 2, 3, 4]

# fb_axes = []
fb_axes = [0,1,2]

other_axes = []
# other_axes = [0,3,4,5,6]
# other_axes = [1,2,3,4,5,6]

elec_axes = []
# elec_axes = [0, 1, 2, 3, 4, 5, 6, 7]

cant_axes = []
#cant_axes = [0,1,2]
#other_axes = [5,7]
plot_power = True
# plot_power = False

drive_ax = 1

xlim = ()
# xlim = (1, 1000)

ylim = ()
# ylim = (1e0, 1e5)
# ylim = (1e-19, 1e-16)
# ylim = (1e-7, 1e+4)

lpf = 2500   # Hz

#file_inds = (0, 3)
file_inds = (0, 100)

file_step = 1

fullNFFT = True
userNFFT = 2**12

cascade = False
cascade_fac = 0.1

#window = mlab.window_hanning
window = mlab.window_none

###########################################################

# cmap = 'inferno'
cmap = 'plasma'
#cmap = 'jet'

posdic = {0: 'x', 1: 'y', 2: 'z'}

psd_ylabel = '$\\sqrt{\\mathrm{PSD}}$ $[\\mathrm{N}/\\sqrt{\\mathrm{Hz}}]$'
psd_arb_ylabel = '$\\sqrt{\\mathrm{PSD}}$ $[\\mathrm{Arb}/\\sqrt{\\mathrm{Hz}}]$'


def plot_many_spectra(files, data_axes=[0,1,2], cant_axes=[], elec_axes=[], other_axes=[], \
                      fb_axes=[], plot_power=False, diag=True, colormap='plasma', \
                      sort='time', file_inds=(0,10000), file_step=1):
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
                                    figsize=figsize)
    else:
        dfig, daxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True, \
                                    figsize=figsize)
    dfig.suptitle('XYZ Data', fontsize=18)

    if len(amp_axes):
        ampfig, ampaxarr = plt.subplots(len(amp_axes),1,sharex=True,sharey=True)
        if len(amp_axes) == 1:
            ampaxarr = [ampaxarr]
        ampfig.suptitle('Amp Data', fontsize=18)
    if len(phase_axes):
        phasefig, phaseaxarr = plt.subplots(len(phase_axes),1,sharex=True,sharey=True)
        if len(phase_axes) == 1:
            phaseaxarr = [phaseaxarr]
        phasefig.suptitle('Phase Data', fontsize=18)
    if len(cant_axes):
        cfig, caxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True)
        if len(cant_axes) == 1:
            caxarr = [caxarr]
        cfig.suptitle('Attractor Data', fontsize=18)
    if len(elec_axes):
        efig, eaxarr = plt.subplots(len(elec_axes),1,sharex=True,sharey=True)
        if len(elec_axes) == 1:
            eaxarr = [eaxarr]
        efig.suptitle('Electrode Data', fontsize=18)
    if len(other_axes):
        ofig, oaxarr = plt.subplots(len(other_axes),1,sharex=True,sharey=True)
        if len(other_axes) == 1:
            oaxarr = [oaxarr]
        ofig.suptitle('Other Data', fontsize=18)
    if len(fb_axes):
        fbfig, fbaxarr = plt.subplots(len(fb_axes),1,sharex=True,sharey=True, \
                                    figsize=figsize)
        if len(fb_axes) == 1:
            fbaxarr = [fbaxarr]
        fbfig.suptitle('Feedback Data', fontsize=18)
    if plot_power:
        pfig, paxarr = plt.subplots(2,1,sharex=True,figsize=(6,6))
        pfig.suptitle('Power/Power Feedback Data', fontsize=18)

    files = files[file_inds[0]:file_inds[1]:file_step]
    if invert_order:
        files = files[::-1]

    colors = bu.get_colormap(len(files), cmap=colormap)
    #colors = ['C0', 'C1', 'C2']

    old_per = 0
    print("Processing %i files..." % len(files))
    for fil_ind, fil in enumerate(files):
        color = colors[fil_ind]
        
        # Display percent completion
        bu.progress_bar(fil_ind, len(files))

        # Load data
        df = bu.DataFile()
        try:
            if new_trap:
                df.load_new(fil)
            else:
                df.load(fil)
        except:
            continue

        if len(other_axes):
            df.load_other_data()

        df.calibrate_phase()

        # plt.figure()
        # plt.plot(df.phase[4]-np.mean(df.phase[4]))
        # plt.plot(df.phase[0]-np.mean(df.phase[0]))

        # plt.figure()
        # plt.plot(df.phase[4]-df.phase[0])

        # plt.figure()
        # plt.loglog(np.abs(np.fft.rfft(df.phase[4])))
        # plt.loglog(np.abs(np.fft.rfft(df.phase[4]-df.phase[0])))

        # plt.figure()
        # plt.plot( (df.zcal - np.mean(df.zcal)) / \
        #             np.max(df.zcal - np.mean(df.zcal)) )
        # # plt.plot( (df.phase[0] - np.mean(df.phase[0])) / \
        # #             np.max(df.phase[0] - np.mean(df.phase[0])) )
        # # plt.plot( (df.pos_data[0] - np.mean(df.pos_data[0])) / 
        # #             np.max(df.pos_data[0] - np.mean(df.pos_data[0])) )
        # plt.show()
        # input()

        df.calibrate_stage_position()
        
        #df.high_pass_filter(fc=1)
        #df.detrend_poly()

        #plt.figure()
        #plt.plot(df.pos_data[0])
        #plt.show()

        if cascade:
            cascade_scale = (cascade_fac)**fil_ind
        else:
            cascade_scale = 1.0

        freqs = np.fft.rfftfreq(len(df.pos_data[0]), d=1.0/df.fsamp)

        if diag:
            df.diagonalize(maxfreq=lpf, date=tfdate, plot=tf_plot)

        if fil_ind == 0 and len(cant_axes):
            drivepsd = np.abs(np.fft.rfft(df.cant_data[drive_ax]))
            driveind = np.argmax(drivepsd[1:]) + 1
            drive_freq = freqs[driveind]

        for axind, ax in enumerate(data_axes):

            try:
                fac = cascade_scale * df.conv_facs[ax]# * (1.0 / 0.12e-12)
            except:
                fac = cascade_scale

            if fullNFFT:
                NFFT = len(df.pos_data[ax])
            else:
                NFFT = userNFFT
        
            psd, freqs = mlab.psd(df.pos_data[ax], Fs=df.fsamp, \
                                  NFFT=NFFT, window=window)

            norm = bu.fft_norm(df.nsamp, df.fsamp)
            new_freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)
            #fac = 1.0

            if diag:
                dpsd, dfreqs = mlab.psd(df.diag_pos_data[ax], Fs=df.fsamp, \
                                        NFFT=NFFT, window=window)

                daxarr[axind,0].loglog(freqs, np.sqrt(psd) * fac, \
                                       color=color, label=df.fname)
                daxarr[axind,0].grid(alpha=0.5)
                daxarr[axind,1].loglog(new_freqs, \
                                       norm*np.abs(np.fft.rfft(df.diag_pos_data[ax])), \
                                       color='k')
                daxarr[axind,1].loglog(freqs, \
                                       np.sqrt(dpsd), \
                                       color=color)
                daxarr[axind,1].grid(alpha=0.5)
                daxarr[axind,0].set_ylabel(psd_ylabel)
                if ax == data_axes[-1]:
                    daxarr[axind,0].set_xlabel('Frequency [Hz]')
                    daxarr[axind,1].set_xlabel('Frequency [Hz]')
            else:
                daxarr[axind].loglog(freqs, np.sqrt(psd)*fac, \
                                     color=color, label=df.fname)
                daxarr[axind].grid(alpha=0.5)
                daxarr[axind].set_ylabel(psd_arb_ylabel)

                if ax == data_axes[-1]:
                    daxarr[axind].set_xlabel('Frequency [Hz]')


        if len(fb_axes):
            for axind, ax in enumerate(fb_axes):
                fb_psd, freqs = mlab.psd(df.pos_fb[ax], Fs=df.fsamp, \
                                      NFFT=NFFT, window=window)
                fbaxarr[axind].loglog(freqs, np.sqrt(fb_psd) * fac, color=color)
                fbaxarr[axind].set_ylabel('$\\sqrt{\\mathrm{PSD}}$')

        if len(amp_axes):
            for axind, ax in enumerate(amp_axes):
                psd, freqs = mlab.psd(df.amp[ax], Fs=df.fsamp, \
                                      NFFT=NFFT, window=window)
                ampaxarr[axind].loglog(freqs, np.sqrt(psd), color=color )
                ampaxarr[axind].set_ylabel('$\\sqrt{\\mathrm{PSD}}$')

        if len(phase_axes):
            for axind, ax in enumerate(phase_axes):
                psd, freqs = mlab.psd(df.phase[ax], Fs=df.fsamp, \
                                      NFFT=NFFT, window=window)
                phaseaxarr[axind].loglog(freqs, np.sqrt(psd), color=color )
                phaseaxarr[axind].set_ylabel('$\\sqrt{\\mathrm{PSD}}$')

        if len(cant_axes):
            for axind, ax in enumerate(cant_axes):
                psd, freqs = mlab.psd(df.cant_data[ax], Fs=df.fsamp, \
                                      NFFT=NFFT, window=window)
                caxarr[axind].loglog(freqs, np.sqrt(psd), color=color )
                caxarr[axind].set_ylabel('$\\sqrt{\\mathrm{PSD}}$')

        if len(elec_axes):
            for axind, ax in enumerate(elec_axes):
                psd, freqs = mlab.psd(df.electrode_data[ax], Fs=df.fsamp, \
                                      NFFT=NFFT, window=window)
                eaxarr[axind].loglog(freqs, np.sqrt(psd), color=color ) 
                eaxarr[axind].set_ylabel('$\\sqrt{\\mathrm{PSD}}$')

        if len(other_axes):
            for axind, ax in enumerate(other_axes):
                #ax = ax - 3
                psd, freqs = mlab.psd(df.other_data[ax], Fs=df.fsamp, \
                                      NFFT=NFFT, window=window)
                oaxarr[axind].loglog(freqs, np.sqrt(psd), color=color)
                oaxarr[axind].set_ylabel('$\\sqrt{\\mathrm{PSD}}$')

        if plot_power:
            psd, freqs = mlab.psd(df.power, Fs=df.fsamp, \
                                        NFFT=NFFT, window=window)
            psd_fb, freqs_fb = mlab.psd(df.power_fb, Fs=df.fsamp, \
                                        NFFT=NFFT, window=window)
            paxarr[0].loglog(freqs, np.sqrt(psd), color=color)
            paxarr[1].loglog(freqs_fb, np.sqrt(psd_fb), color=color)
            for axind in [0,1]:
                paxarr[axind].set_ylabel('$\\sqrt{\\mathrm{PSD}}$')


    if filename_labels:
        daxarr[0].legend(fontsize=10)
    if len(fb_axes):
        fbaxarr[0].legend(fontsize=10)

    #daxarr[0].set_xlim(0.5, 25000)
    
    if diag:
        derp_ax = daxarr[0,0]
    else:
        derp_ax = daxarr[0]

    # derp_ax.legend(fontsize=10)

    if len(ylim):
        derp_ax.set_ylim(*ylim)
    if len(xlim):
        derp_ax.set_xlim(*xlim)

    dfig.tight_layout()
    dfig.subplots_adjust(top=0.91)

    if plot_power:
        paxarr[-1].set_xlabel('Frequency [Hz]')
        pfig.tight_layout()
        pfig.subplots_adjust(top=0.91)
    if len(amp_axes):
        ampaxarr[-1].set_xlabel('Frequency [Hz]')
        ampfig.tight_layout()
        ampfig.subplots_adjust(top=0.91)
    if len(phase_axes):
        phaseaxarr[-1].set_xlabel('Frequency [Hz]')
        phasefig.tight_layout()
        phasefig.subplots_adjust(top=0.91)
    if len(cant_axes):
        caxarr[-1].set_xlabel('Frequency [Hz]')
        cfig.tight_layout()
        cfig.subplots_adjust(top=0.91)
    if len(elec_axes):
        eaxarr[-1].set_xlabel('Frequency [Hz]')
        efig.tight_layout()
        efig.subplots_adjust(top=0.91)
    if len(other_axes):
        oaxarr[-1].set_xlabel('Frequency [Hz]')
        ofig.tight_layout()
        ofig.subplots_adjust(top=0.91)
    if len(fb_axes):
        fbaxarr[-1].set_xlabel('Frequency [Hz]')
        fbfig.tight_layout()
        fbfig.subplots_adjust(top=0.91)
    
    plt.show()


if use_dir:
    allfiles, lengths = bu.find_all_fnames(dir1, sort_time=True)

allfiles = allfiles[:maxfiles]
#allfiles = bu.sort

plot_many_spectra(allfiles, file_inds=file_inds, diag=diag, \
                  data_axes=data_axes, other_axes=other_axes, \
                  fb_axes=fb_axes, cant_axes=cant_axes, \
                  elec_axes=elec_axes, plot_power=plot_power, \
                  colormap=cmap, file_step=file_step)