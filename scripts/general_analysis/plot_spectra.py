import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

plt.rcParams.update({'font.size': 16})

dir1 = '/data/old_trap/20191017/bead1/spinning/junk/shit_test_10'
maxfiles = 500



#dir1 = '/data/old_trap/20191203/daq_tests/nominal'
dir1 = '/data/old_trap/20191203/daq_tests/new_wiring'


use_dir = False

# allfiles = ['/daq2/20190320/bead2/1_5mbar_zcool.h5', \
#             '/daq2/20190320/bead2/1_5mbar_xzcool_pos.h5', \
#             '/daq2/20190320/bead2/1_5mbar_yzcool_neg.h5', \
#             '/daq2/20190320/bead2/1_5mbar_xyzcool.h5']



#allfiles = ['/daq2/20190327/bead1/1_5mbar_nocool_pos1.h5', \
#            '/daq2/20190327/bead1/1_5mbar_zcool_pos1.h5', \
#            '/daq2/20190327/bead1/1_5mbar_zcool_pos2.h5', \
#            '/daq2/20190327/bead1/1_5mbar_zcool_pos3.h5', \
#            '/daq2/20190327/bead1/1_5mbar_zcool_pos4.h5', \
#            '/daq2/20190327/bead1/1_5mbar_zcool_pos5.h5']

#allfiles = [#'/daq2/20190327/bead1/1_5mbar_zcool.h5', \
#            #'/daq2/20190327/bead1/1_5mbar_xzcool_pos.h5', \
#            #'/daq2/20190327/bead1/1_5mbar_yzcool_neg.h5', \
#            '/daq2/20190327/bead1/1_5mbar_xyzcool.h5', \
#            '/daq2/20190327/bead1/turbombar_xyzcool_saturday.h5', \
#            '/data/20190124/bead2/turbombar_zcool_discharged.h5', \
#            ]

#allfiles = ['/daq2/20190507/bead1/1_5mbar_powfb_xyzcool.h5', \
#            '/daq2/20190507/bead1/turbombar_powfb_xyzcool.h5']


# allfiles = ['/data/old_trap/20190626/bead1/1_5mbar_powfb_zcool_init.h5', \
#             '/data/old_trap/20190626/bead1/1_5mbar_powfb_zcool_low1.h5', \
#             '/data/old_trap/20190626/bead1/1_5mbar_powfb_zcool_low2.h5', \
#             #'/daq2/20190619/bead1/1_5mbar_powfb_zcool_low3.h5', \
#             #'/daq2/20190619/bead1/1_5mbar_powfb_zcool_low4.h5', \
#             '/data/old_trap/20190626/bead1/1_5mbar_powfb_xzcool_low2.h5', \
#             '/data/old_trap/20190626/bead1/1_5mbar_powfb_yzcool_low2.h5', \
#             ]


# allfiles = [#'/data/old_trap/20191007/bead1/prebead/pow_term.h5', \
#             #'/data/old_trap/20191007/bead1/prebead/powfb_nofb.h5', \
#             #'/data/old_trap/20191007/bead1/prebead/powfb_i-gain.h5', \
#             #'/data/old_trap/20191007/bead1/prebead/powfb_di-gain.h5', \
#             #'/data/old_trap/20191007/bead1/1_5mbar_powfb_nocool.h5', \
#             #'/data/old_trap/20191007/bead1/1_5mbar_powfb_zcool_i-gain.h5', \
#             #'/data/old_trap/20191007/bead1/1_5mbar_powfb_zcool_pid-gain.h5', \
#             #'/data/old_trap/20191007/bead1/1_5mbar_powfb_zcool_pid-gain_more.h5', \
#             #'/data/old_trap/20191007/bead1/1_5mbar_powfb_zcool_pid-gain_less.h5', \
#             #'/data/old_trap/20191007/bead1/1_5mbar_powfb_zcool_pid-gain_less_2.h5', \
#             #'/data/old_trap/20191007/bead1/1_5mbar_powfb_zcool_init.h5', \
#             #'/data/old_trap/20191007/bead1/1_5mbar_powfb_zcool_low1.h5', \
#             #'/data/old_trap/20191007/bead1/1_5mbar_powfb_zcool_low2.h5', \
#             #'/data/old_trap/20191007/bead1/1_5mbar_powfb_zcool_low3.h5', \
#             '/data/old_trap/20191007/bead1/1_5mbar_powfb_zcool.h5', \
#             '/data/old_trap/20191007/bead1/1_5mbar_powfb_xzcool.h5', \
#             '/data/old_trap/20191007/bead1/1_5mbar_powfb_yzcool.h5', \
#             ]


allfiles = ['/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_nofb_nocool.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-i-gain_nocool.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pi-gain_nocool.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pi-gain_zcool-i-gain.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pi-gain_zcool-pi-gain.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pi-gain-more_zcool-pi-gain.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pi-gain_zcool-pi-gain-more.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pi-gain_zcool-pid-gain.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pi-gain_zcool-pid-gain-more.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pi-gain_zcool-pid-gain-more2.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pi-gain_zcool-pid-gain-more3.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pi-gain_zcool-pid-gain-more4.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pid-gain_zcool-pid-gain.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb-pid-gain_zcool-pid-gain_2.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb_zcool.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb_zcool_low1.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb_zcool_low2.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb_zcool_low3.h5', \
            #'/data/old_trap/20191017/bead1/fb_tuning/1_5mbar_powfb_zcool_low4.h5', \
            '/data/old_trap/20191017/bead1/1_5mbar_powfb_zcool.h5'
            ]

allfiles = ['/data/new_trap/20191204/Bead1/InitialTest/Data28.h5', \
            #'/data/new_trap/20191204/Bead1/Discharge/Discharge_18.h5', \
            #'/data/new_trap/20191204/Bead1/Discharge/Discharge_19.h5', \
            #'/data/new_trap/20191204/Bead1/Discharge/Discharge_20.h5', \
            #'/data/new_trap/20191204/Bead1/Discharge/Discharge_21.h5', \
            #'/data/new_trap/20191204/Bead1/Discharge/Discharge_22.h5', \
            #'/data/new_trap/20191204/Bead1/TransFunc/TransFunc_X_1.h5', \
            #'/data/new_trap/20191204/Bead1/TransFunc/TransFunc_X_4.h5', \
            #'/data/new_trap/20191204/Bead1/TransFunc/TransFunc_X_7.h5', \
            #'/data/new_trap/20191204/Bead1/TransFunc/TransFunc_Y_2.h5', \
            #'/data/new_trap/20191204/Bead1/TransFunc/TransFunc_Y_5.h5', \
            #'/data/new_trap/20191204/Bead1/TransFunc/TransFunc_Y_8.h5', \
            #'/data/new_trap/20191204/Bead1/TransFunc/TransFunc_Z_3.h5', \
            ]

new_trap = True


tfdate = '20191204'

#filename_labels = True 
filename_labels = False

#labs = ['1','2', '3']

data_axes = [0,1,2]
fb_axes = []
#fb_axes = [0,1,2]

other_axes = []
#other_axes = [5,6,7]

cant_axes = []
#cant_axes = [0,1,2]
#other_axes = [0,1,2,3,4,5,6,7]
#other_axes = [5,7]
plot_power = False

drive_ax = 1

step10 = False #True
invert_order = False

#### HACKED SHITTTT
savefigs = False
title_pre = '/home/charles/plots/20180105_precession/test1_100V_muchlater3'

#lim = ()
xlim = (1, 1000)

#ylim = ()
ylim = (1e-18, 1e-15)
#ylim = (1e-7, 1e+4)

lpf = 2500   # Hz

#file_inds = (0, 3)
file_inds = (0, 1800)

userNFFT = 2**13
diag = True

fullNFFT = False

#window = mlab.window_hanning
window = mlab.window_none

###########################################################

cmap = 'inferno'
#cmap = 'jet'

posdic = {0: 'x', 1: 'y', 2: 'z'}



def plot_many_spectra(files, data_axes=[0,1,2], cant_axes=[], elec_axes=[], other_axes=[], \
                      fb_axes=[], plot_power=False, diag=True, colormap='plasma', \
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
        dfig, daxarr = plt.subplots(len(data_axes),2,sharex=True,sharey=True, \
                                    figsize=(8,8))
    else:
        dfig, daxarr = plt.subplots(len(data_axes),1,sharex=True,sharey=True, \
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
    if len(fb_axes):
        fbfig, fbaxarr = plt.subplots(len(fb_axes),1,sharex=True,sharey=True)
        if len(fb_axes) == 1:
            fbaxarr = [fbaxarr]
    if plot_power:
        pfig, paxarr = plt.subplots(1,1)

    kludge_fig, kludge_ax = plt.subplots(1,1)

    files = files[file_inds[0]:file_inds[1]]
    if step10:
        files = files[::10]
    if invert_order:
        files = files[::-1]

    colors = bu.get_color_map(len(files), cmap=colormap)
    #colors = ['C0', 'C1', 'C2']

    old_per = 0
    print("Processing %i files..." % len(files))
    for fil_ind, fil in enumerate(files):
        color = colors[fil_ind]
        
        # Display percent completion
        bu.progress_bar(fil_ind, len(files))

        # Load data
        df = bu.DataFile()
        if new_trap:
            df.load_new(fil)
        else:
            df.load(fil)

        if len(other_axes):
            df.load_other_data()

        df.calibrate_stage_position()
        
        #df.high_pass_filter(fc=1)
        #df.detrend_poly()

        #plt.figure()
        #plt.plot(df.pos_data[0])
        #plt.show()

        freqs = np.fft.rfftfreq(len(df.pos_data[0]), d=1.0/df.fsamp)

        df.diagonalize(maxfreq=lpf, interpolate=False, date=tfdate)

        if fil_ind == 0 and len(cant_axes):
            drivepsd = np.abs(np.fft.rfft(df.cant_data[drive_ax]))
            driveind = np.argmax(drivepsd[1:]) + 1
            drive_freq = freqs[driveind]

        for axind, ax in enumerate(data_axes):

            try:
                fac = df.conv_facs[ax]# * (1.0 / 0.12e-12)
            except:
                fac = 1.0

            if fullNFFT:
                NFFT = len(df.pos_data[ax])
            else:
                NFFT = userNFFT
        
            psd, freqs = mlab.psd(df.pos_data[ax], Fs=df.fsamp, \
                                  NFFT=NFFT, window=window)
            fb_psd, freqs = mlab.psd(df.pos_fb[ax], Fs=df.fsamp, \
                                  NFFT=NFFT, window=window)

            norm = bu.fft_norm(df.nsamp, df.fsamp)
            new_freqs = np.fft.rfftfreq(df.nsamp, d=1.0/df.fsamp)
            #fac = 1.0

            kludge_fac = 1.0
            #kludge_fac = 1.0 / np.sqrt(10)
            if diag:
                dpsd, dfreqs = mlab.psd(df.diag_pos_data[ax], Fs=df.fsamp, \
                                        NFFT=NFFT, window=window)
                kludge_ax.loglog(freqs, np.sqrt(dpsd) *kludge_fac, color='C'+str(axind), \
                                    label=posdic[axind])
                kludge_ax.set_ylabel('$\sqrt{\mathrm{PSD}}$ $[\mathrm{N}/\sqrt{\mathrm{Hz}}]$')
                kludge_ax.set_xlabel('Frequency [Hz]')

                daxarr[axind,0].loglog(new_freqs, fac*norm*np.abs(np.fft.rfft(df.pos_data[ax]))*kludge_fac, color='k', label='np.fft with manual normalization')
                daxarr[axind,0].loglog(freqs, np.sqrt(psd) * fac *kludge_fac, color=color, label='mlab.psd')
                daxarr[axind,0].grid(alpha=0.5)
                daxarr[axind,1].loglog(new_freqs, norm*np.abs(np.fft.rfft(df.diag_pos_data[ax])) *kludge_fac, color='k')
                daxarr[axind,1].loglog(freqs, np.sqrt(dpsd) *kludge_fac, color=color)
                daxarr[axind,1].grid(alpha=0.5)
                daxarr[axind,0].set_ylabel('$\sqrt{\mathrm{PSD}}$ $[\mathrm{N}/\sqrt{\mathrm{Hz}}]$')
                if ax == data_axes[-1]:
                    daxarr[axind,0].set_xlabel('Frequency [Hz]')
                    daxarr[axind,1].set_xlabel('Frequency [Hz]')
            else:
                daxarr[axind].loglog(new_freqs, norm*np.abs(np.fft.rfft(df.pos_data[ax])), color='k', label='np.fft with manual normalization')
                daxarr[axind].loglog(freqs, np.sqrt(psd)*fac, color=color, label='mlab.psd')
                daxarr[axind].grid(alpha=0.5)
                daxarr[axind].set_ylabel('$\\sqrt{\mathrm{PSD}}$ $[\\mathrm{Arb}/\\sqrt{\mathrm{Hz}}]$')
                #daxarr[axind].set_ylabel('$\sqrt{\mathrm{PSD}}$ $[\mathrm{N}/\sqrt{\mathrm{Hz}}]$')

                if len(fb_axes):
                    fbaxarr[axind].loglog(freqs, np.sqrt(fb_psd) * fac, color=color)
                    fbaxarr[axind].grid(alpha=0.5)
                    fbaxarr[axind].set_ylabel('$\sqrt{\mathrm{PSD}}$')


                if ax == data_axes[-1]:
                    daxarr[axind].set_xlabel('Frequency [Hz]')
                    if len(fb_axes):
                        fbaxarr[axind].set_xlabel('Frequency [Hz]')

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
                #ax = ax - 3
                psd, freqs = mlab.psd(df.other_data[ax], Fs=df.fsamp, \
                                      NFFT=NFFT, window=window)
                oaxarr[axind].loglog(freqs, np.sqrt(psd), color=color)

        if plot_power:
            psd, freqs = mlab.psd(df.power, Fs=df.fsamp, \
                                        NFFT=NFFT, window=window)
            paxarr.loglog(freqs, np.sqrt(psd), color=color)


    if filename_labels:
        daxarr[0].legend(fontsize=10)
    if len(fb_axes):
        fbaxarr[0].legend()

    #daxarr[0].set_xlim(0.5, 25000)
    
    if diag:
        derp_ax = daxarr[0,0]
    else:
        derp_ax = daxarr[0]

    derp_ax.legend(fontsize=10)

    if len(ylim):
        derp_ax.set_ylim(*ylim)
        kludge_ax.set_ylim(*ylim)
    if len(xlim):
        derp_ax.set_xlim(*xlim)
        kludge_ax.set_xlim(1,500)

    dfig.tight_layout()

    kludge_ax.grid()
    kludge_ax.legend()
    kludge_fig.tight_layout()

    if len(cant_axes):
        cfig.tight_layout()
    if len(elec_axes):
        efig.tight_layout()
    if len(other_axes):
        ofig.tight_layout()
    if len(fb_axes):
        fbfig.tight_layout()


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
    allfiles, lengths = bu.find_all_fnames(dir1, sort_time=True)

allfiles = allfiles[:maxfiles]
#allfiles = bu.sort

plot_many_spectra(allfiles, file_inds=file_inds, diag=diag, \
                  data_axes=data_axes, other_axes=other_axes, \
                  fb_axes=fb_axes, cant_axes=cant_axes, \
                  plot_power=plot_power, colormap=cmap)