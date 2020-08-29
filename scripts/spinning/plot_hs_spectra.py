import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti
import scipy.signal as signal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

plt.rcParams.update({'font.size': 16})

# dirname = '/data/old_trap/20190905/bead1/spinning/ringup/100kHz_1'
# dirname = '/data/old_trap/20190905/bead1/spinning/ringup/test3'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/nominal_config'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/monitors_unplugged'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/monitors_unplugged_term'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/tabor_off'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/tabor_off_proper_conn'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/tabor_unplugged'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/rot_diode_unplugged'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/different_srs_output'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/drive_at_9k'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/drive_at_9k_better_term'
# dirname = '/data/old_trap/20190905/bead1/spinning/residual_field_tests/drive_at_9k_much_later'

# dirname = '/data/old_trap/20190905/bead1/spinning/ringdown_manual/50kHz_start_4'
# dirname = '/data/old_trap/20190905/bead1/spinning/ringdown_manual/100kHz_start_2'
# dirname = '/data/old_trap/20190905/bead1/spinning/ringdown_manual/terminal_velocity_test'
# dirname = '/data/old_trap/20190905/bead1/spinning/ringdown_manual/terminal_velocity_test_later'

# dirname = '/data/old_trap/20190905/bead1/spinning/high_speed_tests/210kHz'

# dirname = '/data/old_trap/20190905/bead1/spinning/ringdown/210kHz_start_1'


# dirname = '/data/old_trap/20191017/bead1/spinning/ringdown/term_velocity_check_7'

# dirname = '/data/old_trap/20191017/bead1/spinning/junk/shit_test_99'
# dirname = '/data/old_trap/20191017/bead1/spinning/ringdown/110kHz_start_test'

# dirname = '/data/old_trap/20200322/gbead1/spinning/ringdown/110kHz_1'
# dirname = '/data/old_trap/20200322/gbead1/spinning/wobble/wobble_init/wobble_0000'
# dirname = '/data/old_trap/20200322/gbead1/spinning/ringdown/terminal_velocity_check_2'

# dirname = '/data/old_trap/20200727/bead1/spinning/hf_spin_up'
# dirname = '/data/old_trap/20200727/bead1/spinning/junk/test3'
# dirname = '/data/old_trap/20200727/bead1/spinning/amplitude_change_much-later'
# dirname = '/data/old_trap/20200727/bead1/spinning/wobble_slow/wobble_0/'

dirname = '/data/old_trap/20200727/bead1/spinning/lowp_arb_spinup/'
dirname = '/data/old_trap/20200727/bead1/spinning/lowp_arb_spinup_2/'


use_dir = True
#use_dir = False

# bases = ['/data/old_trap/20190905/bead1/spinning/residual_field_tests/nominal_config', \
#          '/data/old_trap/20190905/bead1/spinning/residual_field_tests/monitors_unplugged_term', \
#          '/data/old_trap/20190905/bead1/spinning/residual_field_tests/rot_diode_unplugged', \
#          '/data/old_trap/20190905/bead1/spinning/residual_field_tests/drive_at_9k_better_term', \
#         ]
# labels = ['Initial: Tabor off, optically driven rotation', \
#           'Tabor monitor -> Terminator', \
#           'Rot. diode -> Terminator', \
#           'Tabor on, but chamber elec. terminated', \
#          ]


savefig = False
fig_savename = ''
#fig_savename = '/home/cblakemore/plots/20190905/residual_field_tests.svg'


data_axes = [0,1]
# data_axes = [0]
# axes_labels = ['Cross-polarized light']
axes_labels = ['Cross-polarized light', 'Tabor Monitor']

use_filename_labels = False
labels = []

decimate = False
decimate_factor = 5

invert_order = False
#file_inds = (-10,-1)
file_inds = (-10, 1000)

userNFFT = 2**14
fullNFFT = True

plot_freqs = (55000.0, 65000.0)
plot_freqs = (100.0, 65000.0)

waterfall = False
waterfall_fac = 0.01

#window = mlab.window_hanning
window = mlab.window_none

#ylim = (1e-21, 1e-14)
#ylim = (1e-7, 1e-1)
xlim = ()
ylim = ()

track_feature = False
tracking_band = 500.0
tracking_axis = 0
tracking_savefile = ''
# tracking_savefile = '/data/old_trap_processed/spinning/ringdown/20200322/term_velocity_check_2.npy'

# approx_feature_loc = 10000.0
# approx_feature_loc = 4216.0
approx_feature_loc = 0.0

###########################################################

cmap = 'plasma'
#cmap = 'jet'


def lorentzian(x, A, mu, gamma, c):
    return A * (gamma**2 / ((x-mu)**2 + gamma**2)) + c



def plot_many_spectra(files, data_axes=[0,1,2], colormap='jet', \
                      sort='time', file_inds=(0,10000), \
                      plot_freqs=(0.0,1000000.0), labels=[]):
    '''Loops over a list of file names, loads each file,
       then plots the amplitude spectral density of any number 
       of data channels

       INPUTS: files, list of files names to extract data
               data_axes, list of pos_data axes to plot

       OUTPUTS: none, plots stuff
    '''

    if track_feature:
        obj0 = bu.hsDat(files[0], load=True)
        freqs0 = np.fft.rfftfreq(obj0.attribs['nsamp'], d=1.0/obj0.attribs['fsamp'])
        asd = bu.fft_norm(obj0.attribs['nsamp'], obj0.attribs['fsamp']) * \
                np.abs(np.fft.rfft(obj0.dat[:,0]))

        fsamp = obj0.attribs['fsamp']
        nsamp = obj0.attribs['nsamp']
        t0 = obj0.attribs['time']

        if approx_feature_loc == 0.0:
            plt.loglog(freqs0, asd)
            plt.show()
            center_guess = float(input('Feature location [Hz]: '))
        else:
            center_guess = approx_feature_loc

        maxind = np.argmax(asd * (np.abs(freqs0 - center_guess)<0.5*tracking_band))
        fc = freqs0[maxind]

        upper1 = (2.0 / fsamp) * (fc + 0.5 * tracking_band)
        lower1 = (2.0 / fsamp) * (fc - 0.5 * tracking_band)
        b1, a1 = signal.butter(3, [lower1, upper1], btype='bandpass')

        dat_filt = signal.filtfilt(b1, a1, obj0.dat[:,0])

        dat_filt_asd = np.abs(np.fft.rfft(dat_filt))
        popt, pcov = opti.curve_fit(lorentzian, freqs0, dat_filt_asd, p0=[dat_filt_asd[maxind], fc, 5, 0])

        fc_old = popt[1]



    dfig, daxarr = plt.subplots(len(data_axes),sharex=True,sharey=False, \
                                figsize=(8,8))
    if len(data_axes) == 1:
        daxarr = [daxarr]


    files = files[file_inds[0]:file_inds[1]]
    if decimate:
        files = files[::decimate_factor]
    if invert_order:
        files = files[::-1]

    colors = bu.get_color_map(len(files), cmap=colormap)
    #colors = ['C0', 'C1', 'C2']

    if track_feature:
        times = []
        feature_locs = []

    old_per = 0
    print("Processing %i files..." % len(files))
    for fil_ind, fil in enumerate(files):
        color = colors[fil_ind]
        
        # Display percent completion
        bu.progress_bar(fil_ind, len(files))

        # Load data
        obj = bu.hsDat(fil, load=True)

        #plt.figure()
        #plt.plot(df.pos_data[0])
        #plt.show()

        fsamp = obj.attribs['fsamp']
        nsamp = obj.attribs['nsamp']
        t = obj.attribs['time']

        freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)

        if waterfall:
            fac = waterfall_fac**fil_ind
        else:
            fac = 1.0

        if not fullNFFT:
            NFFT = userNFFT
        else:
            NFFT = nsamp


        for axind, ax in enumerate(data_axes):
            # if fullNFFT:
            #     NFFT = len(df.pos_data[ax])
            # else:
            #     NFFT = userNFFT
        
            # asd = np.abs(np.fft.rfft(obj.dat[:,axind]))

            psd, freqs = mlab.psd(obj.dat[:,axind], Fs=obj.attribs['fsamp'], \
                                    NFFT=NFFT, window=window)
            asd = np.sqrt(psd)

            plot_inds = (freqs > plot_freqs[0]) * (freqs < plot_freqs[1])

            if track_feature and (axind == tracking_axis):

                freqs_full = np.fft.rfftfreq(nsamp, d=1.0/fsamp)
                asd_full = np.abs(np.fft.rfft(obj.dat[:,axind]))

                maxind = np.argmax(asd_full * (np.abs(freqs_full - fc_old)<0.5*tracking_band))
                fc = freqs_full[maxind]

                upper1 = (2.0 / fsamp) * (fc + 5 * tracking_band)
                lower1 = (2.0 / fsamp) * (fc - 5 * tracking_band)
                b1, a1 = signal.butter(3, [lower1, upper1], btype='bandpass')

                dat_filt = signal.filtfilt(b1, a1, obj.dat[:,axind])

                dat_filt_asd = np.abs(np.fft.rfft(dat_filt))
                try:
                    popt, pcov = opti.curve_fit(lorentzian, freqs_full, dat_filt_asd, \
                                                    p0=[dat_filt_asd[maxind], fc, 5, 0])
                    fc_old = popt[1]
                except:
                    print('bad fit')

                times.append((t-t0)*1e-9)
                feature_locs.append(fc_old)

            if len(labels):
                daxarr[axind].loglog(freqs[plot_inds], asd[plot_inds]*fac, \
                                     label=labels[fil_ind], color=colors[fil_ind])
            else:
                daxarr[axind].loglog(freqs[plot_inds], asd[plot_inds]*fac, \
                                     color=colors[fil_ind])

            daxarr[axind].set_ylabel('$\sqrt{\mathrm{PSD}}$')
            if ax == data_axes[-1]:
                daxarr[axind].set_xlabel('Frequency [Hz]')

    if len(axes_labels):
        for labelind, label in enumerate(axes_labels):
            daxarr[labelind].set_title(label)

    if len(labels):
        daxarr[0].legend(fontsize=10)
    if len(xlim):
        daxarr[0].set_xlim(xlim[0], xlim[1])
    if len(ylim):
        daxarr[0].set_ylim(ylim[0], ylim[1])
    plt.tight_layout()

    if savefig:
        dfig.savefig(fig_savename)

    if track_feature:
        trackfig, trackax = plt.subplots(1,1)
        trackax.plot(times, feature_locs)
        trackax.set_xlabel('Time [s]')
        trackax.set_ylabel('Center Frequency [Hz]')

        if len(tracking_savefile):
            np.save( open(tracking_savefile, 'wb'), np.array([times, feature_locs]) )

        plt.tight_layout()


    plt.show()

    # if not savefigs:
    #     plt.show()


if use_dir:
    allfiles, lengths = bu.find_all_fnames(dirname, sort_time=True)

if use_filename_labels:
    labels = np.copy(allfiles)

plot_many_spectra(allfiles, file_inds=file_inds, \
                  data_axes=data_axes, colormap=cmap, \
                  plot_freqs=plot_freqs, labels=labels)