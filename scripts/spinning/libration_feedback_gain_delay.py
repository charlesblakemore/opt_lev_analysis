import os, sys, time, itertools, re, warnings, h5py
import numpy as np
import dill as pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as optimize
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed

# ncore = 1
ncore = 10

plt.rcParams.update({'font.size': 14})
warnings.filterwarnings('ignore')





#################################
###                           ###
###   WHICH DATA TO ANALYZE   ###
###                           ###
#################################


dir_base = '/data/old_trap/'
processed_base = '/data/old_trap_processed/spinning/'

input_dict = {}


beadtype = 'bangs5'
# beadtype = 'german7'


def formatter20210119_delay(measString, freq):
    return os.path.join(measString, f'{freq:d}Hz')

# meas_base = '20210119/dds_delay_test'
# input_dict['delay'] = \
#     [ formatter20210119_delay(meas_base, freq) \
#         for freq in [10, 100, 1000] ]

# meas_base = '20210119/dds_delay_test'
# input_dict['delay'] = \
#     [ formatter20210119_delay(meas_base, freq) \
#         for freq in ( [i for i in range(10,150,10)] \
#                         + [i for i in range(200,2600,100)] ) ]


def formatter20210119_gain(measString, freq, ind, trial):
    if ind == 0:
        return os.path.join(measString, f'{freq:d}Hz_{trial:04d}')
    if ind == 1:
        return os.path.join(measString, f'{freq:d}Hz_higher_{trial:04d}')

meas_base = '20210119/dds_gain_test'
input_dict['gain'] = \
    [ formatter20210119_gain(meas_base, freq, ind, trial) \
        for freq in [700] for ind in [0,1] for trial in range(10) ]



file_step = 1
file_inds = (0, 100)
# file_inds = (0, 1)

save_gain_and_delay = True
# gain_and_delay_filename = \
#     os.path.join(processed_base, '20210119/dds_delay_test', \
#                  'delay_vs_frequency.p')
gain_and_delay_filename = \
    os.path.join(processed_base, '20210119/dds_delay_test', \
                 'scaling_vs_gain.p')





#####################################
###                               ###
###   SIGNAL PROCESSING OPTIONS   ###
###                               ###
#####################################

phase_conversion_fac = ((2**16 - 1) / 20.0) * (2.0*np.pi/32000.0)
delay_phi_dg = 2.0

### Carrier filter stuff
# drive_bandwidth = 1000.0
drive_bandwidth = 10000.0
signal_bandwidth = 10000.0

### Hard lower limit on libration to cut out misidentified libration features
min_libration_freq = 500
max_libration_freq = 1000

### Scale the spectrum by freq**alpha solely to help feature identification
### when there is a large background
noise_adjust = False
noise_alpha = 1.0

### Some hilbert transform options
detrend = False

pad = True
npad = 1.0              ### In units of full waveform length
pad_mode = 'constant'   ### 'constant' equivalent to zero-padding

### Downsampling options. Picks the smallest sufficient such that 
### the decaying oscillation is sufficiently oversampled to get good 
### amplitude reconstruction with the final Hilbert transofrm: >20*fsig 
out_fsamps = [2000.0, 5000.0, 10000.0, 15000.0, 20000.0]

out_fsamp = 20000.0







############################
###                      ###
###   PLOTTING OPTIONS   ###
###                      ###
############################


### Boolean flags for various sorts of plotting (used for debugging usually)
plot_nsamp = 5000000

plot_mod_peak_finding = False
plot_carrier_demod = False
plot_phase_demod = False
plot_downsample = False

plot_final_result = True

plot_debug = False












######################################################################
######################################################################
######################################################################
###                                                                ###
###        GO AWAY!!! DON'T TOUCH THINGS DOWN HERE UNLESS          ###
###        YOU REALLY KNOW WHAT'S GOING ON!!... THERE SHOULD       ###
###        NOT BE ANY (SIGNIFICANT) HARD-CODED OPTIONS BELOW       ###
###                                                                ###
######################################################################
######################################################################
######################################################################



if plot_carrier_demod or plot_mod_peak_finding or plot_downsample \
    or plot_phase_demod:
    ncore = 1 


def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0 * (x - mu)**2 / (2.0 * sigma**2)) + c

def ngauss(x, A, mu, sigma, c, n):
    return A * np.exp(-1.0 * np.abs(x - mu)**n / (2.0 * sigma**n)) + c


freq_vs_delay = []

for key in input_dict.keys():
    for meas in input_dict[key]:

        dir_name = os.path.join(dir_base, meas)
        files, _ = bu.find_all_fnames(dir_name, ext='.h5', sort_time=True)
        if not len(files):
            print()
            continue

        all_delay = []
        try:
            file_mod_freq = float(re.search(r"\/(\d+)Hz", dir_name)[1])
        except:
            file_mod_freq = 0.0

        files = files[file_inds[0]:file_inds[1]:file_step]

        fobj = bu.hsDat(files[0], load=True, load_attribs=True)

        nsamp = fobj.nsamp
        fsamp = fobj.fsamp
        fft_fac = bu.fft_norm(nsamp, fsamp)

        time_vec = np.arange(nsamp) * (1.0 / fsamp)
        full_freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

        sig = fobj.dat[:,0]
        ext_phase_mod = fobj.dat[:,1] * phase_conversion_fac

        sig_fft = np.fft.rfft(sig)

        fspin = full_freqs[np.argmax(np.abs(sig_fft))]
        inds = np.abs(full_freqs - fspin) < 200.0

        weights = np.abs(sig_fft[inds])**2
        true_fspin = np.sum(full_freqs[inds] * weights) / np.sum(weights)
        wspin = 2.0*np.pi*true_fspin

        out_nsamp = int(nsamp * out_fsamp / fsamp)




        def proc_file(file):

            fobj = bu.hsDat(file, load=True)
            file_time = fobj.time

            sig = fobj.dat[:,0]
            ext_phase_mod = fobj.dat[:,1] * phase_conversion_fac

            sig_fft = np.fft.rfft(sig)
            ext_phase_mod_fft = np.fft.rfft(ext_phase_mod)

            if key == 'delay':
                phi_dg = delay_phi_dg
            else:
                try:
                    phi_dg = fobj.attribs['phi_dg']
                except:
                    phi_dg = 0.0

            inds = np.abs(full_freqs - fspin) < 200.0

            true_fspin = np.average(full_freqs[inds], \
                                    weights=np.abs(sig_fft[inds])**2)

            if not file_mod_freq:
                mod_inds = (full_freqs > 5.0) * (full_freqs < 2550.0)
                peaks = bu.select_largest_n_fft_peaks(\
                            full_freqs[mod_inds], \
                            np.abs(ext_phase_mod_fft[mod_inds]), \
                            npeak=1, plot=plot_mod_peak_finding, \
                            adjust_baseline=True)

                mod_freq_guess = peaks[0,0]
            else:
                mod_freq_guess = file_mod_freq

            filt_band = [mod_freq_guess-5.0, mod_freq_guess+5.0]
            if not filt_band[0]:
                filt_band[0] += 0.01

            mod_inds = (full_freqs > filt_band[0]) * (full_freqs < filt_band[1])


            sig_amp, sig_phase, sig_debug = \
                    bu.demod(sig, true_fspin, fsamp, plot=plot_carrier_demod, \
                             filt=True, bandwidth=drive_bandwidth, harmind=1.0, \
                             pad=pad, npad=npad, pad_mode=pad_mode, debug=True,\
                             optimize_frequency=True)

            true_fspin = sig_debug['optimized_frequency']
            slope, offset = bu.detrend_linalg(sig_phase, xvec=time_vec, \
                                              coeffs=True)

            sig_phase_mod = sig_phase - (time_vec*slope + offset)
            sig_phase_mod_fft = np.fft.rfft(sig_phase_mod)

            sig_phase_mod_filt = np.fft.irfft(sig_phase_mod_fft\
                                                *np.exp(1j*np.pi/2)*mod_inds)
            ext_phase_mod_filt = np.fft.irfft(ext_phase_mod_fft*mod_inds)

            sig_phase_mod_deriv = np.gradient(sig_phase_mod_filt)\
                                    *fsamp/(2.0*np.pi*mod_freq_guess)

            sig_amp_scalar, _, sig_phase_scalar, _ = \
                bu.get_sine_amp_phase(sig_phase_mod_filt, plot=plot_phase_demod, \
                                      fit=True, cosine_fit=False, \
                                      plot_nsamp=plot_nsamp, \
                                      half_width=np.pi/3, freq=mod_freq_guess)

            sig_deriv_amp_scalar, _, sig_deriv_phase_scalar, _ = \
                bu.get_sine_amp_phase(sig_phase_mod_deriv, plot=plot_phase_demod, \
                                      fit=True, cosine_fit=False, \
                                      plot_nsamp=plot_nsamp, \
                                      half_width=np.pi/3, freq=mod_freq_guess)

            mod_amp_scalar, _, mod_phase_scalar, _ = \
                bu.get_sine_amp_phase(ext_phase_mod_filt, plot=plot_phase_demod, \
                                      fit=True, cosine_fit=False, \
                                      plot_nsamp=plot_nsamp, \
                                      half_width=np.pi/3, freq=mod_freq_guess)

            # sig_phase_amp, sig_phase_phase = \
            #         bu.demod(sig_phase_mod, \
            #                  mod_freq_guess, fsamp, \
            #                  plot=plot_phase_demod, filt=True, \
            #                  filt_band=filt_band, \
            #                  optimize_frequency=False, \
            #                  tukey=False, tukey_alpha=5.0e-4, \
            #                  detrend=False, harmind=1.0, \
            #                  pad=pad, npad=npad, pad_mode=pad_mode)

            # ext_phase_amp, ext_phase_phase = \
            #         bu.demod(ext_phase_mod, \
            #                  mod_freq_guess, fsamp, \
            #                  plot=plot_phase_demod, filt=True, \
            #                  filt_band=filt_band, \
            #                  optimize_frequency=False, \
            #                  tukey=False, tukey_alpha=5.0e-4, \
            #                  detrend=False, harmind=1.0, \
            #                  pad=pad, npad=npad, pad_mode=pad_mode)

            scaling = mod_amp_scalar / sig_amp_scalar
            scaling2 = mod_amp_scalar / sig_deriv_amp_scalar

            phase_delay = mod_phase_scalar - sig_phase_scalar
            phase_delay_2 = mod_phase_scalar - sig_deriv_phase_scalar

            # print()
            # print('dg and scaling : ', phi_dg, scaling, scaling2)
            # print('  phase delays : ', phase_delay, phase_delay_2)
            # print('   time delays : ', \
            #             (phase_delay-np.pi/2)/(2.0*np.pi*mod_freq_guess), \
            #             phase_delay_2/(2.0*np.pi*mod_freq_guess) )

            # plt.plot(scaling*np.gradient(sig_phase_mod_filt)\
            #             *fsamp/(2.0*np.pi*mod_freq_guess))
            # plt.plot(ext_phase_mod_filt)
            # plt.show()

            return (mod_freq_guess, phi_dg, scaling, \
                    phase_delay, phase_delay_2)




        all_fb_tests = Parallel(n_jobs=ncore)\
                            (delayed(proc_file)(file) for file in tqdm(files))

        all_fb_tests = np.array(all_fb_tests)
        freq_vs_delay.append(np.mean(all_fb_tests, axis=0))


        # all_fb_tests = np.array(all_fb_tests).T
        # plt.figure()
        # plt.plot(all_fb_tests[0], all_fb_tests[2])
        # plt.figure()
        # plt.plot(all_fb_tests[0], all_fb_tests[3])

        # if save_downsampled_data:
        #     print()
        #     print('Saving data to:')
        #     print(f'    {ringdown_data_path}')
        #     if not np.sum(impulse_vec):
        #         print('    ..., but no impulse found.....')
        #     print()

        #     with h5py.File(ringdown_data_path, 'w') as fobj:

        #         ### Save the data arrays
        #         fobj.create_dataset('all_time', data=all_time)
        #         fobj.create_dataset('all_lib', data=all_lib)
        #         fobj.create_dataset('all_lib_amp', data=all_lib_amp)

        #         ### Save the attributes for the measurement
        #         fobj.attrs['nfile'] = len(filenames)
        #         fobj.attrs['lib_freqs'] = lib_freqs
        #         fobj.attrs['phi_dgs'] = phi_dgs
        #         fobj.attrs['drive_amps'] = drive_amps
        #         fobj.attrs['filenames'] = filenames
        #         fobj.attrs['file_times'] = file_times
        #         fobj.attrs['impulse_vec'] = impulse_vec
        #         fobj.attrs['impulse_index'] = impulse_index



freq_vs_delay = np.array(freq_vs_delay).T
if save_gain_and_delay:
    bu.make_all_pardirs(gain_and_delay_filename, confirm=False)
    pickle.dump(freq_vs_delay, open(gain_and_delay_filename, 'wb'))

if plot_final_result:

    fig, axarr = plt.subplots(2,1, sharex=True)

    axarr[0].plot(freq_vs_delay[0], \
                  freq_vs_delay[2]/(2.0*np.pi*freq_vs_delay[0]))

    axarr[0].set_ylabel('Scaling [abs]')

    axarr[1].plot(freq_vs_delay[0], \
                  (freq_vs_delay[3]-np.pi/2)/(2.0*np.pi*freq_vs_delay[0]))
    axarr[1].plot(freq_vs_delay[0], \
                  freq_vs_delay[4]/(2.0*np.pi*freq_vs_delay[0]))

    axarr[1].set_xlabel('Frequency [Hz]')
    axarr[1].set_ylabel('Delay [s]')

    fig.tight_layout()

    plt.figure()
    plt.hist(freq_vs_delay[4]/(2.0*np.pi*freq_vs_delay[0]), 20)
    plt.tight_layout()

    plt.figure()
    plt.hist((freq_vs_delay[3]-np.pi/2)/(2.0*np.pi*freq_vs_delay[0]), 20)
    plt.tight_layout()

    plt.show()

