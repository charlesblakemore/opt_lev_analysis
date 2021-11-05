import os, sys, time, itertools, re, warnings, h5py
import numpy as np
import dill as pickle

# import matplotlib
# matplotlib.use('Qt5Agg')

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
ncore = 1
# ncore = 25

warnings.filterwarnings('ignore')



#############################
### Which data to analyze ###
#############################


dir_base = '/data/old_trap/'
processed_base = '/data/old_trap_processed/spinning/'

input_dict = {}


def formatter20200727(measString, ind, trial):
    if ind == 1:
        return os.path.join(measString, f'trial_{trial:04d}')
    else:
        return os.path.join(measString + f'_{ind}', f'trial_{trial:04d}')

beadtype = 'bangs5'
# beadtype = 'german7'
meas_base = 'bead1/spinning/dds_phase_impulse_'
input_dict['20200727'] = [formatter20200727(meas_base + meas, ind, trial) \
              for meas in ['lower_dg', 'low_dg', 'mid_dg', 'high_dg'] \
              for ind in [1, 2, 3] for trial in range(10)]
input_dict['20200727'] = [formatter20200727(meas_base + meas, ind, trial) \
              for meas in ['high_dg'] \
              for ind in [1, 2, 3] for trial in range(10)]

file_step = 1
file_inds = (15, 100)

save_downsampled_data = False


### Carrier filter stuff
bandwidth = 10000.0

notch_freqs = [49020.3, 44990.5]
notch_qs = [1000.0, 1000.0]


### Hard lower limit on libration to cut out misidentified libration features
min_libration_freq = 200


### Some hilbert transform options
detrend = False

pad = True
npad = 1.0              ### In units of full waveform length
pad_mode = 'constant'   ### 'constant' equivalent to zero-padding


### Should probably measure these monitor factors
# tabor_mon_fac = 100
# tabor_mon_fac = 100.0 * (1.0 / 0.95)
tabor_mon_fac = 100.0 * (53000.0 / 50000.0)

out_fsamps = [2000.0, 5000.0, 10000.0, 15000.0, 20000.0]


### Boolean flags for various sorts of plotting (used for debugging usually)
plot_efield_estimation = False
plot_nsamp = 1000

plot_carrier_demod = False
plot_libration_demod = False
plot_libration_peak_finding = False
plot_downsample = False

plot_final_result = True

plot_debug = False



########################################################################
########################################################################
########################################################################

if plot_carrier_demod or plot_libration_demod or plot_downsample:
    ncore = 1 


def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0 * (x - mu)**2 / (2.0 * sigma**2)) + c

def ngauss(x, A, mu, sigma, c, n):
    return A * np.exp(-1.0 * np.abs(x - mu)**n / (2.0 * sigma**n)) + c


for date in input_dict.keys():
    for meas in input_dict[date]:

        dir_name = os.path.join(dir_base, date, meas)
        ringdown_data_path = os.path.join(processed_base, date, meas + '.h5')
        bu.make_all_pardirs(ringdown_data_path)


        dipole = bu.get_dipole(date, substrs=[], verbose=True)
        rhobead = bu.rhobead[beadtype]
        Ibead = bu.get_Ibead(date=date, rhobead=rhobead)

        files, _ = bu.find_all_fnames(dir_name, ext='.h5', sort_time=True)
        if not len(files):
            print()
            continue

        files = files[file_inds[0]:file_inds[1]:file_step]

        fobj = bu.hsDat(files[0], load=True, load_attribs=True)

        nsamp = fobj.nsamp
        fsamp = fobj.fsamp
        fft_fac = bu.fft_norm(nsamp, fsamp)

        time_vec = np.arange(nsamp) * (1.0 / fsamp)
        full_freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

        vperp = fobj.dat[:,0]
        elec3 = fobj.dat[:,1]

        elec3_fft = np.fft.rfft(elec3)

        fspin = full_freqs[np.argmax(np.abs(elec3_fft))]
        inds = np.abs(full_freqs - fspin) < 200.0

        weights = np.abs(elec3_fft[inds])**2
        true_fspin = np.sum(full_freqs[inds] * weights) / np.sum(weights)
        wspin = 2.0*np.pi*true_fspin

        elec3_cut = tabor_mon_fac * elec3[:int(fsamp)]
        zeros = np.zeros_like(elec3_cut)
        voltages = [zeros, zeros, zeros, elec3_cut, zeros, zeros, zeros, zeros]
        efield = bu.trap_efield(voltages, only_x=True)[0]

        ### Factor of 2.0 for two opposing electrodes, only one of which is
        ### digitized due to the (sampling rate limitations
        efield_amp, _, _, _ = \
            bu.get_sine_amp_phase(efield, plot=plot_efield_estimation, \
                                  incoherent=True, plot_nsamp=plot_nsamp, \
                                  half_width=np.pi/3)
        efield_amp *= 2.0

        libration_guess = np.sqrt(efield_amp * dipole['val'] / Ibead['val']) \
                                    / (2.0 * np.pi)

        print('Libration guess:', libration_guess)

        # libration_fit_band = []
        # libration_filt_band = [1000.0, 1450.0]
        # libration_filt_band = [900.0, 1350.0]
        # libration_filt_band = [700.0, 1000.0]
        # libration_filt_band = [175.0, 400.0]   # 20200924 1Vpp
        # libration_filt_band = [350.0, 600.0]   # 20200924 3Vpp
        libration_bandwidth = 400

        libration_filt_band = \
                [np.min([min_libration_freq, 0.5*libration_guess]), \
                 2.5*libration_guess]


        for fsamp_ds in out_fsamps:
            if fsamp_ds >= 20.0 * libration_guess:
                break
        out_nsamp = int(nsamp * fsamp_ds / fsamp)


        def proc_file(file):

            fobj = bu.hsDat(file, load=True)
            file_time = fobj.time

            vperp = fobj.dat[:,0]
            elec3 = fobj.dat[:,1] * tabor_mon_fac

            try:
                phi_dg = fobj.attribs['phi_dg']
            except:
                phi_dg = 0.0

            inds = np.abs(full_freqs - fspin) < 200.0

            cut = int(1e5)
            zeros = np.zeros_like(elec3[:cut])
            voltages_cut = [zeros, zeros, zeros, elec3[:cut], zeros, zeros, zeros, zeros]
            efield_cut = bu.trap_efield(voltages_cut, only_x=True)[0]
            drive_amp_scalar, _, drive_phase_scalar, _ = \
                bu.get_sine_amp_phase(efield_cut, plot=plot_efield_estimation, \
                                  incoherent=True, fit=True, plot_nsamp=plot_nsamp, \
                                  half_width=np.pi/3)
            drive_amp_scalar *= 2.0

            elec3_fft = np.fft.rfft(elec3)
            true_fspin = np.average(full_freqs[inds], weights=np.abs(elec3_fft[inds])**2)

            zeros = np.zeros_like(elec3)
            voltages = [zeros, zeros, zeros, elec3, zeros, zeros, zeros, zeros]
            efield = bu.trap_efield(voltages, only_x=True)[0]

            drive_amp, drive_phase, drive_debug = \
                    bu.demod(efield, true_fspin, fsamp, plot=plot_carrier_demod, \
                             filt=True, bandwidth=1000.0, harmind=1.0, pad=pad, \
                             npad=npad, pad_mode=pad_mode, debug=True)
            carrier_amp, carrier_phase_mod, carrier_debug = \
                    bu.demod(vperp, true_fspin, fsamp, plot=plot_carrier_demod, \
                             filt=True, bandwidth=bandwidth,
                             notch_freqs=notch_freqs, notch_qs=notch_qs, \
                             tukey=False, tukey_alpha=5.0e-4, \
                             detrend=False, keep_mean=False, harmind=2.0, \
                             pad=pad, npad=npad, pad_mode=pad_mode, \
                             debug=True)
            plt.plot(drive_phase)
            plt.plot(carrier_phase_mod)
            # plt.axvline(impulse_ind)
            plt.show()
            input()
            return None

            slope, offset = bu.detrend_linalg(drive_phase, coeffs=True)

            phase_diff = np.mean(drive_phase[100:1000]) - np.mean(drive_phase[-1000:-100])
            if phase_diff > np.pi/4:
                impulse_ind = np.argmax(np.gradient(drive_phase))
                carrier_phase_mod[:impulse_ind] -= \
                        np.arange(len(drive_phase))[:impulse_ind]*slope + offset
                carrier_phase_mod[impulse_ind:] -= np.pi/2*np.sign(phase_diff) +\
                        np.arange(len(drive_phase))[impulse_ind:]*slope + offset
            else:
                carrier_phase_mod -= np.arange(len(drive_phase))*slope + offset

            sos = signal.butter(3, libration_filt_band, btype='bandpass', \
                                fs=fsamp, output='sos')

            if len(libration_filt_band):
                libration_inds = (full_freqs > libration_filt_band[0]) \
                                        * (full_freqs < libration_filt_band[1])
            else:
                libration_inds = np.abs(full_freqs - libration_guess) \
                                    < 0.5*libration_bandwidth

            phase_mod_fft = np.fft.rfft(carrier_phase_mod) * fft_fac

            lib_fit_x = full_freqs[libration_inds]
            lib_fit_y = np.abs(phase_mod_fft[libration_inds])

            try:
                try:
                    fft_delta_fac = 5.0
                    fft_peak_window = 50
                    peaks = bu.find_fft_peaks(lib_fit_x, lib_fit_y, \
                                delta_fac=fft_delta_fac, window=fft_peak_window, \
                                plot=plot_libration_peak_finding)
                    ind = np.argmax(peaks[:,1])

                except:
                    fft_delta_fac = 3.0
                    fft_peak_window = 100
                    peaks = bu.find_fft_peaks(lib_fit_x, lib_fit_y, \
                                delta_fac=fft_delta_fac, window=fft_peak_window, \
                                plot=plot_libration_peak_finding)
                    ind = np.argmax(peaks[:,1])

                true_libration_freq = peaks[ind,0]

            except:
                print('Fitting the libration peak to find the central '\
                        + 'frequency has failed. Using naive maximum')
                if plot_debug:
                    fig, ax = plt.subplots(1,1)
                    ax.set_title('WARNING: Spectrum that failed feature recognition')
                    ax.loglog(lib_fit_x, lib_fit_y)
                    fig.tight_layout()

                    plt.show()

                true_libration_freq = lib_fit_x[np.argmax(lib_fit_y)]

            libration_amp, libration_phase, lib_debug = \
                    bu.demod(carrier_phase_mod, true_libration_freq, fsamp, \
                             plot=plot_libration_demod, filt=True, \
                             filt_band=libration_filt_band, \
                             bandwidth=libration_bandwidth, \
                             tukey=False, tukey_alpha=5.0e-4, \
                             detrend=False, harmind=1.0, debug=True, \
                             pad=pad, npad=npad, pad_mode=pad_mode)

            carrier_phase_mod_filt = lib_debug['sig_filt']
            # carrier_phase_mod_filt = signal.sosfiltfilt(sos, carrier_phase_mod)

            libration_ds, time_vec_ds = \
                    signal.resample(carrier_phase_mod_filt, t=time_vec, num=out_nsamp)
            libration_amp_ds, time_vec_ds = \
                    signal.resample(libration_amp, t=time_vec, num=out_nsamp)


            if plot_downsample:
                short_dt = time_vec_ds[1]-time_vec_ds[0]
                short_freqs = np.fft.rfftfreq(out_nsamp, d=short_dt)
                short_fac = bu.fft_norm(out_nsamp, 1.0/short_dt)

                fig, axarr = plt.subplots(2,1,figsize=(8,6))
                axarr[0].plot(time_vec, carrier_phase_mod_filt, color='C0', \
                              lw=1, label='Filtered libration')
                axarr[0].plot(time_vec_ds, libration_ds, color='C0', lw=3, \
                              ls='--', label='Filtered/downsampled libration')
                axarr[0].plot(time_vec, libration_amp, color='C1', lw=1, \
                              label='Demodulated amplitude')
                axarr[0].plot(time_vec_ds, libration_amp_ds, color='C1', lw=3, \
                              ls='--', label='Demodulated/downsampled amplitude')
                axarr[0].set_ylim(-1.2*np.pi/2, 1.2*np.pi/2)
                axarr[0].legend()

                axarr[1].loglog(full_freqs, \
                                fft_fac*np.abs(np.fft.rfft(carrier_phase_mod_filt)), \
                                label='Filtered libration')
                axarr[1].loglog(short_freqs, \
                                short_fac*np.abs(np.fft.rfft(libration_ds)), \
                                label='Filtered/downsampled libration')
                axarr[1].legend()

                plt.show()

                input()

            libration_ds += np.mean(carrier_phase_mod_filt)

            return (time_vec_ds, libration_ds, libration_amp_ds, \
                        true_libration_freq, phi_dg, drive_amp_scalar, \
                        file, file_time)


        all_amp = Parallel(n_jobs=ncore)\
                    (delayed(proc_file)(file) for file in tqdm(files))


        t0 = all_amp[0][-1]*1e-9

        all_time, all_lib, all_lib_amp, lib_freqs, phi_dgs, drive_amps, \
            filenames, file_times = [list(result) for result in zip(*all_amp)]

        all_time = np.array(all_time)
        all_lib = np.array(all_lib)
        all_lib_amp = np.array(all_lib_amp)

        if save_downsampled_data:
            print()
            print('Saving data to:')
            print(f'    {ringdown_data_path}')
            print()

            with h5py.File(ringdown_data_path, 'w') as fobj:

                ### Save the data arrays
                fobj.create_dataset('all_time', data=all_time)
                fobj.create_dataset('all_lib', data=all_lib)
                fobj.create_dataset('all_lib_amp', data=all_lib_amp)

                ### Save the attributes for the measurement
                fobj.attrs['nfile'] = len(filenames)
                fobj.attrs['lib_freqs'] = lib_freqs
                fobj.attrs['phi_dgs'] = phi_dgs
                fobj.attrs['drive_amps'] = drive_amps
                fobj.attrs['filenames'] = filenames
                fobj.attrs['file_times'] = file_times

        if plot_final_result:
            for time_ind, time in enumerate(file_times):
                all_time[time_ind,:] += time*1e-9

            plt.plot(all_time.flatten() - t0, \
                     all_lib.flatten())
            plt.show()


            input()









