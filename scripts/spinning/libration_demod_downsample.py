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

ncore = 1
# ncore = 25

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


def formatter20200727(measString, ind, trial):
    if ind == 1:
        return os.path.join(measString, f'trial_{trial:04d}')
    else:
        return os.path.join(measString + f'_{ind}', f'trial_{trial:04d}')

# meas_base = 'bead1/spinning/dds_phase_impulse_'
# input_dict['20200727'] = [ formatter20200727(meas_base + meas, ind, trial) \
#               for meas in ['mid_dg'] \
#               for ind in [1] for trial in [4] ]

meas_base = 'bead1/spinning/dds_phase_impulse_'
input_dict['20200727'] = [ formatter20200727(meas_base + meas, ind, trial) \
              for meas in ['many'] \
              for ind in [1] for trial in range(10) ]

# meas_base = 'bead1/spinning/dds_phase_impulse_'
# input_dict['20200727'] = [ formatter20200727(meas_base + meas, ind, trial) \
#               for meas in ['lower_dg', 'low_dg', 'mid_dg', 'high_dg', ''] \
#               for ind in [1, 2, 3] for trial in range(10) ]



def formatter20200924(measString, voltage, dg, ind, trial):
    trial_str = f'trial_{trial:04d}'
    parent_str = f'{measString}_{voltage}Vpp'
    if dg:
        parent_str += f'_{dg}'
    if ind > 1:
        parent_str += f'_{ind}'
    return os.path.join(parent_str, trial_str)

# meas_base = 'bead1/spinning/dds_phase_impulse'
# input_dict['20200924'] = [ formatter20200924(meas_base, voltage, dg, ind, trial) \
#               for voltage in [1, 2, 3, 4, 5, 6, 7, 8] \
#               for dg in ['lower_dg', 'low_dg']#, 'mid_dg', 'high_dg', ''] \
#               for ind in [1, 2, 3] for trial in range(10) ]

# meas_base = 'bead1/spinning/dds_phase_impulse'
# input_dict['20200924'] = [ formatter20200924(meas_base, voltage, dg, ind, trial) \
#               for voltage in [1, 2, 3, 4, 5, 6, 7, 8] \
#               for dg in ['mid_dg', 'high_dg', ''] \
#               for ind in [1, 2, 3] for trial in range(10) ]

# meas_base = 'bead1/spinning/dds_phase_impulse'
# input_dict['20200924'] = [ formatter20200924(meas_base, voltage, dg, ind, trial) \
#               for voltage in [3] \
#               for dg in ['high_dg'] \
#               for ind in [1] for trial in [4,5,6,7,8,9] ] #range(10) ]


# ### The same formatter for 20200924 works for 20201030 as the same convention
# ### for naming was followed. The values of voltage and ranges on 'dg' were different
# ### though so we have our own path constructor statement here
# meas_base = 'bead1/spinning/dds_phase_impulse'
# input_dict['20201030'] = [ formatter20200924(meas_base, voltage, dg, ind, trial) \
#               for voltage in [3, 6, 8] \
#               for dg in ['lower_dg', 'low_dg', 'mid_dg', 'high_dg', 'higher_dg', ''] \
#               for ind in [1, 2, 3] for trial in range(10) ]


dipole_substrs = { '20200727': [''], \
                   '20200924': [''], \
                   '20201030': ['initial'] }

file_step = 1
# file_inds = (0, 100)
# file_inds = (14, 23)
file_inds = (20, 100)

save_downsampled_data = False





####################################
###                              ###
###   PRIOR KNOWLEDGE OF STUFF   ###
###                              ###
####################################


impulse_magnitude = np.pi / 2.0

ni_dac_bandwidth = 140000.0  ### From +-20V step settling time
tabor_bandwidth = 200000.0   ### limit from input side

### Empirically determined value. Only really helps with edge effects
### either at the start/end of an integration, or at an impulse. For
### the ringdown fitting, these edges are usually excluded regardless
# user_impulse_bandwidth = 0.0
# user_impulse_bandwidth = 200.0
user_impulse_bandwidth = np.min([tabor_bandwidth, ni_dac_bandwidth])

notch_freqs = [49020.3, 44990.5]
notch_qs = [1000.0, 1000.0]

### Should probably measure these monitor factors
# tabor_mon_fac = 100
# tabor_mon_fac = 100.0 * (1.0 / 0.95)
tabor_mon_fac = 100.0 * (53000.0 / 50000.0)







#####################################
###                               ###
###   SIGNAL PROCESSING OPTIONS   ###
###                               ###
#####################################


### Carrier filter stuff
# drive_bandwidth = 1000.0
drive_bandwidth = 50000.0
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







############################
###                      ###
###   PLOTTING OPTIONS   ###
###                      ###
############################


### Boolean flags for various sorts of plotting (used for debugging usually)
plot_efield_estimation = False
plot_nsamp = 10000

plot_drive_demod = True
plot_carrier_demod = False
plot_libration_peak_finding = False
plot_libration_demod = False
plot_downsample = False

plot_final_result = False

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



if plot_drive_demod or plot_carrier_demod or plot_libration_demod\
    or plot_downsample or plot_efield_estimation \
    or plot_libration_peak_finding:# or plot_debug:
    ncore = 1 


def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0 * (x - mu)**2 / (2.0 * sigma**2)) + c

def ngauss(x, A, mu, sigma, c, n):
    return A * np.exp(-1.0 * np.abs(x - mu)**n / (2.0 * sigma**n)) + c


for date in input_dict.keys():
    for meas in input_dict[date]:

        dir_name = os.path.join(dir_base, date, meas)
        files, _ = bu.find_all_fnames(dir_name, ext='.h5', sort_time=True)
        if not len(files):
            print()
            continue

        all_delay = []

        ringdown_data_path = os.path.join(processed_base, date, meas + '.h5')
        bu.make_all_pardirs(ringdown_data_path, confirm=False)

        dipole = bu.get_dipole(date, substrs=dipole_substrs[date], verbose=True)
        rhobead = bu.rhobead[beadtype]
        Ibead = bu.get_Ibead(date=date, rhobead=rhobead)


        files = files[file_inds[0]:file_inds[1]:file_step]

        fobj = bu.hsDat(files[1], load=True, load_attribs=True)

        nsamp = fobj.nsamp
        fsamp = fobj.fsamp
        fft_fac = bu.fft_norm(nsamp, fsamp)

        time_vec = np.arange(nsamp) * (1.0 / fsamp)
        full_freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

        vperp = fobj.dat[:,0]
        elec3 = fobj.dat[:,1] * tabor_mon_fac

        elec3_fft = np.fft.rfft(elec3)

        fspin = full_freqs[np.argmax(np.abs(elec3_fft))]
        inds = np.abs(full_freqs - fspin) < 200.0

        weights = np.abs(elec3_fft[inds])**2
        true_fspin = np.sum(full_freqs[inds] * weights) / np.sum(weights)
        wspin = 2.0*np.pi*true_fspin

        elec3_cut = elec3[:int(fsamp)]
        zeros = np.zeros_like(elec3_cut)
        voltages = [zeros, zeros, zeros, elec3_cut, zeros, zeros, zeros, zeros]
        efield = bu.trap_efield(voltages, only_x=True)[0]

        ### Factor of 2.0 for two opposing electrodes, only one of which is
        ### digitized due to the sampling rate limitations
        efield_amp, _, _, _ = \
            bu.get_sine_amp_phase(efield, plot=plot_efield_estimation, \
                                  incoherent=True, plot_nsamp=plot_nsamp, \
                                  half_width=np.pi/3)

        libration_guess = np.sqrt(efield_amp * dipole['val'] / Ibead['val']) \
                                    / (2.0 * np.pi)

        print('Libration guess:', libration_guess)

        libration_bandwidth = 400

        libration_filt_band = \
                [ np.min(np.abs([min_libration_freq, 0.75*libration_guess])), \
                  np.max(np.abs([max_libration_freq, 1.5*libration_guess])) ]

        ### Decide on the smallest acceptable downsampling frequency to use
        ### from the given options and based on the estimate of the libration
        ### frequency
        for fsamp_ds in out_fsamps:
            if fsamp_ds >= 20.0 * libration_guess:
                break
        out_nsamp = int(nsamp * fsamp_ds / fsamp)

        ### If the bandwidth of the impulse is unkonwn, here we try to 
        ### guess the band limit of the impulse based on the applied 
        ### filters. Either of the two values below would band limit the
        ### signal in the analysis pipeline
        if user_impulse_bandwidth == 0.0:
            val1 = 0.5*signal_bandwidth
            val2 = libration_filt_band[1]
            user_impulse_bandwidth = np.min([val1, val2])


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

            if len(libration_filt_band):
                libration_inds = (full_freqs > libration_filt_band[0]) \
                                        * (full_freqs < libration_filt_band[1])
            else:
                libration_inds = np.abs(full_freqs - libration_guess) \
                                    < 0.5*libration_bandwidth

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
            efield *= 2.0

            drive_amp, drive_phase, drive_debug = \
                    bu.demod(efield, true_fspin, fsamp, plot=plot_drive_demod, \
                             filt=True, bandwidth=drive_bandwidth, harmind=1.0, \
                             pad=pad, npad=npad, pad_mode=pad_mode, debug=True,\
                             optimize_frequency=True)

            true_fspin = drive_debug['optimized_frequency']

            ### Look for a phase impulse in the drive, given that's the type
            ### of data we're usually processing
            deriv = np.gradient(drive_phase, time_vec[1]-time_vec[0]) \
                                / (2.0 * np.pi)

            ### Threshold determined empirically by plotting. Should probably
            ### depend on the magnitude of the impulse so we've bootstrapped
            ### that in using the magnitude of impulse for which the empirical
            ### threshold was determined
            impulse_inds = np.abs(deriv) > \
                100.0*(impulse_magnitude/(np.pi/2)) * np.std(deriv)
            impulse_inds[:int(0.001*nsamp)] = 0.0
            impulse_inds[int(0.999*nsamp):] = 0.0
            yes_impulse = np.sum(impulse_inds)

            if yes_impulse:
                ### Find the middle of the impulse. With the filtering 
                ### applied, the derivative is very smooth (no HF terms)
                impulse_ind = int(np.mean( np.arange(nsamp)[impulse_inds] ))

                ### It's "bad" if it's too close to the ends, where there may 
                ### residual artifacts from the hilbert transform
                bad_impulse = (impulse_ind < 50) or (nsamp - impulse_ind < 50)
            else:
                bad_impulse = True


            if yes_impulse and not bad_impulse:
                ### Define some indices to fit the phase of the digitized
                ### drive signal, in order to extract any residual frequency
                ### offset between digitizer and source
                lower_inds = [int(0.002*nsamp), int(0.96*impulse_ind)]
                upper_inds = [int(1.04*impulse_ind), int(0.998*nsamp)]

                ### Phase impulse were applied in both directions
                impulse_sign = np.sign(deriv[impulse_ind])

                fit_x = np.concatenate(\
                            (time_vec[lower_inds[0]:lower_inds[1]],\
                             time_vec[upper_inds[0]:upper_inds[1]]) )
                fit_y = np.concatenate(\
                            (drive_phase[lower_inds[0]:lower_inds[1]],\
                             drive_phase[upper_inds[0]:upper_inds[1]] \
                                - impulse_sign*impulse_magnitude) )
                slope, offset = bu.detrend_linalg(fit_y, xvec=fit_x, \
                                                  coeffs=True)

            else:
                impulse_ind = 0
                impulse_sign = 0.0
                temp_inds = [int(0.005*nsamp), int(0.995*nsamp)]
                slope, offset = bu.detrend_linalg(drive_phase[temp_inds], \
                                                  xvec=time_vec[temp_inds], \
                                                  coeffs=True)

            carrier_amp, carrier_phase, carrier_debug = \
                    bu.demod(vperp, true_fspin, fsamp, plot=plot_carrier_demod, \
                             filt=True, bandwidth=signal_bandwidth,
                             notch_freqs=notch_freqs, notch_qs=notch_qs, \
                             tukey=False, tukey_alpha=5.0e-4, \
                             detrend=False, keep_mean=False, harmind=2.0, \
                             pad=pad, npad=npad, pad_mode=pad_mode, \
                             optimize_frequency=False, debug=True)

            carrier_phase_mod = carrier_phase - (time_vec*slope + offset)
            drive_phase_mod = drive_phase - (time_vec*slope + offset)

            phase_mod_fft = np.fft.rfft(carrier_phase_mod) * fft_fac
            drive_phase_mod_fft = np.fft.rfft(drive_phase_mod) * fft_fac

            plt.plot(drive_phase_mod)
            plt.figure()
            plt.loglog(full_freqs, np.abs(drive_phase_mod_fft)**2, 'o')
            plt.figure()
            plt.loglog(full_freqs, np.abs(phase_mod_fft)**2, 'o')
            plt.show()

            lib_fit_x = full_freqs[libration_inds]
            lib_fit_y = np.abs(phase_mod_fft[libration_inds])

            if noise_adjust:
                scaling = lib_fit_x**noise_alpha
                scaling *= 1.0 / np.mean(scaling)
                lib_fit_y *= lib_fit_x**noise_alpha

            try:
                peaks = bu.select_largest_n_fft_peaks(\
                            lib_fit_x, lib_fit_y, \
                            npeak=1, plot=plot_libration_peak_finding, \
                            adjust_baseline=True)

                true_libration_freq = peaks[0,0]

            except:
                print()
                print('Fitting the libration peak to find the central '\
                        + 'frequency has failed. Using naive maximum')
                print(f'    {dir_name}')
                if plot_debug:
                    fig, ax = plt.subplots(1,1)
                    ax.set_title('WARNING: Spectrum that failed feature recognition')
                    ax.loglog(lib_fit_x, lib_fit_y)
                    fig.tight_layout()
                    plt.show()

                true_libration_freq = lib_fit_x[np.argmax(lib_fit_y)]
            

            new_libration_filt_band = [0.5*true_libration_freq, 2.0*true_libration_freq]
            sos = signal.butter(3, new_libration_filt_band, btype='bandpass', \
                                fs=fsamp, output='sos')
            carrier_phase_mod_filt = signal.sosfiltfilt(sos, carrier_phase_mod)

            if yes_impulse and not bad_impulse:

                impulse = np.zeros_like(drive_phase)
                impulse[impulse_ind:] += impulse_sign*impulse_magnitude

                impulse_bandwidth = np.min([user_impulse_bandwidth, \
                                            0.5*signal_bandwidth])

                sos_impulse = signal.butter(3, impulse_bandwidth, \
                                            btype='lowpass', \
                                            fs=fsamp, output='sos')
                sos_impulse2 = signal.butter(3, new_libration_filt_band, \
                                             btype='bandstop', \
                                             fs=fsamp, output='sos')

                impulse_filt = signal.sosfiltfilt(sos_impulse, impulse)
                impulse_filt2 = signal.sosfiltfilt(sos_impulse2, impulse_filt)

                carrier_phase_mod_filt += impulse_filt2
                carrier_phase_mod_to_demod = np.copy(carrier_phase_mod_filt)
                carrier_phase_mod_to_demod[impulse_ind:] \
                                -= impulse_sign*impulse_magnitude
            else:
                carrier_phase_mod_to_demod = np.copy(carrier_phase_mod_filt)

            libration_amp, libration_phase, lib_debug = \
                    bu.demod(carrier_phase_mod_to_demod, \
                             true_libration_freq, fsamp, \
                             plot=plot_libration_demod, filt=False, \
                             filt_band=new_libration_filt_band, \
                             bandwidth=libration_bandwidth, \
                             optimize_frequency=False, \
                             tukey=False, tukey_alpha=5.0e-4, \
                             detrend=False, harmind=1.0, debug=True, \
                             pad=pad, npad=npad, pad_mode=pad_mode)

            # true_libration_freq = lib_debug['optimized_frequency']

            libration_ds, time_vec_ds = \
                    signal.resample(carrier_phase_mod_filt, t=time_vec, num=out_nsamp)
            libration_amp_ds, time_vec_ds = \
                    signal.resample(libration_amp, t=time_vec, num=out_nsamp)

            new_impulse_ind = int(impulse_ind * out_nsamp / nsamp)


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
                axarr[0].set_ylim(-np.pi, np.pi)
                axarr[0].legend(fontsize=10)

                axarr[1].loglog(full_freqs, \
                                fft_fac*np.abs(np.fft.rfft(carrier_phase_mod_filt)), \
                                label='Filtered libration')
                axarr[1].loglog(short_freqs, \
                                short_fac*np.abs(np.fft.rfft(libration_ds)), \
                                label='Filtered/downsampled libration')
                axarr[1].legend(fontsize=10)

                plt.show()

                input()

            # libration_ds += np.mean(carrier_phase_mod_filt)

            return (time_vec_ds, libration_ds, libration_amp_ds, \
                        true_libration_freq, phi_dg, drive_amp_scalar, \
                        file, file_time, impulse_sign, new_impulse_ind)


        all_amp = Parallel(n_jobs=ncore)\
                    (delayed(proc_file)(file) for file in tqdm(files))

        all_time, all_lib, all_lib_amp, lib_freqs, phi_dgs, drive_amps, \
            filenames, file_times, impulse_vec, impulse_index \
                    = [list(result) for result in zip(*all_amp)]

        t0 = file_times[0]*1e-9
        # print(np.array(file_times)*1e-9 - t0)

        all_time = np.array(all_time)
        all_lib = np.array(all_lib)
        all_lib_amp = np.array(all_lib_amp)

        if save_downsampled_data:
            print()
            print('Saving data to:')
            print(f'    {ringdown_data_path}')
            if not np.sum(impulse_vec):
                print('    ..., but no impulse found.....')
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
                fobj.attrs['impulse_vec'] = impulse_vec
                fobj.attrs['impulse_index'] = impulse_index

        if plot_final_result:
            offset = 0
            first = False

            fig, ax = plt.subplots(1,1)

            plot_ind = np.argmax(impulse_vec)

            # for time_ind, time in enumerate(file_times):
            #     all_time[time_ind,:] += time*1e-9
            #     all_lib[time_ind,:] += offset
            #     if impulse_vec[time_ind]:
            #         offset += impulse_vec[time_ind]*impulse_magnitude
            # ax.plot(all_time.flatten()[::100] - t0, all_lib.flatten()[::100])

            print(f'Plotting file {plot_ind}')
            ax.plot(all_time[plot_ind] + file_times[plot_ind]*1e-9 - t0, all_lib[plot_ind])

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Libration [rad]')

            fig.tight_layout()
            plt.show()


            input()









