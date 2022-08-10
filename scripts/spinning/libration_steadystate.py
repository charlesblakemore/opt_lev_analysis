import os, sys, time, itertools, re, warnings, h5py
import numpy as np
import dill as pickle
from iminuit import Minuit, describe

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as optimize
import scipy.signal as signal
import scipy.constants as constants

from tqdm import tqdm
from joblib import Parallel, delayed
# ncore = 1
# ncore = 5
ncore = 30

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 14})





#################################
###                           ###
###   WHICH DATA TO ANALYZE   ###
###                           ###
#################################


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
#               for meas in ['many'] \
#               for ind in [1] for trial in range(10) ]

# meas_base = 'bead1/spinning/dds_phase_impulse_'
# input_dict['20200727'] = [ formatter20200727(meas_base + meas, ind, trial) \
#               for meas in ['lower_dg'] \
#               for ind in [1] for trial in range(10) ]

meas_base = 'bead1/spinning/dds_phase_impulse_'
input_dict['20200727'] = [ formatter20200727(meas_base + meas, ind, trial) \
              for meas in ['lower_dg', 'low_dg', 'mid_dg', 'high_dg', 'many', ''] \
              for ind in [1, 2, 3] for trial in range(10) ]



def formatter20200924(measString, voltage, dg, ind, trial):
    trial_str = f'trial_{trial:04d}'
    parent_str = f'{measString}_{voltage}Vpp'
    if dg:
        parent_str += f'_{dg}'
    if ind > 1:
        parent_str += f'_{ind}'
    return os.path.join(parent_str, trial_str)

meas_base = 'bead1/spinning/dds_phase_impulse'
input_dict['20200924'] = [ formatter20200924(meas_base, voltage, dg, ind, trial) \
              for voltage in [1, 2, 3, 4, 5, 6, 7, 8] \
              for dg in ['lower_dg', 'low_dg', 'mid_dg', 'high_dg', ''] \
              for ind in [1, 2, 3] for trial in range(10) ]

meas_base = 'bead1/spinning/dds_phase_impulse'
input_dict['20201030'] = [ formatter20200924(meas_base, voltage, dg, ind, trial) \
              for voltage in [3, 6, 8] \
              for dg in ['lower_dg', 'low_dg', 'mid_dg', 'high_dg', 'higher_dg', ''] \
              for ind in [1, 2, 3] for trial in range(10) ]

# meas_base = 'bead1/spinning/dds_phase_impulse'
# input_dict['20201030'] = [ formatter20200924(meas_base, voltage, dg, ind, trial) \
#               for voltage in [8] \
#               for dg in ['high_dg'] \
#               for ind in [1, 2, 3] for trial in range(10) ]



save_spectra = True


####################################
###                              ###
###   PRIOR KNOWLEDGE OF STUFF   ###
###                              ###
####################################

### Just useful for plotting really
impulse_magnitude = np.pi / 2.0

delta_t = 1.0 / 20000.0

delay = 4.0*delta_t
fix_delay = False

gamma0 = 0.01





#####################################
###                               ###
###   SIGNAL PROCESSING OPTIONS   ###
###                               ###
#####################################

files_for_std = 10
out_cut = 50

### Integration time in seconds for each contiguous file. Should probably
### have this automatically detected eventually
file_length = 10.0

libration_fit_width = 5.0

kd_to_fit = 6.0

# opt_ext = ''
opt_ext = f'_{int(kd_to_fit):d}kd_v3'



############################
###                      ###
###   PLOTTING OPTIONS   ###
###                      ###
############################

plot_raw_spectra = False

plot_individual_fits = False
plot_concatenation_fit = False
plot_concatenation_compensated_fit = False
plot_avg_fit = False
plot_squash_fit = False
plot_avg_fit_voigt = False

save_spectra_fit_plot = True
show_spectra_fit = False

### 0 = basic HO fit, concatenated data
### 1 = basic HO fit, concatenated with drift compensation
### 2 = basic HO fit, averaged with drift compensation
### 3 = squashed HO fit, averaged with drift compensation
plot_types_to_save = [2, 3]


plot_base = '/home/cblakemore/plots/'




########################################################################
########################################################################
########################################################################

if plot_raw_spectra or show_spectra_fit or plot_individual_fits \
    or plot_concatenation_fit or plot_concatenation_compensated_fit \
    or plot_avg_fit or plot_avg_fit_voigt or plot_squash_fit:
    ncore = 1



def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0 * (x - mu)**2 / (2.0 * sigma**2)) + c

def ngauss(x, A, mu, sigma, c, n):
    return A * np.exp(-1.0 * np.abs(x - mu)**n / (2.0 * sigma**n)) + c






def proc_spectra(spectra_file):

    date = re.search(r"\d{8,}", spectra_file)[0]
    rhobead = bu.rhobead[beadtype]
    Ibead = bu.get_Ibead(date=date, rhobead=rhobead)

    ### Use the context manager to hold the hdf5 file and copy the data
    ### and measurement parameters from the file
    with h5py.File(spectra_file, 'r') as fobj:

        ### Unpack the hdf5 attributes into local memory
        attrs = {}
        for key in fobj.attrs.keys():
            attrs[key] = fobj.attrs[key]

        nsamp = fobj['all_time'].shape[1]

        ### All times will be relative to this timestamp
        t0 = attrs['file_times'][0]*1e-9

        lib_freqs = attrs['lib_freqs']

        ### Find the impulse file to offset the libration vector (but 
        ### not the libration AMPLITUDE vector)
        impulse_start_file = np.argmax(np.abs(attrs['impulse_vec']))
        if impulse_start_file < files_for_std:
            print()
            print(f'Early impulse found in: {spectra_file}')
            print('    NOT COMPUTING THIS ONE')
            print()

            return spectra_file, np.mean(attrs['phi_dgs']), \
                    np.mean(attrs['drive_amps']), [], [], [], [], \
                    [], [], [], [], lambda *args: None, lambda *args: None, \
                    [], []

        time_arr = np.copy(fobj['all_time'])
        lib_arr = np.copy(fobj['all_lib'])
        lib_amp_arr = np.copy(fobj['all_lib_amp'])

        for i in range(attrs['nfile']):
            ctime = attrs['file_times'][i]*1e-9 - t0
            time_arr[i,:] += ctime

            if i > impulse_start_file:
                lib_off = impulse_magnitude \
                                * attrs['impulse_vec'][impulse_start_file]
            else:
                lib_off = 0
            lib_arr[i,:] += lib_off

    meas_phi_dg = np.mean(attrs['phi_dgs'])
    meas_drive_amp = np.mean(attrs['drive_amps'])

    ### Cropping indices to remove Gibbs phenomena
    cut_inds = [False for i in range(out_cut)] + \
               [True for i in range(nsamp - 2*out_cut)] + \
               [False for i in range(out_cut)]
    cut_inds_all = [ cut_inds ] * attrs['nfile']

    cut_inds = np.array(cut_inds)
    cut_inds_all = np.array(cut_inds_all)

    fsamp = 1.0 / (time_arr[0,1] - time_arr[0,0])


    if plot_raw_spectra:

        fig, ax = plt.subplots(1,1)

        dt = time_arr[0,1] - time_arr[0,0]
        freqs = np.fft.rfftfreq(nsamp, d=dt)
        fac = bu.fft_norm(nsamp, fsamp)

        colors = bu.get_color_map(impulse_start_file, cmap='plasma')
        for i in range(impulse_start_file):

            fft = np.fft.rfft(lib_arr[i,:])
            asd = np.abs(fft)

            ax.loglog(freqs, fac*asd, color=colors[i], alpha=plot_alpha, \
                      zorder=plot_zorder, lw=plot_lw)
            # ax.semilogy(freqs-lib_freq, fac*asd, color=colors[i], alpha=plot_alpha, \
            #             zorder=plot_zorder, lw=plot_lw)

        long_freqs = np.fft.rfftfreq(int(nsamp*impulse_start_file), d=dt)
        long_arr = lib_arr[:impulse_start_file,:].flatten()
        long_fac = bu.fft_norm(nsamp*impulse_start_file, fsamp)

        ax.loglog(long_freqs, long_fac*np.abs(np.fft.rfft(long_arr)), \
                  color='k', lw=2, zorder=4)

        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Libration ASD')

        lower = np.mean(lib_freqs[:impulse_start_file]) - 0.5*plot_freq_width
        upper = np.mean(lib_freqs[:impulse_start_file]) + 0.5*plot_freq_width

        ax.set_xlim(lower, upper)
        # ax.set_xlim(-2.0, 2.0)

        ax.set_ylim(1e-4, 1e-1)

        fig.tight_layout()
        plt.show()


    fits = []
    fit_funcs = []
    for i in range(impulse_start_file):

        # init_gamma = (fsamp / nsamp)
        init_gamma = 0.0

        fit_band = [lib_freqs[i]-5, lib_freqs[i]+5]
        # print(fit_band)
        # print(nsamp, fsamp, nsamp/fsamp)
        # plt.loglog(np.fft.rfftfreq(nsamp, d=1.0/fsamp), np.abs(np.fft.rfft(lib_arr[i,:])))
        # plt.show()
        # input()
        try:
            popt_init, pcov_init = \
                bu.fit_damped_osc_amp(lib_arr[i,:], fsamp, fit_band=fit_band, \
                                      plot=False, \
                                      freq_guess=lib_freqs[i], \
                                      gamma_guess=init_gamma)

            popt, pcov, fit_func = \
                bu.fit_damped_osc_amp(lib_arr[i,:], fsamp, fit_band=fit_band, \
                                      plot=plot_individual_fits, \
                                      optimize_fit_band=False, \
                                      freq_guess=popt_init[1], \
                                      gamma_guess=popt_init[2], \
                                      ngamma=200, return_func=True)
        except:
            popt = [1.0, lib_freqs[i], init_gamma, 0.0]
            pcov = np.zeros( (len(popt), len(popt)) )
            fit_func = lambda a,b,c,d: None

        fits.append(popt)
        fit_funcs.append(fit_func)

    fits = np.array(fits)
    # mean_lib_freq = np.mean(fits[:,1])
    mean_lib_freq = np.mean(lib_freqs[:impulse_start_file])

    kd = ( meas_phi_dg / 1024.0) * \
            (5.0*np.pi*delta_t * (2.0*np.pi*mean_lib_freq)**2)
    thermal_amp = np.sqrt(2*constants.k*300.0*gamma0/Ibead['val'])

    lib_arr_shifted = []
    for i in range(impulse_start_file):
        delta_f = mean_lib_freq - fits[i,1]

        hilbert = signal.hilbert(lib_arr[i,:])
        shifted = hilbert * np.exp(1.0j * 2.0*np.pi*delta_f * time_arr[i,:])

        lib_arr_shifted.append(shifted.real)

    lib_arr_shifted = np.array(lib_arr_shifted)
    fft_arr_shifted = np.fft.rfft(lib_arr_shifted, axis=1)

    # print(impulse_start_file, nsamp, fsamp)

    freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)
    fac = bu.fft_norm(nsamp, fsamp)
    freqs_long = np.fft.rfftfreq(impulse_start_file*nsamp, 1.0/fsamp)
    fac_long = bu.fft_norm(nsamp*impulse_start_file, fsamp)
    long_sig = lib_arr[:impulse_start_file,:].flatten()
    long_sig_shifted = lib_arr_shifted[:impulse_start_file,:].flatten()

    avg_shifted_asd = \
        fac * np.sqrt(np.mean(np.abs(fft_arr_shifted)**2, axis=0))
    avg_shifted_unc = (1.0 / np.sqrt(impulse_start_file)) * \
        fac * np.sqrt(np.std(np.abs(fft_arr_shifted)**2, axis=0))

    mean_gamma = np.mean(fits, axis=0)[2]
    if not kd:
        width = np.max([kd_to_fit*mean_gamma, 50])
    else:
        width = np.max([kd_to_fit*kd/(2.0*np.pi), 50])
    fit_band = [mean_lib_freq - 0.5*width, mean_lib_freq + 0.5*width]
    p0 = [1.0, mean_lib_freq, mean_gamma, 0.0]
    pcov0 = np.zeros( (len(p0), len(p0)) )


    try:
        popt_long, pcov_long = \
            bu.fit_damped_osc_amp(long_sig, fsamp, fit_band=fit_band, \
                                  plot=plot_concatenation_fit, \
                                  freq_guess=mean_lib_freq, gamma_guess=mean_gamma)
    except:
        popt_long = p0
        pcov_long = pcov0




    try:
        popt_long_shifted, pcov_long_shifted = \
            bu.fit_damped_osc_amp(long_sig_shifted, fsamp, fit_band=fit_band, \
                                  plot=plot_concatenation_compensated_fit, \
                                  freq_guess=mean_lib_freq, gamma_guess=mean_gamma)
    except:
        popt_long_shifted = p0
        pcov_long_shifted = pcov0




    try:
        popt_avg, pcov_avg, fit_func_avg = \
            bu.fit_damped_osc_amp(avg_shifted_asd, fsamp, \
                                  fit_band=fit_band, \
                                  plot=plot_avg_fit, \
                                  freq_guess=mean_lib_freq, \
                                  gamma_guess=mean_gamma, \
                                  sig_asd=True, asd_errs=avg_shifted_unc, \
                                  return_func=True)
        popt_avg_unc = np.sqrt(np.diagonal(pcov_avg))
    except:
        popt_avg = p0
        pcov_avg = pcov0
        popt_avg_unc = np.sqrt(np.diagonal(pcov_avg))
        fit_func_avg = lambda f,A,f0,g,c: \
                            np.sqrt(bu.damped_osc_amp(f,A,f0,g)**2 + c)


    try:
        diff = fit_band[1] - fit_band[0]
        squash_fit_band = [fit_band[0]-0.5*diff, fit_band[1]+0.5*diff]

        half_width = np.min([np.max([0.5*kd_to_fit*kd / (2.0*np.pi), 25]), 500])
        squash_fit_band = [mean_lib_freq-half_width, \
                           mean_lib_freq+half_width ]

        popt_squash, popt_squash_unc, fit_func_squash = \
            bu.fit_damped_osc_amp_squash(\
                                  avg_shifted_asd, fsamp, \
                                  fit_band=squash_fit_band, \
                                  plot=plot_squash_fit, \
                                  amp_guess=thermal_amp, \
                                  freq_guess=mean_lib_freq, \
                                  gamma_guess=gamma0, \
                                  noise_guess=1e-7, \
                                  deriv_gain_guess=kd, \
                                  deriv_delay_guess=delay, \
                                  fix_delay=fix_delay, \
                                  constant_guess=0.0, \
                                  # constant_guess=1e-7, \
                                  sig_asd=True, asd_errs=avg_shifted_unc, \
                                  return_func=True, verbose=True)

        # print(popt_squash[4], popt_squash[5])

    except: 
        popt_squash = [1.0, mean_lib_freq, mean_gamma, 0.0, 0.0, 0.0, 0.0]
        popt_squash_unc = np.zeros( len(popt_squash) )
        fit_func_squash = lambda f,A,f0,g,noise,kd,tfb,c: \
                            np.sqrt( bu.damped_osc_amp_squash(f,A,f0,g,\
                                                       noise,kd,tfb)**2 + c )


    # try:
    #     popt_avg_voigt, pcov_avg_voigt, fit_func_voigt = \
    #         bu.fit_voigt_profile(avg_shifted_asd, fsamp, fit_band=fit_band, \
    #                               plot=plot_avg_fit_voigt, \
    #                               freq_guess=mean_lib_freq, gamma_guess=mean_gamma, \
    #                               sig_asd=True, asd_errs=avg_shifted_unc, \
    #                               return_func=True)
    # except:
    #     popt_avg_voigt = [1.0, mean_lib_freq, mean_gamma, mean_gamma, 0.0]
    #     pcov_avg_voigt = np.zeros( (len(popt_avg_voigt), len(popt_avg_voigt)) )
    #     fit_func_voigt = lambda f,A,f0,s,g,c: \
    #                         np.sqrt( (A*special.voigt_profile((f-f0),s,g))**2 + c**2 )



    if show_spectra_fit or save_spectra_fit_plot:

        fit_band = squash_fit_band

        fit_freqs = np.linspace(fit_band[0], fit_band[1], 1000)

        fit_mult_span = fit_band[1] / fit_band[0]
        lower_plot_freq = fit_band[0] / (0.2 * (fit_mult_span - 1) + 1)
        upper_plot_freq = fit_band[1] * (0.2 * (fit_mult_span - 1) + 1)
        plot_inds = (freqs > lower_plot_freq) * (freqs < upper_plot_freq)
        plot_inds_long = (freqs_long > lower_plot_freq) * (freqs_long < upper_plot_freq)

        plot_inds[0] = False
        plot_inds_long[0] = False

        title_base = f'FPGA gain setting: {meas_phi_dg:0.3g},  '\
                        + f'$k_d$: {kd:0.3g} s$^{{-1}}$,' \
                        + f'\nDrive amp: {1e-3*meas_drive_amp:0.1f} kV/m'

        save_ext_plot_arr = ['_ho_appended', '_ho_appended_shifted', \
                             '_ho_shifted_average', '_ho_shifted_average_squash']

        annotation_plot_arr = ['Concatentation without\ndrift compensation', \
                               'Concatentation WITH\ndrift compensation', \
                               'Averaging WITH\ndrift compensation', \
                               'Averaging WITH\ndrift compensation']

        freq_plot_arr = [freqs_long[plot_inds_long], freqs_long[plot_inds_long], \
                            freqs[plot_inds], freqs[plot_inds]]

        data_plot_arr = \
            [fac_long * np.abs(np.fft.rfft(long_sig))[plot_inds_long], \
             fac_long * np.abs(np.fft.rfft(long_sig_shifted))[plot_inds_long], \
             avg_shifted_asd[plot_inds], \
             avg_shifted_asd[plot_inds]]

        fit_plot_arr = [popt_long, popt_long_shifted, popt_avg, popt_squash]

        func_plot_arr = [fit_func_avg, fit_func_avg, fit_func_avg, fit_func_squash]

        figs = []
        axes = []
        savenames = []

        # try:
        for i in range(4):

            if i not in plot_types_to_save:
                continue

            gamma_val = 2.0*np.pi*fit_plot_arr[i][2]
            if i != 3:
                title = title_base + ',  HO fit'
                label = f'Fit: $\\hat{{\\gamma}} = {gamma_val:0.3g}$ s$^{{-1}}$'
            else:
                title = title_base + ',  Noise-squashed HO'
                kd_val = fit_plot_arr[i][4]
                label = f'Fit: $\\hat{{\\gamma}} = {gamma_val:0.3g}$ s$^{{-1}}$\n' \
                            + f'      $\\hat{{k}}_d = {kd_val:0.3g}$ s$^{{-1}}$'

            fig, ax = plt.subplots(1,1,figsize=(8,5))

            ax.set_title(title)

            ax.loglog(freq_plot_arr[i], data_plot_arr[i]**2, zorder=3, lw=2)
            ax.loglog(fit_freqs, func_plot_arr[i](fit_freqs, *fit_plot_arr[i])**2, \
                      color='r', ls='--', lw=3, alpha=0.8, zorder=4, label=label)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Libration PSD [rad$^2$/Hz]')
            try:
                ax.set_xlim(freq_plot_arr[i][0], freq_plot_arr[i][-1])
            except:
                print()
                print('kd : ', kd)
                print(freq_plot_arr[i])
                print()
            ax.xaxis.set_major_formatter(lambda x, pos: f'{x:0.1f}')
            ax.xaxis.set_minor_formatter(lambda x, pos: f'{x:0.1f}')

            ax.legend(loc='upper right', fontsize=13)
            ax.text(0.05, 0.95, annotation_plot_arr[i], ha='left', va='top', \
                    transform=ax.transAxes, fontsize=12)

            # ylim = ax.get_ylim()
            # if ylim[0] < 0.1*fit_plot_arr[i][-1]:
            #     ax.set_ylim(0.1*fit_plot_arr[i][-1], ylim[1])

            fig.tight_layout()

            date = re.search(r"\d{8,}", spectra_file)[0]
            meas_file = 'dds_phase_impulse' + spectra_file.split('dds_phase_impulse')[-1]
            meas_name, trial = (os.path.splitext(meas_file)[0]).split('/')

            plot_name = date + '_' + meas_name + '_' + trial \
                            + save_ext_plot_arr[i] + f'{opt_ext}.svg'
            plot_path = os.path.join(plot_base, date, 'spinning', \
                                     meas_name, 'spectra', plot_name)
            bu.make_all_pardirs(plot_path, confirm=False)

            figs.append(fig)
            axes.append(ax)
            savenames.append(plot_path)

        if save_spectra_fit_plot:
            for fig_ind, fig in enumerate(figs):
                fig.savefig(savenames[fig_ind])

        if show_spectra_fit:
            plt.show()

        for fig in figs:
            plt.close(fig)
        # except:
        #     print()
        #     print('THIS ONE FUCKED UP: ')
        #     print(f'    {spectra_file:s}')
        #     print()

    save_inds = (freqs >= 10.0) * (freqs <= 3000.0)

    return spectra_file, meas_phi_dg, meas_drive_amp, fits, fit_funcs, \
            popt_long, popt_long_shifted, popt_avg, popt_avg_unc, \
            popt_squash, popt_squash_unc, fit_func_avg, fit_func_squash, \
            freqs[save_inds], avg_shifted_asd[save_inds], 







##### Build up the file paths
spectra_file_paths = []
for date in input_dict.keys():
    for meas in input_dict[date]:
        new_filename = os.path.join(processed_base, date, meas+'.h5')
        if os.path.exists(new_filename):
            spectra_file_paths.append( new_filename )

### Perform the fits in parallel
all_spectra = Parallel(n_jobs=ncore)(delayed(proc_spectra)(file) \
                                      for file in tqdm(spectra_file_paths))

### Unpack the result from the fitting
filenames, phi_dgs, drive_amps, fits, fit_funcs, popt_long, \
    popt_long_shifted, popt_avg, popt_avg_unc, popt_squash, \
    popt_squash_unc, fit_func_avg, fit_func_squash, freqs, asd = \
            [list(result) for result in zip(*all_spectra)]






### Save the fits, with a filename reference to the downsampled data
### for easy plotting
if save_spectra:

    for ind, filename in enumerate(filenames):

        date = re.search(r"\d{8,}", filename)[0]

        spectra_data_path = \
                os.path.join(processed_base, date, \
                             date+f'_libration_spectra{opt_ext}.p')

        # failed_spectra_data_path = \
        #         os.path.join(processed_base, date, \
        #                      date+f'_libration_spectras{opt_ext}_failed.p')

        # if ind == 0:
        print()
        print('Saving data to file: ')
        print(f'    {spectra_data_path:s}')
        print()
        # print('Saving failed filenames: ')
        # print(f'    {failed_spectra_data_path:s}')
        # print()


        # if not np.sum(fits[ind]):
        #     try:
        #         failed_spectra_dict = \
        #                 pickle.load(open(failed_spectra_data_path, 'rb'))
        #     except FileNotFoundError:
        #         failed_spectra_dict = {}

        #     failed_spectra_dict[phi_dgs[ind]] = (filename, drive_amps[ind])

        #     pickle.dump(failed_spectra_dict, open(failed_spectra_data_path, 'wb'))

        #     continue

        meas_phi_dg = phi_dgs[ind]
        meas_drive_amp = drive_amps[ind]

        try:
            spectra_dict = pickle.load(open(spectra_data_path, 'rb'))
        except FileNotFoundError:
            spectra_dict = {}

        if meas_phi_dg not in list(spectra_dict.keys()):
            spectra_dict[meas_phi_dg] = {}
            spectra_dict[meas_phi_dg]['paths'] = []
            spectra_dict[meas_phi_dg]['drive_amp'] = []
            spectra_dict[meas_phi_dg]['individual_fits'] = []
            spectra_dict[meas_phi_dg]['long_fit'] = []
            spectra_dict[meas_phi_dg]['long_shifted_fit'] = []
            spectra_dict[meas_phi_dg]['avg_shifted_fit'] = []
            spectra_dict[meas_phi_dg]['avg_shifted_fit_unc'] = []
            spectra_dict[meas_phi_dg]['avg_shifted_squash_fit'] = []
            spectra_dict[meas_phi_dg]['avg_shifted_squash_fit_unc'] = []
            spectra_dict[meas_phi_dg]['freqs'] = []
            spectra_dict[meas_phi_dg]['asd'] = []

        saved = False
        for pathind, path in enumerate(spectra_dict[meas_phi_dg]['paths']):
            if path == filename:
                saved = True
                print(f'Already saved this one ({filename})... overwriting!')
                break

        if saved:
            spectra_dict[meas_phi_dg]['drive_amp'][pathind] = drive_amps[ind]
            spectra_dict[meas_phi_dg]['individual_fits'][pathind] = fits[ind]
            spectra_dict[meas_phi_dg]['long_fit'][pathind] = popt_long[ind]
            spectra_dict[meas_phi_dg]['long_shifted_fit'][pathind] = popt_long_shifted[ind]
            spectra_dict[meas_phi_dg]['avg_shifted_fit'][pathind] = popt_avg[ind]
            spectra_dict[meas_phi_dg]['avg_shifted_fit_unc'][pathind] = popt_avg_unc[ind]
            spectra_dict[meas_phi_dg]['avg_shifted_squash_fit'][pathind] = popt_squash[ind]
            spectra_dict[meas_phi_dg]['avg_shifted_squash_fit_unc'][pathind] = popt_squash_unc[ind]
            spectra_dict[meas_phi_dg]['freqs'][pathind] = freqs[ind]
            spectra_dict[meas_phi_dg]['asd'][pathind] = asd[ind]

        else:
            spectra_dict[meas_phi_dg]['paths'].append( filename )
            spectra_dict[meas_phi_dg]['drive_amp'].append( drive_amps[ind] )
            spectra_dict[meas_phi_dg]['individual_fits'].append( fits[ind] )
            spectra_dict[meas_phi_dg]['long_fit'].append( popt_long[ind] )
            spectra_dict[meas_phi_dg]['long_shifted_fit'].append( popt_long_shifted[ind] )
            spectra_dict[meas_phi_dg]['avg_shifted_fit'].append( popt_avg[ind] )
            spectra_dict[meas_phi_dg]['avg_shifted_fit_unc'].append( popt_avg_unc[ind] )
            spectra_dict[meas_phi_dg]['avg_shifted_squash_fit'].append( popt_squash[ind] )
            spectra_dict[meas_phi_dg]['avg_shifted_squash_fit_unc'].append( popt_squash_unc[ind] )
            spectra_dict[meas_phi_dg]['freqs'].append( freqs[ind] )
            spectra_dict[meas_phi_dg]['asd'].append( asd[ind] )

        pickle.dump(spectra_dict, open(spectra_data_path, 'wb'))














