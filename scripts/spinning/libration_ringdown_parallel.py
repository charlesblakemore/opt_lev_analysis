import os, sys, time, itertools, re, warnings, h5py
import numpy as np
import dill as pickle
from iminuit import Minuit, describe

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
#               for meas in ['mid_dg'] \
#               for ind in [1] for trial in [4] ]

meas_base = 'bead1/spinning/dds_phase_impulse_'
input_dict['20200727'] = [ formatter20200727(meas_base + meas, ind, trial) \
              for meas in ['lower_dg', 'low_dg', 'mid_dg', 'high_dg'] \
              for ind in [1, 2, 3] for trial in range(10) ]

meas_base = 'bead1/spinning/dds_phase_impulse_'
input_dict['20200727'] = [ formatter20200727(meas_base + meas, ind, trial) \
              for meas in ['lower_dg', 'low_dg', 'mid_dg', 'high_dg', 'many'] \
              for ind in [1, 2, 3] for trial in range(10) ]



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
#               for dg in ['lower_dg', 'low_dg', 'mid_dg', 'high_dg', ''] \
#               for ind in [1, 2, 3] for trial in range(10) ]

# meas_base = 'bead1/spinning/dds_phase_impulse'
# input_dict['20200924'] = [ formatter20200924(meas_base, voltage, dg, ind, trial) \
#               for voltage in [3] \
#               for dg in ['high_dg'] \
#               for ind in [1] for trial in [3,4,5,6,7]]#range(10) ]

# meas_base = 'bead1/spinning/dds_phase_impulse'
# input_dict['20201030'] = [ formatter20200924(meas_base, voltage, dg, ind, trial) \
#               for voltage in [3, 6, 8] \
#               for dg in ['lower_dg', 'low_dg', 'mid_dg', 'high_dg', 'higher_dg', ''] \
#               for ind in [1, 2, 3] for trial in range(10) ]



save_ringdowns = True




####################################
###                              ###
###   PRIOR KNOWLEDGE OF STUFF   ###
###                              ###
####################################

impulse_magnitude = np.pi / 2.0






#####################################
###                               ###
###   SIGNAL PROCESSING OPTIONS   ###
###                               ###
#####################################

files_for_std = 10
out_cut = 100

### Efoldings to cut from the beginning of the impulse given the
### filtering artifacts that appear despite my best attempts at
### contstructing the filters faithfully
impulse_efolding_buffer = 0.1

efoldings_to_fit = 2.0
opt_ext = ''
opt_ext = f'_{int(efoldings_to_fit):d}efoldings'

bins_per_file = 1000
correlation_fac = 0.6

### Integration time in seconds for each contiguous file. Should probably
### have this automatically detected eventually
file_length = 10.0





############################
###                      ###
###   PLOTTING OPTIONS   ###
###                      ###
############################

plot_selection_criteria = False
plot_rebin = False

### Colors and alpha values for libration / libration amp curves
plot_colors = bu.get_colormap(3, cmap='plasma')[1::-1]
plot_colors = ['C0', 'C1']
plot_alphas = [1.0, 1.0]

# ylim = ()
ylim = (-1.8, 1.8)
yticks = [-np.pi/2.0, 0.0, np.pi/2.0]
yticklabels = ['$-\\pi/2$', '0', '$\\pi/2$']

save_ringdown_fit_plot = True
show_ringdown_fit = False

plot_base = '/home/cblakemore/plots/'







########################################################################
########################################################################
########################################################################

if plot_selection_criteria or plot_rebin or show_ringdown_fit:
    ncore = 1



def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0 * (x - mu)**2 / (2.0 * sigma**2)) + c

def ngauss(x, A, mu, sigma, c, n):
    return A * np.exp(-1.0 * np.abs(x - mu)**n / (2.0 * sigma**n)) + c






def proc_ringdown(ringdown_file):

    ### Use the context manager to hold the hdf5 file and copy the data
    ### and measurement parameters from the file
    with h5py.File(ringdown_file, 'r') as fobj:

        time_arr = np.copy(fobj['all_time'])
        lib_arr = np.copy(fobj['all_lib'])
        lib_amp_arr = np.copy(fobj['all_lib_amp'])

        ### Unpack the hdf5 attributes into local memory
        attrs = {}
        for key in fobj.attrs.keys():
            attrs[key] = fobj.attrs[key]

        nsamp = fobj['all_time'].shape[1]

        ### All times will be relative to this timestamp
        t0 = attrs['file_times'][0]*1e-9

        # test_ind = 0
        # check = False
        # n = 0
        # while (not check) and (n <= lib_arr.shape[0]):
        #     attrs['impulse_vec'][test_ind] = 0.0
        #     test_ind = np.argmax(np.abs(attrs['impulse_vec']))
        #     first = np.mean(lib_arr[test_ind][out_cut:out_cut+20])
        #     last = np.mean(lib_arr[test_ind][-out_cut-20:-out_cut])
        #     check = np.abs(first - last) > np.pi / 4
        #     n += 1

        ### Find the impulse file to offset the libration vector (but 
        ### not the libration AMPLITUDE vector)
        impulse_start_file = np.argmax(np.abs(attrs['impulse_vec']))

        if impulse_start_file < files_for_std:
            print()
            print(f'Early impulse found in: {ringdown_file}')
            print('    NOT COMPUTING THIS ONE')
            print(attrs['impulse_vec'])
            print(attrs['impulse_index'])
            print()
            return ringdown_file, np.mean(attrs['phi_dgs']), \
                    np.mean(attrs['lib_freqs']), \
                    np.mean(attrs['drive_amps']), \
                    [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], \
                    0.0, [0.0, 0.0], [], []
            # exit()


        for i in range(attrs['nfile']):
            ctime = attrs['file_times'][i]*1e-9 - t0
            time_arr[i,:] += ctime

            if i > impulse_start_file:
                lib_off = impulse_magnitude \
                                * attrs['impulse_vec'][impulse_start_file]
            else:
                lib_off = 0
            lib_arr[i,:] += lib_off

    ### Cropping indices to remove Gibbs phenomena
    cut_inds = [False for i in range(out_cut)] + \
               [True for i in range(nsamp - 2*out_cut)] + \
               [False for i in range(out_cut)]
    cut_inds_all = [ cut_inds ] * attrs['nfile']

    cut_inds = np.array(cut_inds)
    cut_inds_all = np.array(cut_inds_all)

    # print(attrs['impulse_vec'])
    # input()

    ### Index of the impulse within the full time vector
    impulse_start_time = time_arr[impulse_start_file,
                                  attrs['impulse_index'][impulse_start_file]]

    ### Some stuff to figure out when the ringdown has dropped low enough
    cross_thresh = impulse_magnitude / np.e
    efolding_times = np.array([])
    for i in range(attrs['nfile']):
        crossings = np.array(bu.zerocross_pos2neg(lib_amp_arr[i,cut_inds] - cross_thresh))
        efolding_times = np.concatenate( (efolding_times, \
                                          time_arr[i,cut_inds][crossings]) )

    ### If no threshold crossing is detected, print and exit
    if not len(efolding_times):
        print('DID NOT DETECT A RINGDOWN!' + f' - {ringdown_file}')
        return ringdown_file, np.mean(attrs['phi_dgs']), \
                np.mean(attrs['lib_freqs']), \
                np.mean(attrs['drive_amps']), \
                [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], \
                0.0, [0.0, 0.0], [], []

    efolding_times = efolding_times[efolding_times > impulse_start_time]
    efolding_inds = ((efolding_times - impulse_start_time) \
                        < 2.0*(efolding_times[0] - impulse_start_time))

    efolding_time = np.mean(efolding_times[efolding_inds]) - impulse_start_time
    impulse_end_time = impulse_start_time + efoldings_to_fit*efolding_time

    impulse_fit_offset = impulse_efolding_buffer*efolding_time


    if plot_selection_criteria:

        print()
        print('Plotting selectron criteria for:')
        print(f'    {ringdown_file}')

        fig, ax = plt.subplots(1,1)

        ax.plot(time_arr[cut_inds_all].flatten(), \
                lib_arr[cut_inds_all].flatten())
        ax.plot(time_arr[cut_inds_all].flatten(), \
                lib_amp_arr[cut_inds_all].flatten())
        ax.axvline(impulse_start_time, color='g', label='Impulse start')
        ax.axvline(efolding_times[efolding_times > impulse_start_time][0], \
                    color='r', ls='--')
        ax.axvline(efolding_times[efolding_times > impulse_start_time][-1], \
                    color='r', ls='--')
        ax.axvline(impulse_start_time+efolding_time, color='r', label='One $e$-folding')
        ax.axvline(impulse_end_time, color='b', label='Impulse End')
        ax.axhline(cross_thresh, color='k')
        ax.legend(loc='upper right', fontsize=10)

        ax.set_title('Impulse fitting selection criteria')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Libration [rad]')

        # ax.set_xlim(impulse_start_time-efolding_time, impulse_end_time+efolding_time)

        fig.tight_layout()
        plt.show()

        input()

    ringdown_length = impulse_end_time - impulse_start_time

    calc_times = []

    fit_times = []
    fit_amps = []
    fit_uncs = []

    init_amps = []

    for i in range(attrs['nfile']):

        if (i < files_for_std) and (i < impulse_start_file):
            init_amps.append(np.mean(lib_amp_arr[i,cut_inds]))
            continue

        if i < impulse_start_file:
            continue

        if (attrs['file_times'][i]*1e-9 - t0) > impulse_end_time:
            break

        if time_arr[i,-1] > impulse_end_time:
            diff = impulse_end_time - time_arr[i,0]
            nbin = np.max( [int(bins_per_file*diff/file_length), 100] )
            time_inds = (time_arr[i] < impulse_end_time) \
                         * (time_arr[i] > impulse_start_time + impulse_fit_offset)
        else:
            nbin = bins_per_file
            time_inds = time_arr[i] > impulse_start_time + impulse_fit_offset

        fit_inds = time_inds * cut_inds

        if not np.sum(fit_inds):
            continue

        fit_time, fit_amp, fit_unc = \
                bu.rebin(time_arr[i,fit_inds], lib_amp_arr[i,fit_inds], \
                         nbin=nbin, plot=plot_rebin, correlated_errs=True, \
                         correlation_fac=correlation_fac)

        fit_times.append(fit_time)
        fit_amps.append(fit_amp)
        fit_uncs.append(fit_unc)

    fit_x = np.concatenate(fit_times)
    fit_y = np.concatenate(fit_amps)
    fit_unc = np.concatenate(fit_uncs)

    thermal_amp = np.mean(init_amps)
    thermal_amp_unc = (1.0 / np.sqrt(len(init_amps))) * np.std(init_amps)

    plot_x = np.linspace(fit_x[0], fit_x[-1], 500)

    fit_func = lambda x, amp0, t0, tau, c: amp0 * np.exp(-1.0 * (x - t0) / tau) + c

    ### Cost function, defined with two priors: (1) on the starting amplitude 
    ### since it's defined by the level of the applied phase-jump, and (2) on 
    ### the final 'thermal' amplitude, since that's measured for ~100 seconds
    ### Prior to the impulse being applied
    npts = len(fit_x)
    def cost(amp0, t0, tau, c):
        resid = np.abs(fit_y - fit_func(fit_x, amp0, t0, tau, c))**2
        variance = fit_unc**2
        prior1 = np.abs(amp0 - impulse_magnitude)**2 / np.mean(variance)
        prior2 = np.abs(c - thermal_amp)**2 / thermal_amp_unc**2
        return (1.0 / (npts - 1.0)) * np.sum(resid / variance) + prior1 + prior2

    m = Minuit(cost, \
               amp0 = np.pi/2.0, \
               t0   = impulse_start_time, \
               tau  = efolding_time,
               c    = thermal_amp)

    ### Apply some limits to help keep the fitting well behaved
    m.limits['amp0'] = (np.pi/4.0, 6.0*np.pi/10.0)
    m.limits['t0'] = (fit_x[0]-5.0, fit_x[0]+5.0)
    m.limits['tau'] = (0.1*efolding_time, 5.0*efolding_time)
    m.limits['c'] = (0, np.pi/8.0)

    m.errordef = 1
    m.print_level = 0

    ### Do the actual minimization
    m.migrad(ncall=500000)

    ### The 'fval' attribute of the Minuit class is some internally defined
    ### python object. So let's get rid of that mess
    min_chisq = float(m.fval)

    try:
        # print(fit_unc)
        # print(thermal_amp)
        m.minos()

        ringdown_fit = np.array(m.values)
        ringdown_unc = \
            np.array([np.mean(np.abs([m.merrors['amp0'].lower, m.merrors['amp0'].upper])), \
                      np.mean(np.abs([m.merrors['t0'].lower, m.merrors['t0'].upper])), \
                      np.mean(np.abs([m.merrors['tau'].lower, m.merrors['tau'].upper])), \
                      np.mean(np.abs([m.merrors['c'].lower, m.merrors['c'].upper]))])

    except:
        print(f'MINOS FAILED! - {ringdown_file}')
        ringdown_fit = np.array([m.values['amp0'], m.values['t0'], \
                                 m.values['tau'], m.values['c']])
        ringdown_unc = np.array([m.errors['amp0'], m.errors['t0'], \
                                 m.errors['tau'], m.errors['c']])

    ringdown_fit = np.nan_to_num(ringdown_fit)
    ringdown_unc = np.nan_to_num(ringdown_unc)

    meas_phi_dg = np.mean(attrs['phi_dgs'])
    meas_drive_amp = np.mean(attrs['drive_amps'])
    meas_lib_freq = np.mean(np.array(attrs['lib_freqs'])[:impulse_start_file])

    impulse_time = np.array([impulse_start_time, impulse_end_time])





    if show_ringdown_fit or save_ringdown_fit_plot:

        title = f'Deriv. gain: {meas_phi_dg:0.3g}, ' \
                    + f'Drive amp: {1e-3*meas_drive_amp:0.1f} kV/m'
        label = f'Fit: $\\tau = ({ringdown_fit[2]:0.3g}\\pm{ringdown_unc[2]:0.2g})$ s'

        time_inds = ( time_arr > (impulse_start_time - 0.1*ringdown_length) ) \
                        * ( time_arr < (impulse_end_time + 0.1*ringdown_length) ) \
                        * cut_inds_all

        time_vec = time_arr[time_inds].flatten() - impulse_start_time
        amp_vec = lib_amp_arr[time_inds].flatten()
        lib_vec = lib_arr[time_inds].flatten()

        gap_inds = np.arange(len(time_vec)-1)[np.diff(time_vec) > 0.5]
        ngap = len(gap_inds)

        lib_vec[time_vec > 0] -= lib_off


        fig, ax = plt.subplots(1, 1, figsize=(7,4))

        if ngap:
            for gap_ind, gap in enumerate(gap_inds):
                if gap_ind == 0:
                    lower = 0
                else:
                    lower = gap_inds[gap_ind-1] + 1
                upper = gap
                ax.plot(time_vec[lower:upper], lib_vec[lower:upper], \
                        color=plot_colors[0], alpha=plot_alphas[0], zorder=1)
                ax.plot(time_vec[lower:upper], amp_vec[lower:upper], \
                        color=plot_colors[1], alpha=plot_alphas[1], zorder=2)
            ax.plot(time_vec[gap+1:], lib_vec[gap+1:], \
                    color=plot_colors[0], alpha=plot_alphas[0], zorder=1)
            ax.plot(time_vec[gap+1:], amp_vec[gap+1:], \
                    color=plot_colors[1], alpha=plot_alphas[1], zorder=2)
        else:
            ax.plot(time_vec, lib_vec, color=plot_colors[0], \
                    alpha=plot_alphas[0], zorder=1)
            ax.plot(time_vec, amp_vec, color=plot_colors[1], \
                    alpha=plot_alphas[1], zorder=2)

        fit_x -= impulse_start_time
        fit_gap_inds = np.arange(len(fit_x)-1)[np.diff(fit_x) > 0.5]
        nfitgap = len(fit_gap_inds)

        if nfitgap:
            for gap_ind, fit_gap in enumerate(fit_gap_inds):
                if gap_ind == 0:
                    lower = 0
                else:
                    lower = fit_gap_inds[gap_ind-1] + 1
                upper = fit_gap
                ax.errorbar(fit_x[lower:upper], fit_y[lower:upper], \
                            fmt='o-', color='k', zorder=4)
            ax.errorbar(fit_x[fit_gap+1:], fit_y[fit_gap+1:], yerr=fit_unc[fit_gap+1:], \
                        fmt='o-', color='k', alpha=1.0, zorder=4, label='Averaged data')
        else:
            ax.errorbar(fit_x, fit_y, yerr=fit_unc, fmt='o-', color='k', alpha=1.0, \
                        zorder=4, label='Averaged data')


        ax.plot(fit_x, fit_func(fit_x+impulse_start_time, *ringdown_fit), \
                ls='--', lw=2, color='r', label=label, zorder=5)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Libration [rad]')

        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        ax.set_xlim(-0.1*ringdown_length, 1.1*ringdown_length)

        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=12)

        fig.tight_layout()

        if show_ringdown_fit:
            plt.show()
            input()

        if save_ringdown_fit_plot:
            date = re.search(r"\d{8,}", ringdown_file)[0]

            meas_file = 'dds_phase_impulse' + ringdown_file.split('dds_phase_impulse')[-1]
            meas_name, trial = (os.path.splitext(meas_file)[0]).split('/')

            plot_name = date + '_' + meas_name + '_' + trial + opt_ext + '.svg'
            plot_path = os.path.join(plot_base, date, 'spinning', meas_name, plot_name)

            bu.make_all_pardirs(plot_path, confirm=False)
            fig.savefig(plot_path)

        plt.close(fig)

    # print('Done with: ', ringdown_file)

    return ringdown_file, meas_phi_dg, meas_drive_amp, meas_lib_freq, \
                ringdown_fit, ringdown_unc, min_chisq, impulse_time, \
                fit_x, fit_y







##### Build up the file paths
ringdown_file_paths = []
for date in input_dict.keys():
    for meas in input_dict[date]:
        new_filename = os.path.join(processed_base, date, meas+'.h5')
        if os.path.exists(new_filename):
            ringdown_file_paths.append( new_filename )

### Perform the fits in parallel
all_ringdowns = Parallel(n_jobs=ncore)(delayed(proc_ringdown)(file) \
                                      for file in tqdm(ringdown_file_paths))

### Unpack the result from the fitting
filenames, phi_dgs, drive_amps, lib_freqs, fits, uncs, \
    min_chisqs, impulse_times, xvecs, yvecs = \
            [list(result) for result in zip(*all_ringdowns)]






### Save the fits, with a filename reference to the downsampled data
### for easy plotting
if save_ringdowns:

    for ind, filename in enumerate(filenames):


        date = re.search(r"\d{8,}", filename)[0]

        ringdown_data_path = \
                os.path.join(processed_base, date, \
                             date+f'_libration_ringdowns{opt_ext}.p')

        failed_ringdown_data_path = \
                os.path.join(processed_base, date, \
                             date+f'_libration_ringdowns{opt_ext}_failed.p')

        if ind == 0:
            print()
            print('Saving data to file: ')
            print(f'    {ringdown_data_path:s}')
            print()
            print('Saving failed filenames: ')
            print(f'    {failed_ringdown_data_path:s}')
            print()


        if not np.sum(fits[ind]):
            try:
                failed_ringdown_dict = \
                        pickle.load(open(failed_ringdown_data_path, 'rb'))
            except FileNotFoundError:
                failed_ringdown_dict = {}

            failed_ringdown_dict[phi_dgs[ind]] = (filename, drive_amps[ind])

            pickle.dump(failed_ringdown_dict, open(failed_ringdown_data_path, 'wb'))

            continue

        meas_phi_dg = phi_dgs[ind]
        meas_drive_amp = drive_amps[ind]
        meas_lib_freq = lib_freqs[ind]
        fit = fits[ind]
        unc = uncs[ind]
        min_chisq = min_chisqs[ind]
        impulse_time = impulse_times[ind]
        data = np.array([xvecs[ind], yvecs[ind]])

        try:
            ringdown_dict = pickle.load(open(ringdown_data_path, 'rb'))
        except FileNotFoundError:
            ringdown_dict = {}

        if meas_phi_dg not in list(ringdown_dict.keys()):
            ringdown_dict[meas_phi_dg] = {}
            ringdown_dict[meas_phi_dg]['paths'] = []
            ringdown_dict[meas_phi_dg]['fit'] = []
            ringdown_dict[meas_phi_dg]['unc'] = []
            ringdown_dict[meas_phi_dg]['chi_sq'] = []
            ringdown_dict[meas_phi_dg]['drive_amp'] = []
            ringdown_dict[meas_phi_dg]['lib_freq'] = []
            ringdown_dict[meas_phi_dg]['impulse_time'] = []
            ringdown_dict[meas_phi_dg]['data'] = []

        saved = False
        for pathind, path in enumerate(ringdown_dict[meas_phi_dg]['paths']):
            if path == filename:
                saved = True
                print(f'Already saved this one ({filename})... overwriting!')
                break

        if saved:
            ringdown_dict[meas_phi_dg]['fit'][pathind] = fit
            ringdown_dict[meas_phi_dg]['unc'][pathind] = unc
            ringdown_dict[meas_phi_dg]['chi_sq'][pathind] = min_chisq
            ringdown_dict[meas_phi_dg]['drive_amp'][pathind] = meas_drive_amp
            ringdown_dict[meas_phi_dg]['lib_freq'][pathind] = meas_lib_freq
            ringdown_dict[meas_phi_dg]['impulse_time'][pathind] = impulse_time
            ringdown_dict[meas_phi_dg]['data'][pathind] = data

        else:
            ringdown_dict[meas_phi_dg]['paths'].append(filename)
            ringdown_dict[meas_phi_dg]['fit'].append( fit )
            ringdown_dict[meas_phi_dg]['unc'].append( unc )
            ringdown_dict[meas_phi_dg]['chi_sq'].append( min_chisq )
            ringdown_dict[meas_phi_dg]['drive_amp'].append( meas_drive_amp )
            ringdown_dict[meas_phi_dg]['lib_freq'].append( meas_lib_freq )
            ringdown_dict[meas_phi_dg]['impulse_time'].append( impulse_time )
            ringdown_dict[meas_phi_dg]['data'].append( data )

        pickle.dump(ringdown_dict, open(ringdown_data_path, 'wb'))















