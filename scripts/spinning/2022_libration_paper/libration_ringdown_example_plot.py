import os, sys, re, h5py

import numpy as np
import dill as pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import cm

import bead_util as bu

plt.rcParams.update({'font.size': 14})



min_markersize = 2
max_markersize = 10


processed_base = '/data/old_trap_processed/spinning/'

filenames = [ \
             '20200727/20200727_libration_ringdowns_2efoldings.p', \
             # '20200924/20200924_libration_ringdowns_2efoldings.p', \
             # '20201030/20201030_libration_ringdowns_2efoldings.p', \
            ]

bad_paths = [ \
             '20200727/bead1/spinning/dds_phase_impulse_high_dg/trial_0008.h5', \
            ]

beadtype = 'bangs5'

# colors = bu.get_colormap(3, cmap='plasma')
# markers = ['o', 'o', 'o']

voltage_cmap = 'plasma'
markers = ['x', 'o', '+']
markers = ['s', 'o', '*']
markers = ['<', '*', '>']
markersize = 35
markeralpha = 0.5

colorpad = 0.1


# amp_to_plot = 9.0
# amp_to_plot = 27.0
amp_to_plot = 71.0  # in kV/m
trial_select = 0

impulse_magnitude = np.pi/2

all_plot_upper_lim = 100.0
efoldings_to_plot = 3.0

example_dgs = [1.0e-3, 4.0]
out_cut = 100

n_dg = 4
dg_lims = [1e-4, 15]


plot_xlim = [-0.25, 20]
# plot_xlim = [-1, 80]
plot_ylim = [3e-2, 2]


fit_func = lambda x, amp0, t0, tau, c: amp0 * np.exp(-1.0 * (x - t0) / tau) + c


nbin_per_file = 1000



def proc_ringdown(ringdown_file):

    ### Use the context manager to hold the hdf5 file and copy the data
    ### and measurement parameters from the file
    with h5py.File(ringdown_file, 'r') as fobj:

        ### Unpack the hdf5 attributes into local memory
        attrs = {}
        for key in fobj.attrs.keys():
            attrs[key] = fobj.attrs[key]

        impulse_start_file = np.argmax(np.abs(attrs['impulse_vec']))

        nsamp = fobj['all_time'].shape[1]

        ### All times will be relative to this timestamp
        t0 = attrs['file_times'][0]*1e-9
            # exit()

        lib_freqs = attrs['lib_freqs']

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

    ### Cropping indices to remove Gibbs phenomena
    cut_inds = [False for i in range(out_cut)] + \
               [True for i in range(nsamp - 2*out_cut)] + \
               [False for i in range(out_cut)]
    cut_inds_all = [ cut_inds ] * attrs['nfile']

    cut_inds = np.array(cut_inds)
    cut_inds_all = np.array(cut_inds_all)

    ### Index of the impulse within the full time vector
    impulse_start_time = time_arr[impulse_start_file,
                                  attrs['impulse_index'][impulse_start_file]]

    return time_arr, lib_arr, lib_amp_arr, cut_inds_all, \
            impulse_start_file, impulse_start_time, lib_off





fig = plt.figure(figsize=(8.5,5.0))
gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[5,3])

ax_list = []
ax_list.append( fig.add_subplot(gs[:,0]) )
ax_list.append( fig.add_subplot(gs[0,1]) )
ax_list.append( fig.add_subplot(gs[1,1]) )

nfiles = len(filenames)

tmin = np.inf
tmax = -np.inf

for fileind, filename in enumerate(filenames):
    ringdown_file = os.path.join(processed_base, filename)
    ringdown_dict = pickle.load( open(ringdown_file, 'rb') )

    phi_dgs = list(ringdown_dict.keys())
    phi_dgs.sort(key=float)

    phi_dgs = np.array(phi_dgs)
    inds = (phi_dgs > dg_lims[0]) * (phi_dgs < dg_lims[1])

    phi_dgs = phi_dgs[inds]

    phi_dg_test_vec = np.logspace(np.log10(phi_dgs[0]), np.log10(phi_dgs[-1]), n_dg)

    phi_dg_to_plot = []
    for phi_dg in phi_dg_test_vec:
        phi_dg_to_plot.append(phi_dgs[np.argmin(np.abs(phi_dgs - phi_dg))])

    color_vmin = 0.1 * phi_dg_to_plot[0]
    color_vmax = 10.0 * phi_dg_to_plot[-1]

    # small_dg = phi_dgs[np.argmin(np.abs(phi_dgs - example_dgs[0]))]
    # large_dg = phi_dgs[np.argmin(np.abs(phi_dgs - example_dgs[1]))]

    for phi_dg_ind, phi_dg in enumerate(phi_dg_to_plot):
        color = bu.get_single_color(phi_dg, log=True, vmin=color_vmin, vmax=color_vmax)

        my_dict = ringdown_dict[phi_dg]
        amps_rounded = np.around(np.array(my_dict['drive_amp'])*1e-3)
        good_inds = np.abs(amps_rounded - amp_to_plot) < 5.0

        actual_inds = np.arange(len(my_dict['paths']))[good_inds]
        actual_amp = np.mean(np.array(my_dict['drive_amp'])[good_inds])


        # try:
        path = np.array(my_dict['paths'], dtype=object)[good_inds][trial_select]
        # except IndexError:
            # trial_select -= 1

        fit = np.array(my_dict['fit'], dtype=object)[good_inds][trial_select]
        fit_unc = np.array(my_dict['unc'], dtype=object)[good_inds][trial_select]

        full_time, full_lib, full_amp, full_cut_inds, \
            start_file, start_time, lib_off = proc_ringdown(path)

        fit_x, fit_y = np.array(my_dict['data'][actual_inds[trial_select]])

        tau = fit[2]
        if tau*efoldings_to_plot > tmax:
            tmax = tau*efoldings_to_plot

        time_vec = full_time[full_cut_inds].flatten() - start_time
        dt = time_vec[1] - time_vec[0]
        if dt < tmin:
            tmin = dt

        time_inds = (time_vec > -1.0*tau) \
                        * (time_vec < all_plot_upper_lim)

        plot_x = time_vec[time_inds]
        plot_y = full_lib[full_cut_inds].flatten()[time_inds]
        plot_y_amp = full_amp[full_cut_inds].flatten()[time_inds]

        special_plot_x = plot_x[plot_x > 0]
        special_plot_y_amp = plot_y_amp[plot_x > 0]

        special_label = f'$k_d = {phi_dg:0.2g}$,  $\\hat{{\\tau}} = {tau:0.2g}$'
        label = f'{phi_dg:0.2g},  {tau:0.2g}'

        ax_list[0].plot(special_plot_x, fit_func(special_plot_x+start_time, *fit), \
                        color=color, lw=4, ls='--', zorder=5, label=label)

        plot_gap_inds = np.arange(len(plot_x)-1)[np.diff(plot_x) > 0.5]
        nplotgap = len(plot_gap_inds)        

        special_plot_gap_inds = np.arange(len(special_plot_x)-1)\
                                    [np.diff(special_plot_x) > 0.5]
        special_nplotgap = len(special_plot_gap_inds)

        if nplotgap:
            for gap_ind, plot_gap in enumerate(plot_gap_inds):
                if gap_ind == 0:
                    lower = 0
                else:
                    lower = plot_gap_inds[gap_ind-1] + 1
                upper = plot_gap
                ax_list[0].plot(plot_x[lower:upper], plot_y_amp[lower:upper], \
                        color=color, alpha=0.5, zorder=4)

                if phi_dg_ind == 0:
                    ax_list[1].plot(plot_x[lower:upper], \
                            plot_y[lower:upper]-lib_off, \
                            color=color, alpha=0.5, zorder=4)

                if phi_dg_ind == n_dg-1:
                    ax_list[2].plot(plot_x[lower:upper], \
                            plot_y[lower:upper]-lib_off, \
                            color=color, alpha=0.5, zorder=4)

            ax_list[0].plot(plot_x[plot_gap+1:], plot_y_amp[plot_gap+1:], \
                    color=color, alpha=0.5, zorder=4)#, label=label)
        else:
            # ax_list[0].plot(plot_x, plot_y, color=color, alpha=1.0, zorder=3)
            ax_list[0].plot(plot_x, plot_y_amp, color=color, alpha=0.5, zorder=4)


        if special_nplotgap:
            for gap_ind, plot_gap in enumerate(special_plot_gap_inds):
                if gap_ind == 0:
                    lower = 0
                else:
                    lower = special_plot_gap_inds[gap_ind-1] + 1
                upper = plot_gap

                if phi_dg_ind == 0:
                    ax_list[1].plot(special_plot_x[lower:upper], \
                            special_plot_y_amp[lower:upper], \
                            color=color, alpha=1.0, zorder=4, lw=2)

                if phi_dg_ind == n_dg-1:
                    ax_list[2].plot(special_plot_x[lower:upper], \
                            special_plot_y_amp[lower:upper], \
                            color=color, alpha=1.0, zorder=4, lw=2)

            if phi_dg_ind == 0:
                ax_list[1].plot(special_plot_x[plot_gap+1:], \
                        special_plot_y_amp[plot_gap+1:], \
                        color=color, alpha=1.0, zorder=4, lw=2, \
                        label=special_label)

            if phi_dg_ind == n_dg-1:
                ax_list[2].plot(special_plot_x[plot_gap+1:], \
                        special_plot_y_amp[plot_gap+1:], \
                        color=color, alpha=1.0, zorder=4, lw=2, \
                        label=special_label)
        else:
            if phi_dg_ind == 0:
                ax_list[1].plot(plot_x, plot_y, color=color, alpha=0.5, zorder=4)
                ax_list[1].plot(plot_x, plot_y_amp, color=color, alpha=1.0, zorder=4, \
                                lw=2, label=special_label)

            if phi_dg_ind == n_dg-1:
                ax_list[2].plot(plot_x, plot_y, color=color, alpha=0.5, zorder=4)
                ax_list[2].plot(plot_x, plot_y_amp, color=color, alpha=1.0, zorder=4, \
                                lw=2, label=special_label)

        if phi_dg_ind == 0:
            ax_list[1].set_xlim(-0.1*tau, efoldings_to_plot*tau)
        if phi_dg_ind == n_dg-1:
            ax_list[2].set_xlim(-0.1*tau, efoldings_to_plot*tau)




    # xlim = ax_list[0].get_xlim()
    ax_list[0].set_yscale('log')
    ax_list[0].set_xlim(*plot_xlim)
    ax_list[0].set_ylim(*plot_ylim)
    for i in [1,2]:
        ax_list[i].set_ylim(-1.2*np.pi/2.0, 1.2*np.pi/2.0)
        ax_list[i].set_yticks([-np.pi/2.0, 0, np.pi/2.0])
        ax_list[i].set_yticklabels(['$\\pi/2$', '0', '$\\pi/2$'])
        # ax_list[i].set_yticklabels(['$\\frac{\\pi}{2}$', '0', '$\\frac{\\pi}{2}$'])

    ax_list[0].set_ylabel('Libration Envelope [rad]')
    ax_list[0].set_xlabel('Time [s]')
    ax_list[2].set_xlabel('Time [s]')

    ax_list[0].legend(loc='lower right', ncol=2, fontsize=12, framealpha=1, \
                      title='$k_d$ [arb],  $\\hat{\\tau}$ [s]')
    ax_list[1].legend(loc='lower right', fontsize=12, framealpha=1)
    ax_list[2].legend(loc='lower right', fontsize=12, framealpha=1)

    # axarr[0].grid(axis='y', which='major', lw=.75, color='k', ls='-', alpha=0.2)
    # axarr[0].grid(axis='y', which='minor', lw=.75, color='k', ls='--', alpha=0.2)
    # axarr[1].grid(axis='both', which='major', lw=1, color='k', ls='-', alpha=0.2)
    # axarr[1].grid(axis='both', which='minor', lw=1, color='k', ls='--', alpha=0.2)

    fig.tight_layout()
    # plt.subplots_adjust(wspace=0.05)

    plt.show()