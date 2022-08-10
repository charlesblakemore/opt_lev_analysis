import os, sys, re, h5py

import numpy as np
import dill as pickle

import matplotlib
backends = ['Qt5Agg', 'TkAgg', 'Agg']
matplotlib.use(backends[1])

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

spectra_fit_filenames \
        = [ \
           '20200727/20200727_libration_spectra_6kd.p', \
           # '20200924/20200924_libration_spectra_6kd.p', \
           # '20201030/20201030_libration_spectra_6kd.p', \
          ]

bad_paths = [ \
             '20200727/bead1/spinning/dds_phase_impulse_high_dg/trial_0008.h5', \
            ]

save_dir = '/home/cblakemore/plots/libration_paper_2022'
plot_name = 'fig3_libration_ringdown_examples_log_v2.svg'

save = True
show = True

beadtype = 'bangs5'

# colors = bu.get_color_map(3, cmap='plasma')
# markers = ['o', 'o', 'o']

voltage_cmap = 'plasma'
markers = ['x', 'o', '+']
markers = ['s', 'o', '*']
markers = ['<', '*', '>']
markersize = 35
markeralpha = 0.5


line_alpha = 0.85
# data_alpha = 0.6
# lightening_factor = 0.75
data_alpha = 1.0
lightening_factor = 0.525

legend_alpha = 0.9

colorpad = 0.1


# amp_to_plot = 9.0
# amp_to_plot = 27.0
amp_to_plot = 71.0  # in kV/m
trial_select = 1

impulse_magnitude = np.pi/2

delta_t = 1.0 / 20000.0

Troom = 300.0

all_plot_upper_lim = 100.0
efoldings_to_plot = 3.0

example_dg = 10
# example_dgs = [1.0e-3, 4.0]
out_cut = 100


n_dg = 4
dg_lims = [1e-2, 15]


# plot_xlim = [-0.1, 8]
plot_xlim = [3e-4, 30]
log = True

# inset_bbox = (0.6, 0.6, 0.35, 0.35)
inset_bbox = (0.06, 0.11, 0.375, 0.425)  # With x-axis label
inset_bbox = (0.06, 0.075, 0.375, 0.375)  # Without

# plot_xlim = [-1, 80]
# plot_ylim = [3e-2, 2]
plot_ylim = [2e-2, 2]


fit_func = lambda x, amp0, t0, tau, c: amp0 * np.exp(-1.0 * (x - t0) / tau) + c


nbin_per_file = 1000



fig, ax = plt.subplots(figsize=(6.5,4.0))
# gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[5,3])

ax_inset = inset_axes(ax, width="100%", height="100%", \
                      bbox_to_anchor=inset_bbox, \
                      bbox_transform=ax.transAxes, loc=3)
# ax.add_patch( plt.Rectangle((inset_bbox[0]-0.05, inset_bbox[1]-0.11), \
#                              inset_bbox[2]+0.08, inset_bbox[3]+0.15, ls='', ec='none', \
#                              fc=(1.0, 1.0, 1.0, 0.5), transform=ax.transAxes, zorder=10) )







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





nfiles = len(filenames)

tmin = np.inf
tmax = -np.inf

for fileind, filename in enumerate(filenames):
    ringdown_file = os.path.join(processed_base, filename)
    ringdown_dict = pickle.load( open(ringdown_file, 'rb') )

    spectra_file = os.path.join(processed_base, spectra_fit_filenames[fileind])
    spectra_dict = pickle.load( open(spectra_file, 'rb') )

    phi_dgs = list(ringdown_dict.keys())
    phi_dgs.sort(key=float)

    phi_dgs = np.array(phi_dgs)
    inds = (phi_dgs > dg_lims[0]) * (phi_dgs < dg_lims[1])

    phi_dgs = phi_dgs[inds]

    phi_dg_test_vec = np.logspace(np.log10(phi_dgs[0]), np.log10(phi_dgs[-1]), n_dg)

    phi_dg_to_plot = []
    phi_dg_to_plot = [0.0]
    for phi_dg in phi_dg_test_vec:
        phi_dg_to_plot.append(phi_dgs[np.argmin(np.abs(phi_dgs - phi_dg))])

    example_dg_to_plot = \
        phi_dg_to_plot[np.argmin(np.abs(np.array(phi_dg_to_plot)-example_dg))]

    color_vmin = 0.1 * phi_dg_to_plot[1]
    color_vmax = 20.0 * phi_dg_to_plot[-1]

    # small_dg = phi_dgs[np.argmin(np.abs(phi_dgs - example_dgs[0]))]
    # large_dg = phi_dgs[np.argmin(np.abs(phi_dgs - example_dgs[1]))]

    for phi_dg_ind, phi_dg in enumerate(phi_dg_to_plot):
        if phi_dg:
            color = bu.get_single_color(phi_dg, log=True, \
                                        vmin=color_vmin, \
                                        vmax=color_vmax)
        else:
            color = 'k'

        my_dict = ringdown_dict[phi_dg]
        my_spectra_dict = spectra_dict[phi_dg]
        amps_rounded = np.around(np.array(my_dict['drive_amp'])*1e-3)
        good_inds = np.abs(amps_rounded - amp_to_plot) < 5.0

        actual_inds = np.arange(len(my_dict['paths']))[good_inds]
        actual_amp = np.mean(np.array(my_dict['drive_amp'])[good_inds])


        spectra_fit = np.array(my_spectra_dict['avg_shifted_squash_fit'], \
                               dtype=object)[good_inds][trial_select]
        kd = ( phi_dg / 1024.0) * \
                (5.0*np.pi*delta_t * (2.0*np.pi*spectra_fit[1])**2)

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
                        * (time_vec < all_plot_upper_lim) \
                        * (time_vec < 30.0*tau) \
                        * (time_vec < plot_xlim[-1])

        plot_x = time_vec[time_inds]
        plot_y = full_lib[full_cut_inds].flatten()[time_inds]
        plot_y_amp = full_amp[full_cut_inds].flatten()[time_inds]

        special_plot_x = plot_x[plot_x > 0]
        special_plot_y_amp = plot_y_amp[plot_x > 0]

        kd_str = np.format_float_positional(bu.round_sig(kd,2),trim='-')
        tau_str = np.format_float_positional(bu.round_sig(tau,2),trim='-')
        special_label = f'$k_d = {kd_str:s}$ s$^{{-1}}$'
        label = f'{kd_str:s},  {tau_str:s}'
        label = bu.format_multiple_float_string( \
                        kd, tau, sig_figs=2, extra=2)

        ax.plot(special_plot_x, fit_func(special_plot_x+start_time, *fit), \
                color=color, lw=4, ls='--', zorder=5+2*int(n_dg-phi_dg_ind), \
                label=label, alpha=line_alpha)

        plot_gap_inds = np.arange(len(plot_x)-1)[np.diff(plot_x) > 0.5]
        nplotgap = len(plot_gap_inds)        

        special_plot_gap_inds = np.arange(len(special_plot_x)-1)\
                                    [np.diff(special_plot_x) > 0.5]
        special_nplotgap = len(special_plot_gap_inds)

        if nplotgap and (phi_dg == example_dg_to_plot):
            for gap_ind, plot_gap in enumerate(plot_gap_inds):
                if gap_ind == 0:
                    lower = 0
                else:
                    lower = plot_gap_inds[gap_ind-1] + 1
                upper = plot_gap


                ax_inset.plot(plot_x[lower:upper], plot_y[lower:upper]-lib_off, \
                        color=bu.lighten_color(color, lightening_factor), \
                        alpha=1.0, zorder=3)

            ax_inset.plot(plot_x[plot_gap+1], plot_y[plot_gap+1]-lib_off, \
                    color=bu.lighten_color(color, lightening_factor), \
                    alpha=1.0, zorder=3)
        elif (phi_dg == example_dg_to_plot):
            # ax.plot(plot_x, plot_y, color=color, alpha=1.0, zorder=3)
            ax_inset.plot(plot_x, plot_y-lib_off, \
                          color=bu.lighten_color(color, lightening_factor), \
                          alpha=1.0, zorder=3)


        if special_nplotgap:
            for gap_ind, plot_gap in enumerate(special_plot_gap_inds):
                if gap_ind == 0:
                    lower = 0
                else:
                    lower = special_plot_gap_inds[gap_ind-1] + 1
                upper = plot_gap
                ax.plot(special_plot_x[lower:upper], special_plot_y_amp[lower:upper], \
                        color=bu.lighten_color(color, lightening_factor), \
                        alpha=data_alpha, zorder=4+2*int(n_dg-phi_dg_ind))
                if phi_dg == example_dg_to_plot:
                    ax_inset.plot(special_plot_x[lower:upper], \
                            special_plot_y_amp[lower:upper], \
                            color=color, alpha=1.0, \
                            zorder=4, lw=2)

            ax.plot(special_plot_x[plot_gap+1:], special_plot_y_amp[plot_gap+1:], \
                    color=bu.lighten_color(color, lightening_factor), \
                    alpha=data_alpha, zorder=4+2*int(n_dg-phi_dg_ind))#, label=label)
            if phi_dg == example_dg_to_plot:
                ax_inset.plot(special_plot_x[plot_gap+1:], \
                        special_plot_y_amp[plot_gap+1:], \
                        color=color, alpha=1.0, zorder=4, lw=2, \
                        label=special_label)

        elif not special_nplotgap:
            ax.plot(special_plot_x, special_plot_y_amp, \
                    color=bu.lighten_color(color, lightening_factor), \
                    alpha=line_alpha, zorder=4+2*int(n_dg-phi_dg_ind))

            if phi_dg == example_dg_to_plot:
                ax_inset.plot(plot_x, plot_y-lib_off, \
                              color=bu.lighten_color(color, lightening_factor), \
                              alpha=1.0, zorder=4)
                ax_inset.plot(special_plot_x, special_plot_y_amp, \
                              color=color, alpha=1.0, \
                              zorder=4, lw=2, label=special_label)
                ax_inset.set_xlim(-0.1*tau, 1.1*efoldings_to_plot*tau)




    # xlim = ax.get_xlim()
    if log:
        ax.set_xscale('log')
    ax.set_xlim(*plot_xlim)

    ax.set_yscale('log')
    ax.set_ylim(*plot_ylim)

    ax_inset.set_ylim(-1.2*np.pi/2.0, 1.2*np.pi/2.0)
    ax_inset.set_yticks([-np.pi/2.0, 0, np.pi/2.0])
    ax_inset.set_yticklabels(['-$\\pi/2$', '0', '$\\pi/2$'], fontdict=dict(fontsize=12))
    ax_inset.tick_params(axis='x', labelsize=12)

    ax.set_ylabel('Libration Envelope [rad]')
    ax.set_xlabel('Time [s]')
    # ax_inset.set_xlabel('Time [s]', fontsize=12)

    title_str = '$\\hspace{3}k_d$ [s$^{-1}$],$\\hspace{1.5}\\hat{\\tau}$ [s]'
    legend = ax.legend(loc='lower right', ncol=1, fontsize=10, \
                       title_fontsize=12, framealpha=legend_alpha, \
                       title=title_str)
    legend.set_zorder(99)
    plt.setp(legend.texts, family='monospace')

    ax_inset.legend(loc='lower right', fontsize=10, \
                    title_fontsize=12, framealpha=1)

    ax.grid(which='major', axis='both')
    ax_inset.grid(which='major', axis='both')

    # axarr[0].grid(axis='y', which='major', lw=.75, color='k', ls='-', alpha=0.2)
    # axarr[0].grid(axis='y', which='minor', lw=.75, color='k', ls='--', alpha=0.2)
    # axarr[1].grid(axis='both', which='major', lw=1, color='k', ls='-', alpha=0.2)
    # axarr[1].grid(axis='both', which='minor', lw=1, color='k', ls='--', alpha=0.2)

    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, plot_name))

    if show:
        plt.show()