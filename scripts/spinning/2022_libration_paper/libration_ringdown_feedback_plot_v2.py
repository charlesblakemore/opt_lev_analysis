import os, sys, re

import numpy as np
import dill as pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import cm

from iminuit import Minuit, describe

import bead_util as bu

plt.rcParams.update({'font.size': 14})



min_markersize = 2
max_markersize = 10


processed_base = '/data/old_trap_processed/spinning/'

filenames = [ \
             '20200727/20200727_libration_ringdowns_2efoldings.p', \
             '20200924/20200924_libration_ringdowns_2efoldings.p', \
             '20201030/20201030_libration_ringdowns_2efoldings.p', \
            ]

spectra_fit_filenames \
        = [ \
           '20200727/20200727_libration_spectra_6kd.p', \
           '20200924/20200924_libration_spectra_6kd.p', \
           '20201030/20201030_libration_spectra_6kd.p', \
          ]

bad_paths = [ \
             '20200727/bead1/spinning/dds_phase_impulse_high_dg/trial_0008.h5', \
             '20200924/bead1/spinning/dds_phase_impulse_1Vpp_high_dg', \
            ]

zero_panel_offsets = {'20200727': -1.0, '20200924': 0.0, '20201030': 1.0}

save_dir = '/home/cblakemore/plots/libration_paper_2022'
plot_name = 'fig4_libration_ringdown_summary_normalized.svg'
zero_plot_name = 'libration_ringdown_zero_feedback.svg'

beadtype = 'bangs5'

# colors = bu.get_color_map(3, cmap='plasma')
# markers = ['o', 'o', 'o']

voltage_cmap = 'plasma'
markers = ['o', 's', 'x']
# markers = ['o', '*', '+']
# markers = ['s', 'o', '*']
# markers = ['<', '*', '>']
markersize = 6
# markersize = 35
markeralpha = 0.5

colorpad = 0.15
# colorpad = 0.1
my_cmap = bu.truncate_colormap('plasma', vmin=0, vmax=0.9)

line_alpha = 0.75

min_amp = np.inf
max_amp = -1.0*np.inf

min_phi_dg = np.inf

normalize = False

save = False
show = True


max_fit_dg = 29

max_fit_kd = 1400


delta_t = 1.0 / 20000.0
nonlinear_fac = 1.1



# fig, ax = plt.subplots(1,figsize=(7.5,4.5))
# zero_fig, zero_ax = plt.subplots(1,figsize=(7.5,4.5))
# ax_list = [zero_ax, ax]


fig, ax_list = plt.subplots(1,2,figsize=(6.5,4.0),sharey=True, \
                          gridspec_kw={'width_ratios': [1,8]})

nfiles = len(filenames)

nzeros = 0
zeros_list = [0.0] * nfiles

nfiles = len(filenames)

nzeros = 0
zeros_list = [0.0] * nfiles

for fileind, filename in enumerate(filenames):
    ringdown_file = os.path.join(processed_base, filename)
    ringdown_dict = pickle.load( open(ringdown_file, 'rb') )

    phi_dgs = list(ringdown_dict.keys())
    phi_dgs.sort(key=float)

    found_zero = False
    for phi_dg in phi_dgs:

        if phi_dg == 0.0 and not found_zero:
            found_zero = True
            zeros_list[fileind] = 1.0

        if phi_dg > 0.0 and phi_dg < min_phi_dg:
            min_phi_dg = phi_dg

        drive_amps = ringdown_dict[phi_dg]['drive_amp']
        cmin_amp = np.min(drive_amps)
        cmax_amp = np.max(drive_amps)

        if cmin_amp < min_amp:
            min_amp = cmin_amp

        if cmax_amp > max_amp:
            max_amp = cmax_amp

factor = (max_markersize - min_markersize) / (max_amp - min_amp)

min_amp = 1e-3*min_amp
max_amp = 1e-3*max_amp

span = max_amp - min_amp
vmin = min_amp - colorpad*span
vmax = max_amp + colorpad*span

# vmin = min_amp * (10.0 * min_amp / max_amp - 9)
# vmin = min_amp - ( (np.log10(10.0) * (max_amp - min_amp)) / np.log10(474/0.581) )
# vmax = max_amp + ( (np.log10(20.0) * (max_amp - min_amp)) / np.log10(474/0.581) )

# color_fac = vmax / max_amp

# print()
# print(color_fac)
# print()


nzeros = np.sum(zeros_list)
zeroplot_ind = 0

paths = []
drive_amps = []
drive_amps_rounded = []
phi_dgs = []
lib_freqs = []
taus = []
tau_uncs = []
markercolors = []
for i in range(nfiles):
    paths.append([[], []])
    drive_amps.append([[], []])
    drive_amps_rounded.append([[], []])
    phi_dgs.append([[], []])
    lib_freqs.append([[], []])
    taus.append([[], []])
    tau_uncs.append([[], []])
    markercolors.append([[], []])


for fileind, filename in enumerate(filenames):
    marker = markers[fileind]

    spectra_file = os.path.join(processed_base, spectra_fit_filenames[fileind])
    spectra_dict = pickle.load( open(spectra_file, 'rb') )

    ringdown_file = os.path.join(processed_base, filename)
    ringdown_dict = pickle.load( open(ringdown_file, 'rb') )
    fac = 1.0
    # if '20200727' in filename:
    #     fac = 2**2
    # else:
    #     fac = 1.0

    phi_dg_keys = list(ringdown_dict.keys())
    phi_dg_keys.sort(key=float)

    found_zero = False
    for phi_dg in phi_dg_keys:

        c_drive_amps = ringdown_dict[phi_dg]['drive_amp']
        n_drive_amps = len(c_drive_amps)

        for drive_ind, drive_amp in enumerate(c_drive_amps):

            path = ringdown_dict[phi_dg]['paths'][drive_ind]
            found_bad = False
            for bad_path in bad_paths:
                if bad_path in path:
                    found_bad = True
            if found_bad:
                continue

            drive_amp *= 1e-3
            color = bu.get_single_color(drive_amp, vmin=vmin, vmax=vmax, \
                                        cmap=my_cmap)

            # markersize = (drive_amp - min_amp)*factor + min_markersize
            tau = ringdown_dict[phi_dg]['fit'][drive_ind][2]
            tau_unc = ringdown_dict[phi_dg]['unc'][drive_ind][2]

            if phi_dg != 0.0:
                ind = 1
                c_phi_dg = phi_dg
            else:
                if not found_zero:
                    found_zero = True
                ind = 0
                c_phi_dg = zeroplot_ind - (nzeros-1.0)/2.0 

            # lib_freq = ringdown_dict[phi_dg]['lib_freq'][drive_ind]
            lib_freq = spectra_dict[phi_dg]['avg_shifted_squash_fit'][drive_ind][1]

            if phi_dg != 0.0:
                ind = 1
            else:
                ind = 0

            paths[fileind][ind].append(path)
            drive_amps[fileind][ind].append(drive_amp)
            drive_amps_rounded[fileind][ind].append(round(drive_amp, 0))
            phi_dgs[fileind][ind].append(phi_dg)
            lib_freqs[fileind][ind].append(lib_freq)
            taus[fileind][ind].append(tau)
            tau_uncs[fileind][ind].append(tau_unc)
            markercolors[fileind][ind].append(color)

    if found_zero:
        zeroplot_ind += 1

fits = {}

min_val = np.inf
for fileind, filename in enumerate(filenames):
    date = re.search(r"\d{8,}", filename)[0]
    marker = markers[fileind]
    rhobead = bu.rhobead[beadtype]
    Ibead = bu.get_Ibead(date=date, rhobead=rhobead)['val']

    fits[date] = []
    feedback_amps = np.unique(drive_amps_rounded[fileind][1])
    n_feedback_amps = len(feedback_amps)
    for i in range(len(feedback_amps)):
        fits[date].append([[], [], [], [], []])

    for ind in [0,1]:
        unique_phi_dgs = np.unique(phi_dgs[fileind][ind])
        unique_drive_amps = np.unique(drive_amps_rounded[fileind][ind])
        n_unique_amp = len(unique_drive_amps)
        if not n_unique_amp:
            continue

        for unique_phi_dg in unique_phi_dgs:

            for unique_ind, unique_drive_amp in enumerate(unique_drive_amps):

                vals = []
                drive_vals = []
                freq_vals = []
                for meas_ind, phi_dg in enumerate(phi_dgs[fileind][ind]):
                    if phi_dg != unique_phi_dg:
                        continue
                    if unique_drive_amp != \
                            drive_amps_rounded[fileind][ind][meas_ind]:
                        continue
                    vals.append(taus[fileind][ind][meas_ind]*nonlinear_fac) 
                    drive_vals.append(drive_amps[fileind][ind][meas_ind])
                    freq_vals.append(lib_freqs[fileind][ind][meas_ind])

                kd = (unique_phi_dg / 1024.0) * \
                        (5.0*np.pi*delta_t * (2.0*np.pi*np.mean(freq_vals))**2) 

                if (ind == 1) and (kd < min_val):
                    min_val = kd

                if ind == 0:
                    color = 'k'
                    xval = unique_drive_amp
                else:
                    color = bu.get_single_color(unique_drive_amp, vmin=vmin, \
                                            vmax=vmax, cmap=my_cmap)
                    xval = kd
                    # xval = unique_phi_dg

                color = bu.get_single_color(unique_drive_amp, vmin=vmin, \
                                        vmax=vmax, cmap=my_cmap)
                xval = kd


                if not len(vals):
                    continue

                offset = 0.0
                if not ind:
                    offset = zero_panel_offsets[date]

                if normalize:
                    fac = (Ibead * unique_drive_amp)
                else:
                    fac = 1.0

                for feedback_amp_ind, feedback_amp in enumerate(feedback_amps):
                    if feedback_amp == unique_drive_amp:
                        fits[date][feedback_amp_ind][0].append(kd)
                        fits[date][feedback_amp_ind][1].append(np.mean(vals))
                        fits[date][feedback_amp_ind][2].append(np.std(vals))
                        fits[date][feedback_amp_ind][3].append(np.mean(drive_vals))
                        fits[date][feedback_amp_ind][4].append(np.std(drive_vals))

                ax_list[ind].errorbar([xval+offset], [fac*np.mean(vals)], \
                                       yerr=[fac*np.std(vals)], \
                                       color=color, ecolor=color, \
                                       ls='None', zorder=4)
                ax_list[ind].plot([xval+offset], [fac*np.mean(vals)], \
                                   markeredgecolor=color, \
                                   markerfacecolor='none', ms=markersize,\
                                   marker=marker, markeredgewidth=1.5,\
                                   zorder=5)

        ### Plot all the measurements with an alpha
        # axarr[ind].errorbar(phi_dgs[fileind][ind], taus[fileind][ind], \
        #             yerr=tau_uncs[fileind][ind], \
        #             ecolor=markercolors[fileind][ind], ls='None', \
        #             alpha=markeralpha, zorder=2)
        # axarr[ind].scatter(phi_dgs[fileind][ind], taus[fileind][ind], \
        #             color=markercolors[fileind][ind], alpha=markeralpha, \
        #             s=markersize*0.75, marker=marker, zorder=3)


for date in fits.keys():
    n_feedback_amp = len(fits[date])

    for i in range(n_feedback_amp):
        data = np.array(fits[date][i])
        data = data[:,np.argsort(data[0])]

        inds = data[0] < max_fit_kd
        zero_inds = data[2][inds] == 0.0

        offset = 0.1*data[1][inds] * zero_inds

        def fit_func(kd, gamma0, C):
            return 2.0 / (gamma0 + C*kd)

        ndof = np.sum(inds) - 2
        def cost(gamma0, C):
            resid = (data[1][inds] - fit_func(data[0][inds], gamma0, C))**2
            variance = data[2][inds]**2 + offset**2
            return (1.0 / ndof) * np.sum(resid / variance)

        gamma_guess = 2.0/np.max(data[1][inds])
        C_guess = 2.0/(np.min(data[1][inds])*np.max(data[0][inds]))
        # C_guess = 1.0
        m = Minuit(cost, gamma0=gamma_guess, C=C_guess ) 

        ### Apply some limits to help keep the fitting well behaved
        m.limits['gamma0'] = (0, np.inf )
        # m.limits['C'] = (, )

        # m.values["C"] =  1.0
        # m.fixed["C"] = True
        # print(m.params)

        m.errordef = 1
        m.print_level = 0

        ### Do the actual minimization
        m.migrad(ncall=5000000)
        m.minos()
        C_unc = np.mean(np.abs([m.merrors['C'].lower, m.merrors['C'].upper]))

        popt = np.array(m.values)
        pcov = np.array(m.covariance)

        mean_drive = np.mean(data[3])

        color = bu.get_single_color(mean_drive, vmin=vmin, \
                                    vmax=vmax, cmap=my_cmap)

        print()
        print(date, mean_drive)
        print('kd scaling param', popt[1], C_unc)
        print()

        plot_x = np.logspace(np.log10(min_val/2.0), np.log10(data[0,-1]), 1000)
        ax_list[1].plot(plot_x, fit_func(plot_x, *popt), color=color, \
                        lw=2, ls='--', alpha=line_alpha)
        # ax_list[0].axhline(2.0/popt[0], color=color, lw=2, ls='--', alpha=0.5)



# ax_list[0].set_xlabel('Drive Amplitude [kV/m]')
ax_list[1].set_xlabel('Derivative gain, $k_d$ [s$^{-1}$]')

xlim = ax_list[1].get_xlim()
full_kd_vec = np.logspace(np.log10(min_val/2), np.log10(xlim[1]), 10)
# ax_list[1].plot(full_kd_vec, 2.0/full_kd_vec, ls='--', lw=2, \
#                 alpha=0.5, color='k', zorder=1)

ax_list[1].set_xlim(min_val/2, xlim[1])

if normalize:
    ylabel = '$\\tau \\, E_0 \\, d_{ms}$ [arb]'
else:
    ylabel = 'Libration damping time [s]'

ax_list[0].set_ylabel(ylabel)
for ind in [0,1]:
    ax_list[ind].set_yscale('log')
ax_list[1].set_xscale('log')

ax_list[0].set_xticks([0.0])
ax_list[0].set_xticklabels(['0'])
ax_list[0].xaxis.set_ticks_position('none')

zero_xlim = ax_list[0].get_xlim()
zero_half_span = 0.5*(zero_xlim[1] - zero_xlim[0])
ax_list[0].set_xlim(-2.0*zero_half_span, 2.0*zero_half_span)

ax_list[0].grid(axis='y', which='major', lw=.75, color='k', ls='-', alpha=0.2)
# ax_list[0].grid(axis='y', which='minor', lw=.75, color='k', ls='--', alpha=0.2)
ax_list[1].grid(axis='both', which='major', lw=1, color='k', ls='-', alpha=0.2)
# ax_list[1].grid(axis='both', which='minor', lw=1, color='k', ls='--', alpha=0.2)

ax_list[0].xaxis.set_minor_formatter(lambda x, pos: f'{int(x):d}')

# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# ax.set_xscale('log')
# ax_list[0].set_xscale('symlog', linthresh=linthresh)
# ax.xaxis.set_minor_locator(bu.MinorSymLogLocator(linthresh))
# ax.set_xlim(-0.1*linthresh, xlim[1])]
# ax.set_ylim(*ylim)

# ax.axvline(linthresh, zorder=1, ls='--', lw=2, color='k', alpha=0.6)

# ax.legend(loc='upper right', markerscale=1.0, scatterpoints=1)

ax_cb, cb = bu.add_colorbar(fig, ax_list[1], size=0.1, pad=0.025, vmin=vmin, \
                            vmax=vmax, cmap=my_cmap, label='Drive voltage [kV/m]', \
                            labelpad=7, fontsize=14)
# fig.add_axes(ax_cb)

fig.tight_layout()
# zero_fig.tight_layout()

if save:
    fig.savefig( os.path.join(save_dir, plot_name) )
    # zero_fig.savefig( os.path.join(save_dir, zero_plot_name) )

if show:
    plt.show()