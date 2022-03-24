import os, sys, re

import numpy as np
import dill as pickle

import scipy.constants as constants

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

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
plot_name = 'fig7_libration_spectra_summary_v2.svg'

beadtype = 'bangs5'

# colors = bu.get_color_map(3, cmap=my_cmap)
# markers = ['o', 'o', 'o']

voltage_cmap = 'plasma'
markers = ['o', 's', 'x']
markersize = 6
# markersize = 35
markeralpha = 0.5

colorpad = 0.15
my_cmap = bu.truncate_colormap('plasma', vmin=0, vmax=0.9)


min_amp = np.inf
max_amp = -1.0*np.inf

min_phi_dg = np.inf

save = True
show = True

max_fit_kd = 1400


delta_t = 1.0 / 20000.0
nonlinear_fac = 1.1

Troom = 300.0


# fig, ax = plt.subplots(1,figsize=(7.5,4.5))
# zero_fig, zero_ax = plt.subplots(1,figsize=(7.5,4.5))
# ax_list = [zero_ax, ax]


fig, ax_list = plt.subplots(2,1,figsize=(6.5,5.0),sharex=True, \
                          gridspec_kw={'height_ratios': [8,3]})




nfiles = len(spectra_fit_filenames)

nzeros = 0
zeros_list = [0.0] * nfiles

for fileind, filename in enumerate(spectra_fit_filenames):
    spectra_file = os.path.join(processed_base, spectra_fit_filenames[fileind])
    spectra_dict = pickle.load( open(spectra_file, 'rb') )

    phi_dgs = list(spectra_dict.keys())
    phi_dgs.sort(key=float)

    found_zero = False
    for phi_dg in phi_dgs:

        if phi_dg == 0.0 and not found_zero:
            found_zero = True
            zeros_list[fileind] = 1.0

        if phi_dg > 0.0 and phi_dg < min_phi_dg:
            min_phi_dg = phi_dg

        drive_amps = spectra_dict[phi_dg]['drive_amp']
        cmin_amp = np.min(drive_amps)
        cmax_amp = np.max(drive_amps)

        if cmin_amp < min_amp:
            min_amp = cmin_amp

        if cmax_amp > max_amp:
            max_amp = cmax_amp


min_amp = 1e-3*min_amp
max_amp = 1e-3*max_amp

span = max_amp - min_amp
vmin = min_amp - colorpad*span
vmax = max_amp + colorpad*span

nzeros = np.sum(zeros_list)
zeroplot_ind = 0

paths = []
drive_amps = []
drive_amps_rounded = []
phi_dgs = []
fits = []
fit_uncs = []
markercolors = []
for i in range(nfiles):
    paths.append([[], []])
    drive_amps.append([[], []])
    drive_amps_rounded.append([[], []])
    phi_dgs.append([[], []])
    fits.append([[], []])
    fit_uncs.append([[], []])
    markercolors.append([[], []])


for fileind, filename in enumerate(spectra_fit_filenames):
    marker = markers[fileind]

    spectra_file = os.path.join(processed_base, spectra_fit_filenames[fileind])
    spectra_dict = pickle.load( open(spectra_file, 'rb') )

    phi_dg_keys = list(spectra_dict.keys())
    phi_dg_keys.sort(key=float)

    found_zero = False
    for phi_dg in phi_dg_keys:

        c_drive_amps = spectra_dict[phi_dg]['drive_amp']
        n_drive_amps = len(c_drive_amps)

        for drive_ind, drive_amp in enumerate(c_drive_amps):

            path = spectra_dict[phi_dg]['paths'][drive_ind]
            found_bad = False
            for bad_path in bad_paths:
                if bad_path in path:
                    found_bad = True
            if found_bad:
                continue

            drive_amp *= 1e-3
            color = bu.get_single_color(drive_amp, vmin=vmin, vmax=vmax, \
                                        cmap=my_cmap)

            if phi_dg != 0.0:
                ind = 1
                c_phi_dg = phi_dg
            else:
                if not found_zero:
                    found_zero = True
                ind = 0
                c_phi_dg = zeroplot_ind - (nzeros-1.0)/2.0 

            # lib_freq = ringdown_dict[phi_dg]['lib_freq'][drive_ind]

            spectra_fit = spectra_dict[phi_dg]\
                                    ['avg_shifted_squash_fit'][drive_ind]
            spectra_fit_unc = spectra_dict[phi_dg]\
                                    ['avg_shifted_squash_fit_unc'][drive_ind]

            if not len(spectra_fit):
                continue

            if phi_dg != 0.0:
                ind = 1
            else:
                ind = 0

            paths[fileind][ind].append(path)
            drive_amps[fileind][ind].append(drive_amp)
            drive_amps_rounded[fileind][ind].append(round(drive_amp, 0))
            phi_dgs[fileind][ind].append(phi_dg)
            fits[fileind][ind].append(spectra_fit)
            fit_uncs[fileind][ind].append(spectra_fit_unc)
            markercolors[fileind][ind].append(color)

    if found_zero:
        zeroplot_ind += 1

curves_to_fit = {}

min_val = np.inf
for fileind, filename in enumerate(filenames):
    date = re.search(r"\d{8,}", filename)[0]
    marker = markers[fileind]
    rhobead = bu.rhobead[beadtype]
    Ibead = bu.get_Ibead(date=date, rhobead=rhobead)

    curves_to_fit[date] = []
    feedback_amps = np.unique(drive_amps_rounded[fileind][1])
    n_feedback_amps = len(feedback_amps)
    for i in range(len(feedback_amps)):
        curves_to_fit[date].append([[], [], [], [], [], [], []])

    for ind in [0,1]:
        unique_phi_dgs = np.unique(phi_dgs[fileind][ind])
        unique_drive_amps = np.unique(drive_amps_rounded[fileind][ind])
        n_unique_amp = len(unique_drive_amps)
        if not n_unique_amp:
            continue

        for unique_phi_dg in unique_phi_dgs:

            for unique_ind, unique_drive_amp in enumerate(unique_drive_amps):

                vals = []
                uncs = []
                drive_vals = []
                freq_vals = []
                Tvals = []
                gamma_vals = []
                for meas_ind, phi_dg in enumerate(phi_dgs[fileind][ind]):
                    if phi_dg != unique_phi_dg:
                        continue
                    if unique_drive_amp != \
                            drive_amps_rounded[fileind][ind][meas_ind]:
                        continue

                    drive_vals.append(drive_amps[fileind][ind][meas_ind])

                    fit = fits[fileind][ind][meas_ind]
                    fit_unc = fit_uncs[fileind][ind][meas_ind]

                    # print(fit)

                    gamma0 = 2.0*np.pi*fit[2]
                    Sth = 4.0 * constants.k * Troom * Ibead['val'] * gamma0

                    if phi_dg:
                        gamma_d = 2.0*np.pi*fit[2] + fit[4]
                        Teff = ( Ibead['val'] * (2.0*np.pi*fit[1])**2 / constants.k ) \
                                * ( ( (Sth / Ibead['val']**2) \
                                    / ( 2.0 * (2.0*np.pi*fit[1])**2 * gamma_d ) ) \
                                  + ( (fit[4]**2 * fit[3])  / ( 2.0 * gamma_d ) ) )
                        Teff *= 0.5
                    else:
                        Teff = 300.0

                    Tvals.append(Teff)

                    vals.append(fit[4]) 
                    uncs.append(fit_unc[4])

                    freq_vals.append(fit[1])

                kd = (unique_phi_dg / 1024.0) * \
                        (5.0*np.pi*delta_t * (2.0*np.pi*np.mean(freq_vals))**2) 

                if (ind == 1) and (kd < min_val):
                    min_val = kd

                color = bu.get_single_color(unique_drive_amp, vmin=vmin, \
                                        vmax=vmax, cmap=my_cmap)
                xval = kd
                if not len(vals):
                    continue

                offset = 0.0
                if not ind:
                    offset = zero_panel_offsets[date]

                fac = 1.0
                try:
                    mean, unc = bu.weighted_mean(vals, uncs)
                except:
                    print(vals, uncs)
                    input()

                Tmean = np.mean(Tvals)
                Tunc = np.std(Tvals)

                freqmean = np.mean(freq_vals)
                frequnc = np.std(freq_vals)

                for feedback_amp_ind, feedback_amp in enumerate(feedback_amps):
                    if feedback_amp == unique_drive_amp:
                        curves_to_fit[date][feedback_amp_ind][0]\
                                .append(kd)
                        curves_to_fit[date][feedback_amp_ind][1]\
                                .append(np.mean(mean))
                        curves_to_fit[date][feedback_amp_ind][2]\
                                .append(np.std(unc))
                        curves_to_fit[date][feedback_amp_ind][3]\
                                .append(np.mean(Tmean))
                        curves_to_fit[date][feedback_amp_ind][4]\
                                .append(np.std(Tunc))
                        curves_to_fit[date][feedback_amp_ind][5]\
                                .append(np.mean(drive_vals))
                        curves_to_fit[date][feedback_amp_ind][6]\
                                .append(np.std(drive_vals))


                if ind == 0:
                    continue

                if Tunc > 0.5 * Tmean:
                    Tunc = 0.5 * Tmean

                ax_list[0].errorbar([xval], [Tmean], \
                                    yerr=[Tunc], \
                                    color=color, ecolor=color, \
                                    ls='None', zorder=4)
                ax_list[0].plot([xval], [Tmean], \
                                markeredgecolor=color, \
                                markerfacecolor='none', ms=markersize,\
                                marker=marker, markeredgewidth=1.5,\
                                zorder=5)


                ratio = mean / xval
                ratio_unc = ratio * np.sqrt((unc / mean)**2 + \
                                            4.0*(frequnc / freqmean)**2)
 
                ax_list[1].errorbar([xval], [ratio], \
                                    yerr=[ratio_unc], \
                                    color=color, ecolor=color, \
                                    ls='None', zorder=4)
                ax_list[1].plot([xval], [ratio], \
                                markeredgecolor=color, \
                                markerfacecolor='none', ms=markersize,\
                                marker=marker, markeredgewidth=1.5,\
                                zorder=5)



# for date in fits.keys():
#     n_feedback_amp = len(fits[date])

#     for i in range(n_feedback_amp):
#         data = np.array(fits[date][i])
#         data = data[:,np.argsort(data[0])]

#         inds = data[0] < max_fit_kd
#         zero_inds = data[2][inds] == 0.0

#         offset = 0.1*data[1][inds] * zero_inds

#         def fit_func(kd, gamma0, C):
#             return 2.0 / (gamma0 + C*kd)

#         ndof = np.sum(inds) - 2
#         def cost(gamma0, C):
#             resid = (data[1][inds] - fit_func(data[0][inds], gamma0, C))**2
#             variance = data[2][inds]**2 + offset**2
#             return (1.0 / ndof) * np.sum(resid / variance)

#         gamma_guess = 2.0/np.max(data[1][inds])
#         C_guess = 2.0/(np.min(data[1][inds])*np.max(data[0][inds]))
#         # C_guess = 1.0
#         m = Minuit(cost, gamma0=gamma_guess, C=C_guess ) 

#         ### Apply some limits to help keep the fitting well behaved
#         m.limits['gamma0'] = (0, np.inf )
#         # m.limits['C'] = (, )

#         # m.values["C"] =  1.0
#         # m.fixed["C"] = True
#         # print(m.params)

#         m.errordef = 1
#         m.print_level = 0

#         ### Do the actual minimization
#         m.migrad(ncall=5000000)
#         m.minos()
#         C_unc = np.mean(np.abs([m.merrors['C'].lower, m.merrors['C'].upper]))

#         popt = np.array(m.values)
#         pcov = np.array(m.covariance)

#         mean_drive = np.mean(data[3])

#         color = bu.get_single_color(mean_drive, vmin=vmin, \
#                                     vmax=vmax, cmap=my_cmap)

#         print()
#         print(date, mean_drive)
#         print('kd scaling param', popt[1], C_unc)
#         print()

#         plot_x = np.logspace(np.log10(min_val/2.0), np.log10(data[0,-1]), 1000)
#         ax_list[1].plot(plot_x, fit_func(plot_x, *popt), color=color, \
#                         lw=2, ls='--', alpha=0.5)
#         # ax_list[0].axhline(2.0/popt[0], color=color, lw=2, ls='--', alpha=0.5)



# ax_list[0].set_xlabel('Drive Amplitude [kV/m]')
ax_list[1].set_xlabel('Derivative gain, $k_d$ [s$^{-1}$]')

xlim = ax_list[0].get_xlim()
full_kd_vec = np.logspace(np.log10(min_val/2), np.log10(xlim[1]), 10)
# ax_list[1].plot(full_kd_vec, 2.0/full_kd_vec, ls='--', lw=2, \
#                 alpha=0.5, color='k', zorder=1)

ax_list[0].set_xlim(min_val/2, xlim[1])

ylabel = '$T_{\\rm eff}$ [K]'
ylabel_ratio = '$\\hat{k}_d \\,  / \\, k_d$'

ax_list[0].set_ylabel(ylabel)
ax_list[1].set_ylabel(ylabel_ratio)
for ind in [0,1]:
    ax_list[ind].set_xscale('log')
ax_list[0].set_yscale('log')

ax_list[1].set_yticks([0.0, 2.0])
ax_list[1].set_yticklabels(['0', '2'])
ax_list[1].set_ylim(-0.5, 2.5)
ax_list[1].axhline(1.0, color='k', ls='--', lw=2, zorder=1)

ax_list[0].grid(axis='both', which='major', lw=1, color='k', ls='-', alpha=0.2)
# ax_list[0].grid(axis='y', which='minor', lw=.75, color='k', ls='--', alpha=0.2)
ax_list[1].grid(axis='both', which='major', lw=1, color='k', ls='-', alpha=0.2)
# ax_list[1].grid(axis='both', which='minor', lw=1, color='k', ls='--', alpha=0.2)

fig.tight_layout()

norm = colors.Normalize(vmin=vmin, vmax=vmax)

cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=my_cmap), \
                  ax=ax_list, pad=0.005, fraction=0.125, aspect=30)
cb.set_label('Drive voltage [kV/m]', labelpad=7, fontsize=14)


# fig.subplots_adjust(right=0.8)


if save:
    fig.savefig( os.path.join(save_dir, plot_name) )

if show:
    plt.show()