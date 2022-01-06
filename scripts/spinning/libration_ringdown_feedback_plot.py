import os, sys, re

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
             '20200924/20200924_libration_ringdowns_2efoldings.p', \
             '20201030/20201030_libration_ringdowns_2efoldings.p', \
            ]

bad_paths = [ \
             '20200727/bead1/spinning/dds_phase_impulse_high_dg/trial_0008.h5', \
            ]

beadtype = 'bangs5'

# colors = bu.get_color_map(3, cmap='plasma')
# markers = ['o', 'o', 'o']

voltage_cmap = 'plasma'
markers = ['x', 'o', '+']
markers = ['s', 'o', '*']
markers = ['<', '*', '>']
markersize = 35
markeralpha = 0.5

colorpad = 0.1


min_amp = np.inf
max_amp = -1.0*np.inf

min_phi_dg = np.inf

normalize = False




fig, axarr = plt.subplots(1,2,figsize=(7.5,4.5),sharey=True, \
                          gridspec_kw={'width_ratios': [1,8]})

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


nzeros = np.sum(zeros_list)
zeroplot_ind = 0

paths = []
drive_amps = []
phi_dgs = []
taus = []
tau_uncs = []
markercolors = []
for i in range(nfiles):
    paths.append([[], []])
    drive_amps.append([[], []])
    phi_dgs.append([[], []])
    taus.append([[], []])
    tau_uncs.append([[], []])
    markercolors.append([[], []])

for fileind, filename in enumerate(filenames):
    marker = markers[fileind]

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
                                        cmap='plasma')

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

            paths[fileind][ind].append(path)
            drive_amps[fileind][ind].append(round(drive_amp, 0))
            phi_dgs[fileind][ind].append(c_phi_dg*fac)
            taus[fileind][ind].append(tau)
            tau_uncs[fileind][ind].append(tau_unc)
            markercolors[fileind][ind].append(color)


    if found_zero:
        zeroplot_ind += 1

for fileind, filename in enumerate(filenames):
    date = re.search(r"\d{8,}", filename)[0]
    marker = markers[fileind]
    rhobead = bu.rhobead[beadtype]
    Ibead = bu.get_Ibead(date=date, rhobead=rhobead)['val']

    for ind in [0,1]:
        unique_phi_dgs = np.unique(phi_dgs[fileind][ind])
        unique_drive_amps = np.unique(drive_amps[fileind][ind])
        n_unique_amp = len(unique_drive_amps)
        if not n_unique_amp:
            continue

        for unique_phi_dg in unique_phi_dgs:

            for unique_ind, unique_drive_amp in enumerate(unique_drive_amps):
                color = bu.get_single_color(unique_drive_amp, vmin=vmin, \
                                            vmax=vmax, cmap='plasma')
                vals = []
                for meas_ind, phi_dg in enumerate(phi_dgs[fileind][ind]):
                    if phi_dg != unique_phi_dg:
                        continue
                    if unique_drive_amp != drive_amps[fileind][ind][meas_ind]:
                        continue
                    vals.append(taus[fileind][ind][meas_ind]) 

                if not len(vals):
                    continue

                offset = 0.0
                if not ind:
                    offset = 0.05*(unique_ind - (n_unique_amp-1.0)/2)

                if normalize:
                    fac = (Ibead * unique_drive_amp)
                else:
                    fac = 1.0

                axarr[ind].errorbar([unique_phi_dg+offset], [fac*np.mean(vals)], \
                                    yerr=[fac*np.std(vals)], ecolor=color, \
                                    ls='None', zorder=4)
                axarr[ind].scatter([unique_phi_dg+offset], [fac*np.mean(vals)], \
                                   color=color, s=markersize, marker=marker, \
                                   zorder=5)
    if found_zero:
        zeroplot_ind += 1


        ### Plot all the measurements with an alpha
        # axarr[ind].errorbar(phi_dgs[fileind][ind], taus[fileind][ind], \
        #             yerr=tau_uncs[fileind][ind], \
        #             ecolor=markercolors[fileind][ind], ls='None', \
        #             alpha=markeralpha, zorder=2)
        # axarr[ind].scatter(phi_dgs[fileind][ind], taus[fileind][ind], \
        #             color=markercolors[fileind][ind], alpha=markeralpha, \
        #             s=markersize*0.75, marker=marker, zorder=3)


axarr[1].set_xlabel('Derivative gain [arb]')

if normalize:
    axarr[0].set_ylabel('$\\tau \\, E_0 \\, d_{ms}$ [arb]')
else:
    axarr[0].set_ylabel('Libration damping time [s]')
axarr[0].set_yscale('log')

axarr[0].set_xscale('linear')
axarr[1].set_xscale('log')

axarr[0].set_xticks([0.0])
axarr[0].set_xticklabels(['0'])
zero_xlim = axarr[0].get_xlim()
zero_half_span = 0.5*(zero_xlim[1] - zero_xlim[0])
axarr[0].set_xlim(-2.0*zero_half_span, 2.0*zero_half_span)

axarr[0].grid(axis='y', which='major', lw=.75, color='k', ls='-', alpha=0.2)
# axarr[0].grid(axis='y', which='minor', lw=.75, color='k', ls='--', alpha=0.2)
axarr[1].grid(axis='both', which='major', lw=1, color='k', ls='-', alpha=0.2)
# axarr[1].grid(axis='both', which='minor', lw=1, color='k', ls='--', alpha=0.2)

# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# ax.set_xscale('log')
# axarr[0].set_xscale('symlog', linthresh=linthresh)
# ax.xaxis.set_minor_locator(bu.MinorSymLogLocator(linthresh))
# ax.set_xlim(-0.1*linthresh, xlim[1])]
# ax.set_ylim(*ylim)

# ax.axvline(linthresh, zorder=1, ls='--', lw=2, color='k', alpha=0.6)

# ax.legend(loc='upper right', markerscale=1.0, scatterpoints=1)

ax_cb, cb = bu.add_colorbar(fig, axarr[1], size=0.1, pad=0.025, vmin=vmin, \
                          vmax=vmax, label='Drive voltage [kV/m]', labelpad=7, fontsize=14)
# fig.add_axes(ax_cb)

fig.tight_layout()
plt.subplots_adjust(wspace=0.05)

plt.show()