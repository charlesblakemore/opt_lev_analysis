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

import scipy.constants as constants

import bead_util as bu

plt.rcParams.update({'font.size': 14})



min_markersize = 2
max_markersize = 10


processed_base = '/data/old_trap_processed/spinning/'

filenames = [ \
             '20200727/20200727_libration_spectra_6kd.p', \
             # '20200924/20200924_libration_spectra_6kd.p', \
             # '20201030/20201030_libration_spectra_6kd.p', \
            ]

bad_paths = [ \
             '20200727/bead1/spinning/dds_phase_impulse_high_dg/trial_0008.h5', \
            ]

save_dir = '/home/cblakemore/plots/libration_paper_2022'
plot_name = 'fig6_libration_spectra_examples_v2.svg'

save = True
show = True

plot_psd = True

scale_spectra = True
scaling_fac = 0.1


delta_t = 1.0 / 20000.0
kd_to_fit = 6.0

Troom = 300.0

beadtype = 'bangs5'

# line_alpha = 0.9
# data_alpha = 0.7
# lightening_factor = 0.75

line_alpha = 0.85
data_alpha = 1.0
lightening_factor = 0.525

colorpad = 0.1


# amp_to_plot = 9.0
# amp_to_plot = 27.0
amp_to_plot = 71.0  # in kV/m
trial_select = 0

impulse_magnitude = np.pi/2

all_plot_upper_lim = 100.0
efoldings_to_plot = 3.0

# example_dg = 10
# example_dgs = [1.0e-3, 4.0]
out_cut = 100


n_dg = 4
dg_lims = [1e-2, 15]

center_on_zero = False
plot_xscale = 'log'

# plot_xlim = [1000, 2000]
# plot_xlim = [1250, 1350]
plot_xlim = [1260, 1340]
# plot_xlim = [-25.0, 75.0]
# plot_xlim = [-5.0, 10.0]

# xticks = []
xticks = [1275, 1300, 1325]
# xticks = [1250, 1275, 1300, 1325, 1350]
# xticks = [-25.0, 0.0, 25.0, 50.0, 75.0]
# xticks = [-5.0, 0.0, 5.0, 10.0]
if len(xticks):
    xticklabels = [f'{int(xtick):d}' for xtick in xticks]
else:
    xticklabels = []

# plot_xlim = [-1, 80]
plot_ylim = [3e-4, 1e-1]
plot_ylim = [9e-8, 1e-2]  # For PSD

plot_ylim = [1e-11, 1e-1]

# inset_bbox = (0.6, 0.6, 0.35, 0.35)
inset_bbox = (0.055, 0.11, 0.35, 0.4)



fit_func = lambda f,A,f0,g,noise,kd,tfb,c: \
                np.sqrt( bu.damped_osc_amp_squash(f,A,f0,g,\
                                                  noise,kd,tfb)**2 + c )


nbin_per_file = 1000



fig, ax = plt.subplots(figsize=(6.5,4.0))


nfiles = len(filenames)

tmin = np.inf
tmax = -np.inf

for fileind, filename in enumerate(filenames):
    spectra_file = os.path.join(processed_base, filename)
    spectra_dict = pickle.load( open(spectra_file, 'rb') )

    date = re.search(r"\d{8,}", spectra_file)[0]
    rhobead = bu.rhobead[beadtype]
    Ibead = bu.get_Ibead(date=date, rhobead=rhobead)

    phi_dgs = list(spectra_dict.keys())
    phi_dgs.sort(key=float)

    phi_dgs = np.array(phi_dgs)
    inds = (phi_dgs > dg_lims[0]) * (phi_dgs < dg_lims[1])

    phi_dgs = phi_dgs[inds]

    phi_dg_test_vec = np.logspace(np.log10(phi_dgs[0]), np.log10(phi_dgs[-1]), n_dg)

    phi_dg_to_plot = [0.0]
    for phi_dg in phi_dg_test_vec:
        phi_dg_to_plot.append(phi_dgs[np.argmin(np.abs(phi_dgs - phi_dg))])

    color_vmin = 0.1 * phi_dg_to_plot[1]
    color_vmax = 20.0 * phi_dg_to_plot[-1]

    for phi_dg_ind, phi_dg in enumerate(phi_dg_to_plot):
        if phi_dg:
            color = bu.get_single_color(phi_dg, log=True, \
                                        vmin=color_vmin, \
                                        vmax=color_vmax)
        else:
            color = 'k'

        my_dict = spectra_dict[phi_dg]
        amps_rounded = np.around(np.array(my_dict['drive_amp'])*1e-3)
        good_inds = np.abs(amps_rounded - amp_to_plot) < 5.0

        actual_inds = np.arange(len(my_dict['paths']))[good_inds]
        actual_amp = np.mean(np.array(my_dict['drive_amp'])[good_inds])

        basic_fit = np.array(my_dict['avg_shifted_fit'], \
                       dtype=object)[good_inds][trial_select]
        fit = np.array(my_dict['avg_shifted_squash_fit'], \
                       dtype=object)[good_inds][trial_select]
        freqs = np.array(my_dict['freqs'], \
                         dtype=object)[good_inds][trial_select]
        asd = np.array(my_dict['asd'], \
                       dtype=object)[good_inds][trial_select]

        kd = ( phi_dg / 1024.0) * \
                (5.0*np.pi*delta_t * (2.0*np.pi*fit[1])**2)
        half_width = np.min([np.max([0.5*kd_to_fit*kd / (2.0*np.pi), 25]), 500])

        fit_freqs = np.linspace(fit[1]-half_width, fit[1]+half_width, 500)
        fit_asd = fit_func(fit_freqs, *fit)

        plot_inds = (freqs > plot_xlim[0]) * (freqs < plot_xlim[1])
        plot_inds = (freqs == freqs)

        if not phi_dg:
            freqs -= 20.0
            fit_freqs -= 20.0
            gamma0 = 2.0*np.pi*basic_fit[2]
        else:
            gamma0 = 2.0*np.pi*fit[2]

        Sth = 4.0 * constants.k * Troom * Ibead['val'] * gamma0
        # Sth = fit[0] * Ibead['val']**2

        if phi_dg:
            gamma_d = 2.0*np.pi*fit[2] + fit[4]
            Teff = ( Ibead['val'] * (2.0*np.pi*fit[1])**2 / constants.k ) \
                    * ( ( (Sth / Ibead['val']**2) \
                        / ( 2.0 * (2.0*np.pi*fit[1])**2 * gamma_d ) ) \
                      + ( (fit[4]**2 * fit[3])  / ( 2.0 * gamma_d ) ) )

            ### Adjust by 1/2 to account for single vs double-sided PSDs.
            ### Closed-form results from integral are derived from a 
            ### supplementary material using double-sided, whereas our 
            ### definitions and formalism are usually for the single-sided
            Teff *= 0.5
            # label = f'{kd:0.3g},  {gamma_d:0.3g}, {Teff:0.3g}'
            label = bu.format_multiple_float_string( \
                        kd, gamma_d, Teff, sig_figs=3, extra=2)

        else:
            gamma_d = 2.0*np.pi*basic_fit[2]
            Teff = (Ibead['val'] * (2.0*np.pi*fit[1])**2 / constants.k) \
                    * ( (Sth / Ibead['val']**2) \
                        / (2.0 * (2.0*np.pi*basic_fit[1])**2 * gamma_d ) )

            Teff *= 0.5
            # label = f'0.0, {gamma_d:0.3g},  {Teff:0.3g}'
            label = bu.format_multiple_float_string( \
                        0.0, gamma_d, Teff, sig_figs=3, extra=2)

        if plot_psd:
            asd = asd**2
            fit_asd = fit_asd**2

        if scale_spectra:
            asd *= scaling_fac**phi_dg_ind
            fit_asd *= scaling_fac**phi_dg_ind

        if center_on_zero:
            ax.semilogy(freqs[plot_inds]-fit[1], asd[plot_inds], \
                        color=bu.lighten_color(color, lightening_factor), \
                        zorder=4+int(2.0*phi_dg_ind), \
                        alpha=data_alpha, lw=2)
            ax.semilogy(fit_freqs-fit[1], fit_asd, color=color, lw=3, ls='--', \
                        zorder=5+int(2.0*phi_dg_ind), label=label, \
                        alpha=line_alpha)
        else:
            ax.loglog(freqs[plot_inds], asd[plot_inds], \
                      color=bu.lighten_color(color, lightening_factor), \
                      zorder=4+int(2.0*phi_dg_ind), \
                      alpha=data_alpha, lw=2)
            ax.loglog(fit_freqs, fit_asd, color=color, lw=3, ls='--', \
                      zorder=5+int(2.0*phi_dg_ind), label=label, \
                      alpha=line_alpha)

    if len(plot_xlim):
        ax.set_xlim(*plot_xlim)

    if len(plot_ylim):
        ax.set_ylim(*plot_ylim)

    if len(xticks):
        ax.set_xticks(xticks)
    if len(xticklabels):
        ax.set_xticklabels(xticklabels)

    ax.set_xticks([], minor=True)
    ax.set_xticklabels([], minor=True)

    if scale_spectra:
        ax.set_ylabel('Libration PSD [arb]')
    elif plot_psd:
        ax.set_ylabel('Libration PSD [rad$^2$/Hz]')
    else:
        ax.set_ylabel('Libration ASD [rad/$\\sqrt{\\rm Hz}$]')

    if center_on_zero:
        ax.set_xlabel('$(\\omega - \\omega_{\\phi})/2\\pi$ [Hz]')
    else:
        ax.set_xlabel('Frequency [Hz]')

    title_str = '$\\hspace{3.25} k_d$ [s$^{-1}$],$\\hspace{1}$' \
                 + '$\\hat{\\gamma}_d$ [s$^{-1}$],$\\hspace{1.25}$' \
                 + '$T_{\\rm eff}$ [K]'
    legend = ax.legend(loc='upper right', ncol=1, fontsize=10, \
                       framealpha=1, title=title_str, \
                       title_fontsize=12)
    legend.set_zorder(99)
    plt.setp(legend.texts, family='monospace')

    ax.grid(which='major', axis='y')

    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, plot_name))

    if show:
        plt.show()