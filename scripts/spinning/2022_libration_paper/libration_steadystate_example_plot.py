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

spectra_fit_filenames = \
            [ \
             # '20200727/20200727_libration_spectra_6kd.p', \
             # '20200924/20200924_libration_spectra_6kd.p', \
             # '20201030/20201030_libration_spectra_6kd.p', \
             # '20200727/20200727_libration_spectra_6kd_v2.p', \
             # '20200924/20200924_libration_spectra_6kd_v2.p', \
             # '20201030/20201030_libration_spectra_6kd_v2.p', \
             '20200727/20200727_libration_spectra_6kd_v3.p', \
             # '20200924/20200924_libration_spectra_6kd_v3.p', \
             # '20201030/20201030_libration_spectra_6kd_v3.p', \
            ]

bad_paths = [ \
             '20200727/bead1/spinning/dds_phase_impulse_high_dg/trial_0008.h5', \
            ]

save_dir = '/home/cblakemore/plots/libration_paper_2022'
plot_name = 'fig6_libration_spectra_examples_v3.svg'

save = True
show = True

plot_psd = True

scale_spectra = True
scaling_fac = 0.1


delta_t = 1.0 / 20000.0
kd_to_fit = 6.0


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
trial_select = 1

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



### WHY OH WHY DID I IMPLEMENT DIFFERENT DEFINTIONS OF THE CONSTANT "c"
### WHERE SOMETIMES IT'S SQUARED AND OTHER TIMES NOT
fit_func = lambda f,A,f0,g,noise,kd,tfb,c: \
                np.sqrt( bu.damped_osc_amp_squash(f,A,f0,g,\
                                                  noise,kd,tfb)**2 + c )

basic_fit_func = lambda f,A,f0,g,c: \
                np.sqrt( bu.damped_osc_amp(f,A,f0,g)**2 + c**2 )

nfreq = 500





nfiles = len(spectra_fit_filenames)

tmin = np.inf
tmax = -np.inf




gamma0_dict = {}

for fileind, filename in enumerate(spectra_fit_filenames):
    date = re.search(r"\d{8,}", filename)[0]
    rhobead = bu.rhobead[beadtype]
    Ibead = bu.get_Ibead(date=date, rhobead=rhobead)

    spectra_file = os.path.join(processed_base, spectra_fit_filenames[fileind])
    spectra_dict = pickle.load( open(spectra_file, 'rb') )

    phi_dg_keys = list(spectra_dict.keys())
    phi_dg_keys.sort(key=float)

    gamma0_arr = [[], [], [], []]
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

            if not phi_dg:

                basic_fit = spectra_dict[phi_dg]\
                                        ['avg_shifted_fit'][drive_ind]
                basic_fit_unc = spectra_dict[phi_dg]\
                                        ['avg_shifted_fit_unc'][drive_ind]
                spectra_fit = spectra_dict[phi_dg]\
                                        ['avg_shifted_squash_fit'][drive_ind]
                spectra_fit_unc = spectra_dict[phi_dg]\
                                        ['avg_shifted_squash_fit_unc'][drive_ind]

                freqs = spectra_dict[phi_dg]['freqs'][drive_ind]
                asd = spectra_dict[phi_dg]['asd'][drive_ind]

                gamma0_arr[0].append(round(drive_amp, 0))
                gamma0_arr[1].append(2.0*np.pi*basic_fit[2])
                gamma0_arr[2].append(2.0*np.pi*basic_fit_unc[2])

                f_lib = basic_fit[1]
                omega_lib = 2.0 * np.pi * f_lib
                kd = ( phi_dg / 1024.0) * \
                        (5.0*np.pi*delta_t * omega_lib**2)
                half_width = np.min([np.max([0.5*kd_to_fit*kd / (2.0*np.pi), 25]), 500])

                int_inds = (freqs > f_lib-half_width) * (freqs < f_lib+half_width)
                integral = np.sum( (asd**2)[int_inds] ) * (freqs[1] - freqs[0])
                Tguess = Ibead['val'] * omega_lib**2 * integral / constants.k

                gamma0_arr[3].append(Tguess)

            else:
                continue

    gamma0_arr = np.array(gamma0_arr)

    gamma0_dict[filename] = {}
    unique_drive_amps = np.unique(gamma0_arr[0])
    for amp in unique_drive_amps:
        inds = (gamma0_arr[0] == amp)
        val = np.mean( gamma0_arr[1][inds] )
        unc = np.std( gamma0_arr[1][inds] )
        Troom = np.mean( gamma0_arr[3][inds] )

        gamma0_dict[filename][amp] = (val, unc, Troom)

for filename in spectra_fit_filenames:
    print()
    print(gamma0_dict[filename])
    print()




fig, ax = plt.subplots(figsize=(6.5,4.0))

for fileind, filename in enumerate(spectra_fit_filenames):
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

        drive_amp = round(actual_amp*1e-3, 0)

        basic_fit = np.array(my_dict['avg_shifted_fit'], \
                       dtype=object)[good_inds][trial_select]
        fit = np.array(my_dict['avg_shifted_squash_fit'], \
                       dtype=object)[good_inds][trial_select]
        freqs = np.array(my_dict['freqs'], \
                         dtype=object)[good_inds][trial_select]
        asd = np.array(my_dict['asd'], \
                       dtype=object)[good_inds][trial_select]

        if not phi_dg:

            basic_fit_arr = np.array(my_dict['avg_shifted_fit'], \
                                     dtype=object)[good_inds]
            asd_arr = np.array(my_dict['asd'], dtype=object)[good_inds]

            mean_fit = np.mean(basic_fit_arr, axis=0)

            f_lib = mean_fit[1]
            fit_freqs = np.linspace(f_lib-half_width, f_lib+half_width, nfreq)
            mean_fit_asd = basic_fit_func(fit_freqs, *mean_fit)

        else:
            f_lib = fit[1]


        omega_lib = 2.0 * np.pi * f_lib

        kd = ( phi_dg / 1024.0) * \
                (5.0*np.pi*delta_t * omega_lib**2)
        half_width = np.min([np.max([0.5*kd_to_fit*kd / (2.0*np.pi), 25]), 500])

        fit_freqs = np.linspace(f_lib-half_width, f_lib+half_width, nfreq)
        fit_asd = fit_func(fit_freqs, *fit)

        int_inds = (freqs > f_lib-half_width) * (freqs < f_lib+half_width)
        integral = np.sum( (asd**2)[int_inds] ) * (freqs[1] - freqs[0])
        Tguess = Ibead['val'] * omega_lib**2 * integral / constants.k

        plot_inds = (freqs > plot_xlim[0]) * (freqs < plot_xlim[1])
        plot_inds = (freqs == freqs)

        if not phi_dg:                    
            gamma0 = gamma0_dict[filename][drive_amp][0]
            Troom = gamma0_dict[filename][drive_amp][2]

            print(Troom)

            # Troom = Tguess
            freqs -= 23.0
            fit_freqs -= 23.0
            # gamma0 = 2.0*np.pi*basic_fit[2]

        Sth = 4.0 * constants.k * Troom * Ibead['val'] * gamma0
        # Sth = fit[0] * Ibead['val']**2

        if not phi_dg:
            gamma_d = 2.0*np.pi*mean_fit[2]
            Teff = ( Ibead['val'] * omega_lib**2 \
                        / (4.0 * constants.k) ) \
                    * ( (Sth / Ibead['val']**2) \
                        / ( omega_lib**2 * gamma_d ) )

            # label = f'0.0, {gamma_d:0.3g},  {Teff:0.3g}'
            label = bu.format_multiple_float_string( \
                        0.0, gamma_d, 100*Teff/Troom, sig_figs=[2,2,2], extra=3)
            label += '  '

        else:
            gamma_d = 2.0*np.pi*fit[2] + fit[4]
            Teff = ( Ibead['val'] * omega_lib**2 \
                        / (4.0 * constants.k) ) \
                    * ( ( (Sth / Ibead['val']**2) \
                        / ( omega_lib**2 * gamma_d ) ) \
                      + ( (fit[4]**2 * fit[3])  / gamma_d  ) )

            # label = f'{kd:0.3g},  {gamma_d:0.3g}, {Teff:0.3g}'
            label = bu.format_multiple_float_string( \
                        kd, gamma_d, 100*Teff/Troom, sig_figs=[2,2,2], extra=3)
            label += '  '


        # print()
        # print('amp, kd, Troom, Teff, Tguess')
        # print(actual_amp, kd, Troom, Teff, Tguess)
        # print()

        if plot_psd:
            asd = asd**2
            fit_asd = fit_asd**2
            mean_fit_asd = mean_fit_asd**2
            asd_arr = asd_arr**2

        if scale_spectra:
            asd *= scaling_fac**phi_dg_ind
            fit_asd *= scaling_fac**phi_dg_ind

        data_args = dict(color=bu.lighten_color(color, lightening_factor), \
                         zorder=4+int(2.0*phi_dg_ind), alpha=data_alpha, lw=2)
        fit_args = dict(color=color, lw=3, ls='--', zorder=5+int(2.0*phi_dg_ind), \
                        label=label, alpha=line_alpha)

        if center_on_zero:
            if not phi_dg:
                freq_offset = mean_fit[1]
            else:
                freq_offset = fit[1]
        else:
            freq_offset = 0.0


        if not phi_dg:
            for i in range(asd_arr.shape[0]):
                # ax.semilogy(freqs[plot_inds]-freq_offset, asd_arr[i][plot_inds], \
                #     **{**data_args, 'color': bu.lighten_color(color, 0.5*lightening_factor)})
                ax.semilogy(freqs[plot_inds]-freq_offset, asd_arr[i][plot_inds], **data_args)
            ax.semilogy(fit_freqs-freq_offset, mean_fit_asd, **fit_args)

        else:
            ax.semilogy(freqs[plot_inds]-freq_offset, asd[plot_inds], **data_args)
            ax.semilogy(fit_freqs-freq_offset, fit_asd, **fit_args)


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
        ax.set_ylabel('Libration PSD [arb. units]')
    elif plot_psd:
        ax.set_ylabel('Libration PSD [rad$^2$/Hz]')
    else:
        ax.set_ylabel('Libration ASD [rad/$\\sqrt{\\rm Hz}$]')

    if center_on_zero:
        ax.set_xlabel('$(\\omega - \\omega_{\\phi})/2\\pi$ [Hz]')
    else:
        ax.set_xlabel('Frequency [Hz]')

    title_str = '$\\hspace{3.25} k_d$ [s$^{-1}$],$\\hspace{1}$' \
                 + '$\\hat{\\gamma}_d$ [s$^{-1}$],$\\hspace{1}$' \
                 + '$T_{\\rm eff} / T_0$ [%]'
    legend = ax.legend(loc='upper right', ncol=1, fontsize=10, \
                       framealpha=1, title=title_str, \
                       title_fontsize=11)
    legend.set_zorder(99)
    plt.setp(legend.texts, family='monospace')

    ax.grid(which='major', axis='y')

    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, plot_name))

    if show:
        plt.show()