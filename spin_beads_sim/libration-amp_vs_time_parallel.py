import os, sys, time, h5py

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.optimize as opti
import scipy.constants as constants

from obspy.signal.detrend import polynomial

import bead_util as bu

import dill as pickle

from joblib import Parallel, delayed

ncore = 20
# ncore = 1

plt.rcParams.update({'font.size': 14})


base = '/data/spin_sim_data/libration_tests/'

# dirname = os.path.join(base, 'high_pressure_sweep')
# dirname = os.path.join(base, 'libration_ringdown_manyp_3_hf')
# dirname = os.path.join(base, 'thermalization_manyp')
# dirname = os.path.join(base, 'amp_noise_test_fterm')
dirname = os.path.join(base, 'rot_freq_sweep')
n_mc = bu.count_subdirectories(dirname)

# maxfile = 3
maxfile = 500

hdf5 = True
ext = '.h5'

### Paths for saving
save = True
save_base = '/home/cblakemore/opt_lev_analysis/spin_beads_sim/processed_results/'
save_filename = os.path.join(save_base, 'rot_freq_sweep.p')

### Use this option with care: if you parallelize and ask it to plot,
### you'll get ncore * (a few) plots up simultaneously
plot_first_file = False
plot_raw_dat = False
plot_thermalization_window = False

### Constants
dipole_units = constants.e * (1e-6) # to convert e um -> C m

### Bead-specific constants
p0 = 100.0 * dipole_units  #  C * m


# rhobead = {'val': 1850.0, 'sterr': 1.0, 'syserr': 1.0}
mbead_dic = {'val': 84.3e-15, 'sterr': 1.0e-15, 'syserr': 1.5e-15}
mbead = mbead_dic['val']
Ibead = bu.get_Ibead(mbead=mbead_dic)['val']
kappa = bu.get_kappa(mbead=mbead_dic)['val']

m0 = 18.0 * constants.atomic_mass

############################################################################
############################################################################
############################################################################
############################################################################

colors = bu.get_color_map(n_mc, cmap='plasma')[::1]


def proc_mc(i):
    cdir = os.path.join(dirname, 'mc_{:d}'.format(i))

    param_path = os.path.join(cdir, 'params.p')
    params = pickle.load( open(param_path, 'rb') )

    pressure = params['pressure']
    drive_amp = params['drive_amp']
    drive_amp_noise = params['drive_amp_noise']
    drive_phase_noise = params['drive_phase_noise']
    discretized_phase = params['discretized_phase']
    drive_freq = params['drive_freq']
    fsig = params['drive_freq']
    p0 = params['p0']
    Ibead = params['Ibead']
    t_therm = params['t_therm']
    beta_rot = params['beta_rot']
    try:
        init_angle = params['init_angle']
    except Exception:
        init_angle = np.pi / 2.0

    try:
        fsamp = params['fsamp']
    except Exception:
        fsamp = 1.0e6

    beta_rot = pressure * np.sqrt(m0) / kappa
    phieq = -1.0 * np.arcsin(2.0 * np.pi * drive_freq * beta_rot / (drive_amp * p0))

    # print(phieq)

    offset_phi = 2.0 * np.pi * drive_freq * t_therm + init_angle

    # print(pressure, time_constant)


    def Ephi(t, t_therm=0.0):
        return 2.0 * np.pi * drive_freq * (t + t_therm)

    def energy(xi, t, ndim=3):
        omega = xi[ndim:]
        omega_frame = 2.0 * np.pi * drive_freq

        omega[1] -= omega_frame

        kinetic_term = 0.5 * Ibead * np.sum(omega**2, axis=0)
        potential_term = -1.0 * p0 * drive_amp * (np.sin(xi[0]) * np.cos(Ephi(t) - xi[1]))

        return kinetic_term + potential_term


    datfiles, lengths = bu.find_all_fnames(cdir, ext=ext, verbose=False, \
                                            sort_time=True, use_origin_timestamp=True)
    nfiles = lengths[0]

    integrated_energy = []
    all_energy = np.array([])
    all_t_energy = np.array([])
    all_amp = np.array([])
    all_t_amp = np.array([])

    plot = False
    for fileind, file in enumerate(datfiles):

        if fileind >= maxfile:
            break

        if plot_first_file:
            plot = not fileind

        try:
            if hdf5:
                fobj = h5py.File(file, 'r')
                dat = np.copy(fobj['sim_data'])
                fobj.close()
            else:
                dat = np.load(file)

        except Exception:
            print('Bad File!')
            print(file)
            continue

        nsamp = dat.shape[1]
        ndim = int((dat.shape[0] - 1) / 2)

        if plot_raw_dat:
            fig1, axarr1 = plt.subplots(2,1,sharex=True)
            axarr1[0].set_title('$\\theta$ - Azimuthal Coordinate (Locked)')
            axarr1[0].plot(dat[0], dat[1], zorder=2)
            axarr1[0].axhline(np.pi/2, color='k', alpha=0.7, ls='--', zorder=1, \
                                label='Equatorial plane')
            axarr1[0].set_ylabel('$\\theta$ [rad]')
            axarr1[1].plot(dat[0], dat[1+ndim])
            axarr1[1].set_xlabel('Time [s]')
            axarr1[1].set_ylabel('$\\omega_{\\theta}$ [rad/s]')
            if plot_thermalization_window:
                xlims = axarr1[0].get_xlim()
                ylims0 = axarr1[0].get_ylim()
                ylims1 = axarr1[1].get_ylim()
                xvec = np.linspace(xlims[0], t_therm, 10)
                top0 = np.ones_like(xvec) * ylims0[1]
                bot0 = np.ones_like(xvec) * ylims0[0]
                top1 = np.ones_like(xvec) * ylims1[1]
                bot1 = np.ones_like(xvec) * ylims1[0]
                axarr1[0].fill_between(xvec, bot0, top0, color='k', alpha=0.3, \
                                        label='Thermalization time')
                axarr1[1].fill_between(xvec, bot1, top1, color='k', alpha=0.3, \
                                        label='Thermalization time')
                axarr1[0].set_xlim(*xlims)
                axarr1[0].set_ylim(*ylims0)
                axarr1[1].set_ylim(*ylims1)
            axarr1[0].legend(fontsize=10, loc='lower right')
            fig1.tight_layout()

            fig2a, axarr2a = plt.subplots(2,1,sharex=True)
            axarr2a[0].set_title('$\\phi$ - Rotating Coordinate')
            axarr2a[0].plot(dat[0], dat[2])
            axarr2a[0].set_ylabel('$\\phi$ [rad]')
            axarr2a[1].plot(dat[0], dat[2+ndim])
            axarr2a[1].set_xlabel('Time [s]')
            axarr2a[1].set_ylabel('$\\omega_{\\phi}$ [rad/s]')
            if plot_thermalization_window:
                xlims = axarr2a[0].get_xlim()
                ylims0 = axarr2a[0].get_ylim()
                ylims1 = axarr2a[1].get_ylim()
                xvec = np.linspace(xlims[0], t_therm, 10)
                top0 = np.ones_like(xvec) * ylims0[1]
                bot0 = np.ones_like(xvec) * ylims0[0]
                top1 = np.ones_like(xvec) * ylims1[1]
                bot1 = np.ones_like(xvec) * ylims1[0]
                axarr2a[0].fill_between(xvec, bot0, top0, color='k', alpha=0.3, \
                                        label='Thermalization time')
                axarr2a[1].fill_between(xvec, bot1, top1, color='k', alpha=0.3, \
                                        label='Thermalization time')
                axarr2a[0].set_xlim(*xlims)
                axarr2a[0].set_ylim(*ylims0)
                axarr2a[1].set_ylim(*ylims1)
            axarr2a[0].legend(fontsize=10, loc='lower right')
            fig2a.tight_layout()

            fig2b, axarr2b = plt.subplots(2,1,sharex=True)
            axarr2b[0].set_title("$\\phi'$ - In Rotating Frame")
            axarr2b[0].plot(dat[0], dat[2] - 2.0*np.pi*drive_freq*dat[0] - offset_phi, zorder=2)
            axarr2b[0].axhline(phieq, color='k', alpha=0.7, ls='--', zorder=1, \
                                label='Expected Equilibrium value')
            axarr2b[0].set_ylabel('$\\phi$ [rad]')
            axarr2b[1].plot(dat[0], dat[2+ndim]-2.0*np.pi*drive_freq)
            axarr2b[1].set_xlabel('Time [s]')
            axarr2b[1].set_ylabel('$\\omega_{\\phi}$ [rad/s]')
            if plot_thermalization_window:
                xlims = axarr2b[0].get_xlim()
                ylims0 = axarr2b[0].get_ylim()
                ylims1 = axarr2b[1].get_ylim()
                xvec = np.linspace(xlims[0], t_therm, 10)
                top0 = np.ones_like(xvec) * ylims0[1]
                bot0 = np.ones_like(xvec) * ylims0[0]
                top1 = np.ones_like(xvec) * ylims1[1]
                bot1 = np.ones_like(xvec) * ylims1[0]
                axarr2b[0].fill_between(xvec, bot0, top0, color='k', alpha=0.3, \
                                        label='Thermalization time')
                axarr2b[1].fill_between(xvec, bot1, top1, color='k', alpha=0.3, \
                                        label='Thermalization time')
                axarr2b[0].set_xlim(*xlims)
                axarr2b[0].set_ylim(*ylims0)
                axarr2b[1].set_ylim(*ylims1)
            axarr2b[0].legend(fontsize=10, loc='lower right')
            fig2b.tight_layout()

            if ndim == 3:
                fig3, axarr3 = plt.subplots(2,1,sharex=True)
                axarr3[0].set_title('$\\psi$ - Roll About Dipole (Free)')
                axarr3[0].plot(dat[0], dat[3])
                axarr3[0].set_ylabel('$\\psi$ [rad]')
                axarr3[1].plot(dat[0], dat[3+ndim])
                axarr3[1].set_xlabel('Time [s]')
                axarr3[1].set_ylabel('$\\omega_{\\psi}$ [rad/s]')
                if plot_thermalization_window:
                    xlims = axarr3[0].get_xlim()
                    ylims0 = axarr3[0].get_ylim()
                    ylims1 = axarr3[1].get_ylim()
                    xvec = np.linspace(xlims[0], t_therm, 10)
                    top0 = np.ones_like(xvec) * ylims0[1]
                    bot0 = np.ones_like(xvec) * ylims0[0]
                    top1 = np.ones_like(xvec) * ylims1[1]
                    bot1 = np.ones_like(xvec) * ylims1[0]
                    axarr3[0].fill_between(xvec, bot0, top0, color='k', alpha=0.3, \
                                            label='Thermalization time')
                    axarr3[1].fill_between(xvec, bot1, top1, color='k', alpha=0.3, \
                                            label='Thermalization time')
                    axarr3[0].set_xlim(*xlims)
                    axarr3[0].set_ylim(*ylims0)
                    axarr3[1].set_ylim(*ylims1)
                axarr3[0].legend(fontsize=10, loc='lower right')
                fig3.tight_layout()

            plt.show()

            input()

        tvec = dat[0]
        theta = dat[1]
        phi = dat[2]

        energy_vec = energy(dat[1:], tvec, ndim=ndim)
        freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)
        energy_psd = np.abs( np.fft.rfft(energy_vec-np.mean(energy_vec)) )**2 * bu.fft_norm(nsamp, fsamp)**2
        energy_asd = np.sqrt(energy_psd)
        # plt.loglog(freqs, energy_asd*freqs)
        # plt.show()

        integrated_energy.append(np.sqrt(np.sum(energy_psd) * (freqs[1] - freqs[0])))

        crossp = np.sin(phi)**2

        carrier_amp, carrier_phase \
                = bu.demod(crossp, fsig, fsamp, harmind=2.0, filt=True, \
                           bandwidth=4000.0, plot=plot, tukey=True, \
                           tukey_alpha=1e-3)

        params, cov = bu.fit_damped_osc_amp(carrier_phase, fsamp, plot=plot)

        libration_amp, libration_phase \
                = bu.demod(carrier_phase, params[1], fsamp, harmind=1.0, \
                           filt=True, filt_band=[100, 2000], plot=plot, \
                           tukey=True, tukey_alpha=1e-3)

        amp_ds, tvec_ds = signal.resample(libration_amp, 500, t=tvec, window=None)
        amp_ds_cut = amp_ds[5:-5]
        tvec_ds_cut = tvec_ds[5:-5]

        energy_ds, tvec_ds_2 = signal.resample(energy_vec, 100, t=tvec, window=None)
        energy_ds_cut = energy_ds[5:-5]
        tvec_ds_cut_2 = tvec_ds_2[5:-5]

        all_amp = np.concatenate( (all_amp, amp_ds_cut) )
        all_t_amp = np.concatenate( (all_t_amp, tvec_ds_cut) )

        all_energy = np.concatenate( (all_energy, energy_ds_cut))
        all_t_energy = np.concatenate( (all_t_energy, tvec_ds_cut_2) )

    return [pressure, all_t_amp, all_amp, all_t_energy, all_energy, integrated_energy, \
            drive_amp, drive_amp_noise, drive_phase_noise, discretized_phase]



results = Parallel(n_jobs=ncore)( delayed(proc_mc)(ind) for ind in list(range(n_mc))[::1] )

if save:
    pickle.dump(results, open(save_filename, 'wb'))

fig1, ax1 = plt.subplots(1,1)
fig2, ax2 = plt.subplots(1,1)
fig3, ax3 = plt.subplots(1,1)

for ind, result in enumerate(results[::1]):
    pressure, all_t_amp, all_amp, all_t_energy, all_energy, integrated_energy, \
        drive_amp, drive_amp_noise, drive_phase_noise, discretized_phase = result
    # lab = '{:0.3g} mbar'.format(pressure)
    lab = '$ (\\sigma_{{V}} / V) = {:0.3g}$'.format(drive_amp_noise/drive_amp)
    ax1.plot(all_t_amp, all_amp, color=colors[ind], label=lab)
    ax2.plot(all_t_energy, all_energy, color=colors[ind], label=lab)
    ax3.plot(integrated_energy, color=colors[ind], label=lab)

ax1.legend(fontsize=10, loc='upper right')
ax2.legend(fontsize=10, loc='upper right')
ax3.legend(fontsize=10, loc='upper right')

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude of Phase Modulation [rad]')

ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Energy (kinetic + potential) [J]')

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

plt.show()


