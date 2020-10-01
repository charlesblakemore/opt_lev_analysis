import os, sys, time, itertools, re, warnings
import numpy as np
import dill as pickle
from iminuit import Minuit, describe

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as optimize
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed
# ncore = 1
ncore = 24

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 14})



###################################################################################
###################################################################################
###################################################################################


def ringdown_time(phi_dg, tau0, scale):
    return tau0 / (1.0 + scale*phi_dg)


max_phi_dg = 20
max_spectra_dg = 5
selection_freqs = (1100, 1500)

spectra_xlim = (-20, 20)
# spectra_xlim = (-30, 20)
spectra_ylim = (4e-4, 0.075)

processed_base = '/data/old_trap_processed/spinning/20200727/'
plot_base = '/home/cblakemore/plots/20200727/spinning/'

ringdown_fig_path = os.path.join(plot_base, '20200727_libration_impulse_damping_time_with_feedback.svg')
spectra_fig_path = os.path.join(plot_base, '20200727_libration_spectra_with_feedback_corrected_fewer.svg')
save = True

ringdown_data_path = os.path.join(processed_base, 'dds_libration_ringdowns.p')
spectra_save_path = os.path.join(processed_base, 'dds_feedback_spectra.p')

ringdown_dict = pickle.load( open(ringdown_data_path, 'rb') )
spectra_dict = pickle.load( open(spectra_save_path, 'rb') )


phi_dgs = list(ringdown_dict.keys())
phi_dgs.sort()

phi_dgs = np.array(phi_dgs)
inds = (phi_dgs > 0.0) * (phi_dgs <= max_phi_dg)

phi_dgs = phi_dgs[inds]

taus = np.zeros((2, len(phi_dgs)))

fit0 = np.array( ringdown_dict[0.0]['fit'] )
unc0 = np.array( ringdown_dict[0.0]['unc'] )

tau0, tau0_unc = bu.weighted_mean(fit0[:,2], unc0[:,2])


for i, phi_dg in enumerate(phi_dgs):

    fits = np.array( ringdown_dict[phi_dg]['fit'] )
    uncs = np.array( ringdown_dict[phi_dg]['unc'] )

    tau, tau_unc = bu.weighted_mean(fits[:,2], uncs[:,2])

    taus[0,i] = tau
    taus[1,i] = tau_unc


fit_func = lambda phi_dg, scale: ringdown_time(phi_dg, tau0, scale)
fit_func_log = lambda phi_dg, scale: np.log(ringdown_time(phi_dg, tau0, scale))

npts = len(phi_dgs)

def chi_sq(scale):
    resid = (np.abs(taus[0]) - fit_func(phi_dgs, scale))**2
    variance = taus[1]**2
    return (1.0 / (npts - 1.0)) * np.sum(resid / variance)

def chi_sq_log(scale):
    resid = np.abs(np.log(taus[0]) - fit_func_log(phi_dgs, scale))**2
    variance = (taus[1] / taus[0])**2
    return (1.0 / (npts - 1.0)) * np.sum(resid / variance)

m = Minuit(chi_sq,
           scale = 200.0, # set start parameter
           fix_scale = False,
           # limit_scale = (np.pi/4.0, 3.0*np.pi/4.0), # if you want to limit things
           errordef = 1,
           print_level = 0, 
           pedantic=False)
m.migrad(ncall=500000)

ringdown_red_chi_sq = chi_sq(m.values['scale'])



fig, ax = plt.subplots(1,1,figsize=(6,4))

ax.errorbar(phi_dgs, taus[0,:], yerr=taus[1,:], ls='', color='C0')
ax.set_xscale('log')
ax.set_yscale('log')

xlim = ax.get_xlim()
# plot_x0 = np.linspace(xlim[0], xlim[1], 100)
plot_x0 = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 100)
plot_y0 = np.ones(100) * tau0
ax.plot(plot_x0, plot_y0, color='k', lw=2, label='Zero feedback damping time')
ax.plot(plot_x0, ringdown_time(plot_x0, tau0, m.values['scale']), color='C0', \
        alpha=0.4, label='$\\chi^2/N_{\\rm dof} =$' + '{:0.2f}'.format(ringdown_red_chi_sq))
ax.fill_between(plot_x0, plot_y0+tau0_unc, plot_y0-tau0_unc, \
                color='k', alpha=0.4)
ax.text(2,0.5,ha='center', va='center', fontdict={'size': 16}, \
        s='$\\tau = \\frac{\\tau_0}{1 + k \\, g_d} $')
ax.set_xlim(*xlim)

ax.set_xlabel('Derivative Gain [arb]')
ax.set_ylabel('Damping Time [s]')

ax.legend(loc='lower left', fontsize=12)

fig.tight_layout()

if save:
    fig.savefig(ringdown_fig_path)







fig2, ax2 = plt.subplots(1,1,figsize=(8,5))

# max_dg_val = np.max(phi_dgs)
min_dg_val = np.min(phi_dgs)

max_dg_val = max_spectra_dg

def dg_to_yval(dg):
    scaled = (dg - min_dg_val) / (max_dg_val - min_dg_val)
    return scaled * (spectra_ylim[1] - spectra_ylim[0]) + spectra_ylim[0]

norm = colors.LogNorm(vmin=min_dg_val, vmax=max_dg_val)

divider = make_axes_locatable(ax2)
ax_cb = divider.new_horizontal(size='2%', pad=0.05)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.plasma, \
                                norm=norm, orientation='vertical')
cb1.set_label('Derivative Gain [arb]')

fig2.add_axes(ax_cb)

freqs0 = spectra_dict[0.0]['freqs'][0]
data0 = spectra_dict[0.0]['data']
mask0 = (freqs0 > selection_freqs[0]) * (freqs0 < selection_freqs[1])
center_freqs = []
for i in range(len(data0)):
    trial0 = data0[i]
    for j in range(len(trial0)):
        fft = trial0[j]
        max_freq = freqs0[np.argmax(np.abs(fft)*mask0)]
        center_freqs.append(max_freq)
freq0 = np.mean(center_freqs)


avg_asds = []
phi_dgs = phi_dgs[::-1]
for phi_dg in phi_dgs[::8]:

    if phi_dg > max_spectra_dg:
        continue

    color = bu.get_single_color(phi_dg, cmap='plasma', log=True,\
                                vmin=min_dg_val, vmax=max_spectra_dg)

    try:
        full_freqs = spectra_dict[phi_dg]['freqs'][0]
        data = spectra_dict[phi_dg]['data']
    except:
        continue

    mask = (full_freqs > selection_freqs[0]) * (full_freqs < selection_freqs[1])

    center_freqs = []
    all_freqs = []
    all_spectra = []
    all_maxind = []
    for i in range(len(data)):
        trial = data[i] 
        for j in range(len(trial)):
            fft = trial[j]
            maxind = np.argmax(np.abs(fft)*mask)
            all_freqs.append(full_freqs[maxind-500:maxind+500] - full_freqs[maxind])
            all_spectra.append(fft[maxind-500:maxind+500])
            center_freqs.append(full_freqs[maxind])

    freq_offset = freq0 - np.mean(center_freqs)

    all_spectra = np.array(all_spectra)

    avg_asd = np.sqrt( np.mean(np.abs(all_spectra)**2, axis=0) )

    ax2.semilogy(all_freqs[0], avg_asd, color=color)

ax2.set_xlabel('$f - f_{\\phi}$ [Hz]')
# ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('ASD of Phase Modulation [rad/$\\sqrt{\\rm Hz}$]')

ax2.set_xlim(*spectra_xlim)

fig2.tight_layout()

if save:
    fig2.savefig(spectra_fig_path)


plt.show()















