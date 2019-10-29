import os, time, sys
import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
from iminuit import Minuit, describe
from datetime import datetime

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet
import dill as pickle

import scipy.optimize as opti
import scipy.signal as signal
import scipy.constants as constants

from tqdm import tqdm
from joblib import Parallel, delayed

n_core = 30

plt.rcParams.update({'font.size': 14})

date = '20191017'

# fig_base = '/home/cblakemore/plots/20190626/'
savefig = False
fig_base = '/home/cblakemore/plots/{:s}/'.format(date)
suffix = ''

#dirname = '/data/old_trap_processed/spinning/ringdown/20190626/'
dirname = '/data/old_trap_processed/spinning/ringdown/{:s}/'.format(date)
#dirname = '/data/old_trap_processed/spinning/ringdown_manual/{:s}/'.format(date)


paths, lengths = bu.find_all_fnames(dirname, ext='.p')

newpaths = paths
# # for 20190626:
# newpaths = [paths[1], paths[2]]
# labels = ['Initial', 'Later']

# mbead = 85.0e-15 # convert picograms to kg
# mbead_err = 1.6e-15

priors = False
manual_priors = True
fterm_prior = 7000.0
fterm_prior_width = 1500.0

fit_end_time = 700.0
exp_fit_end_time = 700.0
two_point_end_time = 50.0

newpaths = [#dirname + '100kHz_start_4_all.p', \
            #dirname + '100kHz_start_5_all.p', \
            #dirname + '100kHz_start_6_all.p', \
            #dirname + '100kHz_start_7_all.p', \
            #dirname + '100kHz_start_8_all.p', \
            #dirname + '100kHz_start_9_all.p', \
            #dirname + '100kHz_start_10_all.p', \
            #dirname + '100kHz_start_11_all.p', \
            #dirname + '100kHz_start_12_all.p', \
            #dirname + '100kHz_start_13_all.p', \
            #dirname + '100kHz_start_14_all.p', \
            #dirname + '50kHz_start_1_all.p', \
            #dirname + '50kHz_start_2_all.p', \
            #dirname + '50kHz_start_3_all.p', \
            #dirname + '110kHz_start_1_all.p', \
            dirname + '110kHz_start_2_all.p', \
            dirname + '110kHz_start_3_all.p', \
            #dirname + '110kHz_start_4_all.p', \
            dirname + '110kHz_start_5_all.p', \
            dirname + '110kHz_start_6_all.p', \
            ]

labels = []
for pathind, path in enumerate(newpaths):
    labels.append('Meas. {:d}'.format(pathind))


fterm_paths = [dirname + 'term_velocity_check_1.npy', \
               dirname + 'term_velocity_check_2.npy', \
               #dirname + 'term_velocity_check_3.npy', \
               dirname + 'term_velocity_check_4.npy', \
               dirname + 'term_velocity_check_5.npy', \
               dirname + 'term_velocity_check_6.npy', \
               dirname + 'term_velocity_check_7.npy', \
              ]

all_fterm = []
for pathind, path in enumerate(fterm_paths):
    data = np.load(open(path, 'rb'))
    #plt.plot(data[1])
    #plt.show()
    all_fterm += list(data[1])
all_fterm = np.array(all_fterm)

vals, bin_edge = np.histogram(all_fterm)
bins = bin_edge[:-1] + 0.5*(bin_edge[1] - bin_edge[0])

plot_x = np.linspace(bins[0], bins[-1], 100)

def gauss(x, A, mu, sigma, c):
    return A * np.exp( -1.0 * (x - mu)**2 / (2.8 * sigma**2)) + c

def fit_fun(x, A, mu, sigma):
    return gauss(x, A, mu, sigma, 0)

prior_popt, prior_pcov = opti.curve_fit(fit_fun, bins, vals, p0=[10, 8000, 1000])
# plt.hist(all_fterm)
# plt.plot(plot_x, fit_fun(plot_x, *popt), lw=2, color='r', ls='--')
# plt.show()

fterm_prior = prior_popt[1]
fterm_prior_width = 5.0*np.abs(prior_popt[2])

print 'PRIORS: ', fterm_prior, fterm_prior_width



def line(x, a, b):
    return a * x + b

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def exp_decay(x, f0, tau, fopt):
    return f0 * np.exp(-1.0 * x / tau) + fopt

fig, axarr = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3,1]}, \
                            sharex=True, figsize=(6,4),dpi=200)

fig2, ax2 = plt.subplots(1,1, #gridspec_kw = {'height_ratios':[1,1]}, \
                            sharex=True, figsize=(6,3) ,dpi=200)

kb = constants.Boltzmann
T = 297

## Assuming base pressure limited by water
m0 = 18.0 * 1.66e-27


mbead = bu.get_mbead(date)

rbead = bu.get_rbead(mbead)
Ibead = bu.get_Ibead(mbead)



print 'Optical torque estimate: ', Ibead['val'] * 20.0e3 / 1500.0

kappa = {}
kappa['val'] = 6.47e11
kappa['sterr'] = 0.06e11
kappa['syserr'] = 0.25e11

kappa_calc = bu.get_kappa(mbead)

#colors = bu.get_color_map(len(newpaths)*2 + 1, cmap='plasma')
colors = bu.get_color_map(len(newpaths), cmap='plasma')

two_point_estimates = []

for fileind, file in enumerate(newpaths):

    if fileind < 3:
        fterm_prior = 7000.0
    else:
        fterm_prior = prior_popt[1]

    fterm_prior_width = 1000.0

    print 'PRIORS: ', fterm_prior, fterm_prior_width

    #color = 'C{:d}'.format(fileind)
    #color = colors[fileind*2 + 1]
    color = colors[fileind]
    label = labels[fileind]

    data = pickle.load(open(file, 'rb'))
    times = data['times']
    all_time = data['all_time']
    all_freq = data['all_freq']
    all_freq_err = data['all_freq_err']
    f0 = data['init_freq']

    popt_line, pcov_line = opti.curve_fit(line, all_time[0], all_freq[0], p0=[0, f0])
    t0_line = (f0 - popt_line[1]) /  popt_line[0]

    print f0, 'at', t0_line

    #ndat = len(all_time)
    two_point_estimates.append([])
    ndat = 0
    for i in range(len(all_time)):
        t_mean = np.mean(all_time[i])
        freq_mean = np.mean(all_freq[i])
        if t_mean < two_point_end_time:
            val = (t_mean - t0_line) / np.log(f0 / freq_mean)
            two_point_estimates[-1].append(val)
        if t_mean > exp_fit_end_time:
            break
        ndat += 1

    all_time_flat = all_time.flatten()
    all_freq_flat = all_freq.flatten()
    all_freq_err_flat = all_freq_err.flatten()

    #print times

    #print np.mean(all_freq_err_flat), np.std(all_freq_err_flat)
    print datetime.utcfromtimestamp(int(data['t_init']*1e-9)).strftime('%Y-%m-%d %H:%M:%S')

    fit_inds = all_time_flat < fit_end_time
    npts = np.sum(fit_inds)
    xdat = all_time_flat[fit_inds]
    ydat = all_freq_flat[fit_inds]
    yerr = all_freq_err_flat[fit_inds]


    def fit_fun(x, t0, tau, fterm):
        return f0 * np.exp(-1.0 * (x - t0) / tau) + (fterm) * ( 1 - np.exp(-1.0 * (x - t0) / tau) )

    def fit_fun_2(x, t0, tau, fterm):
        return f0 * np.exp(-1.0 * (x - t0) / tau) + fterm

    def chisquare_1d(t0, tau, fterm):
        resid = ydat - fit_fun(xdat, t0, tau, fterm)
        return (1.0 / (npts - 1.0)) * np.sum(resid**2 / yerr**2)

    def chisquare_1d_2(tau, fterm):
        resid = ydat - fit_fun(xdat, t0_line, tau, fterm)
        return (1.0 / (npts - 1.0)) * np.sum(resid**2 / yerr**2)

    # m=Minuit(chisquare_1d,
    #          f0 = 50000.0,
    #          tau = 1500.0, # set start parameter
    #          limit_tau = (500.0, 4000.0), # if you want to limit things
    #          #fix_a = "True", # you can also fix it
    #          fopt = 1000.0,
    #          limit_fopt = (0.0, 10000.0),
    #          errordef = 1,
    #          print_level = 1, 
    #          pedantic=False)
    # m.migrad(ncall=500000)
    # plt.figure()
    # m.draw_mnprofile('tau', bound = 50, bins = 50)

    print np.mean(two_point_estimates[-1])

    m=Minuit(chisquare_1d_2,
             #t0 = 0,
             #fix_t0 = t0_line, 
             #limit_t0 = (-5, 5),
             tau = 2000.0, # set start parameter
             limit_tau = (500.0, 4000.0), # if you want to limit things
             #fix_a = "True", # you can also fix it
             fterm = 8000.0,
             #limit_fopt = (0.0, 10000.0),
             errordef = 1,
             print_level = 0, 
             pedantic=False)
    m.migrad(ncall=500000)
    minos = m.minos()
    # plt.figure()
    # m.draw_mnprofile('tau', bound = 5, bins = 50)
    # plt.show()

    print
    print file

    # popt_modexp = [m.values['t0'], m.values['tau'], m.values['fterm']]
    popt_modexp = [t0_line, m.values['tau'], m.values['fterm']]


    # plt.figure()
    # plt.semilogy(xdat, np.abs(yerr))
    # plt.figure()
    # plt.plot(xdat, ydat)
    # plt.plot(xdat, fit_fun(xdat, *popt_modexp))
    # plt.show()

    #prior_data = [m.values['t0'], np.mean(np.abs([minos['t0']['upper'], minos['t0']['lower']])), \
    #                 m.values['fterm'], np.mean(np.abs([minos['fterm']['upper'], minos['fterm']['lower']]))]

    prior_data = [m.values['fterm'], \
                    np.mean(np.abs([minos['fterm']['upper'], minos['fterm']['lower']]))]
    #prior_data[1] = 1500.0

    zero_term_prior_width = np.abs(m.values['fterm'])

    #tau_all = minos['tau']['min']
    #tau_all_err = np.mean(np.abs([minos['tau']['upper'], minos['tau']['lower']]))

    plot_x = np.linspace(all_time_flat[0], all_time_flat[int(np.sum(fit_inds)-1)], 500)

    last_ind = np.sum(times < exp_fit_end_time) + 1


    def fit_ringdown(int_index):
        t_mean = np.mean(all_time[int_index])
        t_last = np.max(all_time[int_index]) + 0.25

        # bu.progress_bar(i, last_ind+1)

        fit_inds = all_time_flat < t_last
        derp_inds = all_time_flat < fit_end_time

        npts = np.sum(fit_inds)
        xdat = all_time_flat[fit_inds]
        ydat = all_freq_flat[fit_inds]
        yerr = all_freq_err_flat[fit_inds]
        #yerr = np.sqrt(ydat)
        #yerr = np.random.randn(npts) * np.std(yerr) + np.mean(yerr)

        if priors:
            def chisquare_1d(tau, fterm):
                resid = ydat - fit_fun(xdat, t0_line, tau, fterm)
                prior = (fterm - prior_data[0])**2 / prior_data[1]**2
                chisq = (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)
                return chisq + prior
        elif manual_priors:
            def chisquare_1d(tau, fterm):
                resid = ydat - fit_fun(xdat, t0_line, tau, fterm)
                prior = (fterm - fterm_prior)**2 / fterm_prior_width**2
                chisq = (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)
                return chisq + prior
        else:
            def chisquare_1d(tau, fterm):
                resid = ydat - fit_fun(xdat, t0_line, tau, fterm)
                chisq = (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)
                return chisq
        # m=Minuit(chisquare_1d,
        #          f0=50000.0,
        #          tau = 1500.0, # set start parameter
        #          limit_tau = (500.0, 4000.0), # if you want to limit things
        #          #fix_a = "True", # you can also fix it
        #          fopt = 1000.0,
        #          limit_fopt = (0.0, 10000.0),
        #          errordef = 1,
        #          print_level = 0, 
        #          pedantic=False)
        # m.migrad(ncall=500000)

        m2=Minuit(chisquare_1d,
                 #t0 = 0.0,
                 #fix_t0 = t0_line,
                 #limit_t0 = (-5, 5),
                 tau = np.mean(two_point_estimates[-1]), # set start parameter
                 #fix_tau = True,
                 #limit_tau = (500.0, 4000.0), # if you want to limit things
                 fterm = prior_data[0], # set start parameter
                 # limit_tau = (500.0, 4000.0), # if you want to limit things
                 #fix_a = "True", # you can also fix it
                 # limit_fopt = (0.0, 10000.0),
                 errordef = 1,
                 print_level = 0, 
                 pedantic=False)
        m2.migrad(ncall=500000)

        # plt.figure()
        # m.draw_mnprofile('tau', bound = 3, bins = 50)
        # plt.figure()
        # m2.draw_mnprofile('tau', bound = 3, bins = 50)
        # plt.show()

        # p0 = [f0, 1500.0, 1000.0]
        # popt = [m2.values['t0'], m2.values['tau'], m2.values['fterm']]

        # # print tau_many
        # # print tau_many_2

        # if m.values['tau'] < 1200:
        #     plt.figure()
        #     plt.plot(all_time_flat[derp_inds], all_freq_flat[derp_inds])
        #     plt.plot(all_time_flat[fit_inds], all_freq_flat[fit_inds])
        #     plt.plot(all_time_flat[derp_inds], exp_decay(all_time_flat[derp_inds], *p0), '--', \
        #                 color='k', lw=4, alpha=0.5)
        #     plt.plot(all_time_flat[derp_inds], exp_decay(all_time_flat[derp_inds], *popt), '--', \
        #                 color='r', lw=4, alpha=0.5)
        #     plt.figure()
        #     plt.plot(yerr)
        #     plt.show()

        try:
            minos = m2.minos()
            return [t_mean, minos['tau']['min'], minos['tau']['upper'], \
                    minos['tau']['lower'], minos['fterm']['min']]
        except:
            return [t_mean, m2.values['tau'], m2.errors['tau'], \
                    m2.errors['tau'], m2.values['fterm']]

    ind_list = []
    for int_index in range(ndat):
        if int_index > last_ind:
            break
        ind_list.append(int_index)

    results = Parallel(n_jobs=n_core)(delayed(fit_ringdown)(ind) for ind in tqdm(ind_list))

    results = np.array(results).T
    time_many = results[0]
    tau_many = results[1]
    tau_upper = results[2]
    tau_lower = results[3]
    fterm_many = results[4]

    # tau_many = np.array(tau_many)
    # tau_upper = np.array(tau_upper)
    # tau_lower = np.array(tau_lower)

    tau_fit_err = np.mean(np.abs([tau_upper[-1], tau_lower[-1]]))
    popt_modexp = [t0_line, tau_many[-1], fterm_many[-1]]

    print 'TAU     : ', tau_many[-1]
    print 'TAU ERR : ', tau_fit_err
    print 'FTERM   : ', fterm_many[-1]

    label = ('$\\tau_{{{:d}}} = ({:d} \\pm {:d})$ s'\
                                .format(fileind, int(tau_many[-1]), int(tau_fit_err)))

    ax2.plot(time_many, tau_many, color=color, label='$\\tau_{{{:d}}}$'.format(fileind))
    ax2.fill_between(time_many, tau_many + tau_upper, tau_many + tau_lower, \
                            facecolor=color, edgecolor=color, linewidth=0, alpha=0.35)
    #axarr2[1].plot(time_many, 100.0 * (np.array(tau_many) - tau_many[-1]) / tau_many[-1], color=color)

    #print plot_x
    axarr[0].plot(all_time_flat, all_freq_flat*1e-3, color=color)#, label=label)
    # axarr[0].plot(plot_x, fit_fun(plot_x, *popt_exp), '--', lw=3, color=color, alpha=0.75, \
    #                 label=(label + ': $\\tau=$' + '{:0.1f} s'.format(-1.0/popt_exp[1])))
    axarr[0].plot(plot_x, fit_fun(plot_x, *popt_modexp)*1e-3, ls=(0, (1, 1)), \
                    lw=4, color=color, alpha=0.75, label=label)
    #axarr[1].plot(all_time_flat, all_freq_flat - fit_fun(all_time_flat, *popt_exp), color=color)
    axarr[1].plot(all_time_flat, all_freq_flat - fit_fun(all_time_flat, *popt_modexp), color=color)

    p = (kappa['val'] / np.sqrt(m0)) * (Ibead['val'] / tau_many[-1])
    p_sterr = p * np.sqrt( (kappa['sterr'] / kappa['val'])**2 + \
                           (Ibead['sterr'] / Ibead['val'])**2 )
    p_syserr = p * np.sqrt( (kappa['syserr'] / kappa['val'])**2 + \
                            (Ibead['syserr'] / Ibead['val'])**2 + \
                            (tau_fit_err / tau_many[-1])**2 )

    p2 = (kappa_calc['val'] / np.sqrt(m0)) * (Ibead['val'] / tau_many[-1])
    p2_sterr = p2 * np.sqrt( (kappa_calc['sterr'] / kappa_calc['val'])**2 + \
                            (Ibead['sterr'] / Ibead['val'])**2 )
    p2_syserr = p2 * np.sqrt( (kappa_calc['syserr'] / kappa_calc['val'])**2 + \
                             (Ibead['syserr'] / Ibead['val'])**2 + \
                             (tau_fit_err / tau_many[-1])**2 )

    #print 'kappa unc. - ', kappa_err / kappa
    #print 'Ibead unc. - ', Ibead_err / Ibead
    #print 'tau unc. - ', tau_all_err / tau_all

    print 'PRESSURE: {:0.3g} +- {:0.3g} (st) +- {:0.3g} (sys) mbar'\
                .format(p * 0.01, p_sterr * 0.01, p_syserr * 0.01)#, p_err * 0.01)

    print 'PRESSURE: {:0.3g} +- {:0.3g} (st) +- {:0.3g} (sys) mbar'\
                .format(p2 * 0.01, p2_sterr * 0.01, p2_syserr * 0.01)#, p_err * 0.01)

print two_point_estimates
for i in range(len(two_point_estimates)):
    print np.mean(two_point_estimates[i])

axarr[0].set_xlim(0,700)
#axarr[0].set_ylim(30, 52)
axarr[0].set_ylim(72, 112)
#axarr[0].set_ylim(30, 102)
axarr[1].set_ylim(-120, 120)
axarr[1].set_xlabel('Time [s]')
axarr[0].set_ylabel('Rot. Freq. [kHz]')
axarr[1].set_ylabel('Resid. [Hz]')
axarr[0].legend(loc='upper right', fontsize=10)
fig.tight_layout()

ax2.set_xlabel('Length of data used in exponential fit [s]')
ax2.set_ylabel('Extracted $\\tau$ [s]')
ax2.set_xlim(0,200)
ax2.set_ylim(1500,2500)
ax2.legend(fontsize=10, loc='upper right')
# axarr2[1].set_xlabel('Time [s]')
# axarr2[1].set_ylabel('$\\Delta \\tau$ [%]')
# axarr2[0].set_ylabel('$\\tau$ [s]')
# axarr2[0].legend(fontsize=10)

fig2.tight_layout()

if savefig:
    # fig.savefig('/home/charles/plots/20190626/spindowns.png')
    # fig.savefig('/home/charles/plots/20190626/spindowns.svg')
    fig.savefig(fig_base + 'spindowns{:s}.png'.format(suffix))
    fig.savefig(fig_base + 'spindowns{:s}.svg'.format(suffix))

    # fig2.savefig('/home/charles/plots/20190626/spindown_time.png')
    # fig2.savefig('/home/charles/plots/20190626/spindown_time.svg')
    fig2.savefig(fig_base + 'spindown_time{:s}.png'.format(suffix))
    fig2.savefig(fig_base + 'spindown_time{:s}.svg'.format(suffix))

plt.show()