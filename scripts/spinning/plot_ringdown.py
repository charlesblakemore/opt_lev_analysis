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

plt.rcParams.update({'font.size': 14})

date = '20190905'

# fig_base = '/home/cblakemore/plots/20190626/'
savefig = False
fig_base = '/home/cblakemore/plots/{:s}/'.format(date)
suffix = '_no-priors'

#dirname = '/data/old_trap_processed/spinning/ringdown/20190626/'
dirname = '/data/old_trap_processed/spinning/ringdown/{:s}/'.format(date)


paths, lengths = bu.find_all_fnames(dirname, ext='.p')

newpaths = paths
# # for 20190626:
# newpaths = [paths[1], paths[2]]
# labels = ['Initial', 'Later']

# mbead = 85.0e-15 # convert picograms to kg
# mbead_err = 1.6e-15

priors = True #False
zero_term_velocity = True
zero_term_prior_width = 5000
fit_end_time = 1000.0
exp_fit_end_time = 300.0

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
            dirname + '50kHz_start_2_all.p', \
            dirname + '50kHz_start_3_all.p', \
            ]

labels = []
for pathind, path in enumerate(newpaths):
    labels.append('Meas. {:d}'.format(pathind))


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def exp_decay(x, f0, tau, fopt):
    return f0 * np.exp(-1.0 * x / tau) + fopt

fig, axarr = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3,1]}, \
                            sharex=True, figsize=(6,4),dpi=200)

fig2, ax2 = plt.subplots(1,1, #gridspec_kw = {'height_ratios':[1,1]}, \
                            sharex=True, figsize=(6,3) ,dpi=200)

kb = 1.3806e-23
T = 297

## Assuming base pressure limited by water
m0 = 18.0 * 1.66e-27


mbead, mbead_sterr, mbead_syserr = bu.get_mbead(date)

rbead, rbead_sterr, rbead_syserr = bu.get_rbead(mbead, mbead_sterr, mbead_syserr)
Ibead, Ibead_sterr, Ibead_syserr = bu.get_Ibead(mbead, mbead_sterr, mbead_syserr)



print('Optical torque estimate: ', Ibead * 20.0e3 / 1500.0)

kappa = 6.09e11
kappa_err = 0.03e11

colors = bu.get_colormap(len(newpaths)*2 + 1, cmap='plasma')

for fileind, file in enumerate(newpaths):
    color = 'C{:d}'.format(fileind)
    color = colors[fileind*2 + 1]
    label = labels[fileind]

    data = pickle.load(open(file, 'rb'))
    times = data['times']
    all_time = data['all_time']
    all_freq = data['all_freq']
    all_freq_err = data['all_freq_err']
    f0 = data['init_freq']
    print(f0)

    #ndat = len(all_time)
    ndat = 0
    for i in range(len(all_time)):
        t_mean = np.mean(all_time[i])
        if t_mean > exp_fit_end_time:
            break
        ndat += 1

    all_time_flat = all_time.flatten()
    all_freq_flat = all_freq.flatten()
    all_freq_err_flat = all_freq_err.flatten()

    #print times

    print(np.mean(all_freq_err_flat), np.std(all_freq_err_flat))
    print(datetime.utcfromtimestamp(int(data['t_init']*1e-9)).strftime('%Y-%m-%d %H:%M:%S'))

    fit_inds = all_time_flat < fit_end_time
    npts = np.sum(fit_inds)
    xdat = all_time_flat[fit_inds]
    ydat = all_freq_flat[fit_inds]
    yerr = all_freq_err_flat[fit_inds]


    def fit_fun(x, t0, tau, fterm):
        return f0 * np.exp(-1.0 * (x - t0) / tau) + (fterm) * ( 1 - np.exp(-1.0 * (x - t0) / tau) )

    def chisquare_1d_2(t0, tau, fterm):
        resid = ydat - fit_fun(xdat, t0, tau, fterm)
        return (1.0 / (npts - 1.0)) * np.sum(resid**2 / yerr**2)

    def chisquare_1d(f0, tau, fopt):
        resid = ydat - exp_decay(xdat, f0, tau, fopt)
        return (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)

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

    m=Minuit(chisquare_1d_2,
             t0 = 0,
             tau = 1200.0, # set start parameter
             limit_tau = (500.0, 4000.0), # if you want to limit things
             #fix_a = "True", # you can also fix it
             fterm = 10000.0,
             #limit_fopt = (0.0, 10000.0),
             errordef = 1,
             print_level = 0, 
             pedantic=False)
    m.migrad(ncall=500000)
    minos = m.minos()
    # plt.figure()
    # m.draw_mnprofile('tau', bound = 5, bins = 50)
    # plt.show()

    print()
    print(file)
    print(m.values)

    popt_modexp = [m.values['t0'], m.values['tau'], m.values['fterm']]


    # plt.figure()
    # plt.semilogy(xdat, np.abs(yerr))
    # plt.figure()
    # plt.plot(xdat, ydat)
    # plt.plot(xdat, fit_fun(xdat, *popt_modexp))
    # plt.show()

    prior_data = [m.values['t0'], np.mean(np.abs([minos['t0']['upper'], minos['t0']['lower']])), \
                    m.values['fterm'], np.mean(np.abs([minos['fterm']['upper'], minos['fterm']['lower']]))]

    zero_term_prior_width = np.abs(m.values['fterm'])

    tau_all = minos['tau']['min']
    tau_all_err = np.mean(np.abs([minos['tau']['upper'], minos['tau']['lower']]))

    plot_x = np.linspace(all_time_flat[0], all_time_flat[int(np.sum(fit_inds)-1)], 500)

    last_ind = np.sum(times < exp_fit_end_time) + 1

    time_many = []
    tau_many = []
    tau_upper = []
    tau_lower = []
    for i in range(ndat):
        t_mean = np.mean(all_time[i])
        t_last = np.max(all_time[i]) + 0.25
        if i > last_ind:
            break
        bu.progress_bar(i, last_ind+1)

        fit_inds = all_time_flat < t_last
        derp_inds = all_time_flat < fit_end_time

        npts = np.sum(fit_inds)
        xdat = all_time_flat[fit_inds]
        ydat = all_freq_flat[fit_inds]
        yerr = all_freq_err_flat[fit_inds]
        #yerr = np.sqrt(ydat)
        #yerr = np.random.randn(npts) * np.std(yerr) + np.mean(yerr)

        def chisquare_1d(f0, tau, fopt):
            resid = ydat - exp_decay(xdat, f0, tau, fopt)
            return (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)

        if priors:
            if zero_term_velocity:
                def chisquare_1d_2(t0, tau, fterm):
                    resid = ydat - fit_fun(xdat, t0, tau, fterm)
                    prior1 = (t0 - prior_data[0])**2 / prior_data[1]**2
                    prior2 = (fterm)**2 / zero_term_prior_width**2
                    chisq = (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)
                    return chisq + prior1 + prior2
            else:
                def chisquare_1d_2(t0, tau, fterm):
                    resid = ydat - fit_fun(xdat, t0, tau, fterm)
                    prior1 = (t0 - prior_data[0])**2 / prior_data[1]**2
                    prior2 = (fterm - prior_data[2])**2 / prior_data[3]**2
                    chisq = (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)
                    return chisq + prior1 + prior2
        else:
            def chisquare_1d_2(t0, tau, fterm):
                resid = ydat - fit_fun(xdat, t0, tau, fterm)
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

        m2=Minuit(chisquare_1d_2,
                 t0 = 0.0,
                 tau = 1200.0, # set start parameter
                 limit_tau = (500.0, 4000.0), # if you want to limit things
                 fterm = 10000.0, # set start parameter
                 # limit_tau = (500.0, 4000.0), # if you want to limit things
                 #fix_a = "True", # you can also fix it
                 # limit_fopt = (0.0, 10000.0),
                 errordef = 1,
                 print_level = 0, 
                 pedantic=False)
        m2.migrad(ncall=500000)

        minos = m2.minos()

        # plt.figure()
        # m.draw_mnprofile('tau', bound = 3, bins = 50)
        # plt.figure()
        # m2.draw_mnprofile('tau', bound = 3, bins = 50)
        # plt.show()

        # if len(tau_many):
        #     if np.abs(m.values['tau'] - tau_many[-1]) / tau_many[-1] > 0.33:
        #         continue

        tau_many.append(minos['tau']['min'])
        tau_upper.append(minos['tau']['upper'])
        tau_lower.append(minos['tau']['lower'])
        #tau_err.append(m.errors['tau'])
        time_many.append(t_mean)

        p0 = [f0, 1500.0, 1000.0]
        popt = [m2.values['t0'], m2.values['tau'], m2.values['fterm']]

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

    tau_many = np.array(tau_many)
    tau_upper = np.array(tau_upper)
    tau_lower = np.array(tau_lower)

    print('Original tau error: ', tau_all_err)
    print('Final successive fit value: ', np.mean(np.abs([tau_upper[-1], tau_lower[-1]])))

    label = ('$\\tau_{{{:d}}} = ({:d} \\pm {:d})$ s'\
                                .format(fileind, int(tau_all), int(tau_all_err)))

    ax2.plot(time_many, tau_many, color=color, label='$\\tau_{{{:d}}}$'.format(fileind))
    ax2.fill_between(time_many, tau_many + tau_upper, tau_many + tau_lower, \
                            facecolor=color, edgecolor=color, linewidth=0, alpha=0.35)
    #axarr2[1].plot(time_many, 100.0 * (np.array(tau_many) - tau_many[-1]) / tau_many[-1], color=color)

    #print plot_x
    axarr[0].plot(all_time_flat, all_freq_flat*1e-3, color=color)#, label=label)
    # axarr[0].plot(plot_x, fit_fun(plot_x, *popt_exp), '--', lw=3, color=color, alpha=0.75, \
    #                 label=(label + ': $\\tau=$' + '{:0.1f} s'.format(-1.0/popt_exp[1])))
    axarr[0].plot(plot_x, fit_fun(plot_x, *popt_modexp)*1e-3, '--', lw=4, color=color, alpha=0.75, \
                    label=label)
    #axarr[1].plot(all_time_flat, all_freq_flat - fit_fun(all_time_flat, *popt_exp), color=color)
    axarr[1].plot(all_time_flat, all_freq_flat - fit_fun(all_time_flat, *popt_modexp), color=color)

    p = (kappa / np.sqrt(m0)) * (Ibead / tau_all)
    #p_err = p * np.sqrt( (kappa_err / kappa)**2 + (Ibead_err / Ibead)**2 + (tau_all_err / tau_all)**2)

    print('kappa unc. - ', kappa_err / kappa)
    #print 'Ibead unc. - ', Ibead_err / Ibead
    print('tau unc. - ', tau_all_err / tau_all)

    print(label + ': {:0.3g} +- {:0.3g} mbar'.format(p * 0.01, 0))#, p_err * 0.01)

axarr[0].set_xlim(0,700)
#axarr[0].set_ylim(30, 52)
#axarr[0].set_ylim(52, 102)
axarr[0].set_ylim(30, 102)
axarr[1].set_ylim(-100, 100)
axarr[1].set_xlabel('Time [s]')
axarr[0].set_ylabel('Rot. Freq. [kHz]')
axarr[1].set_ylabel('Resid. [Hz]')
axarr[0].legend(loc='upper right', fontsize=10)
fig.tight_layout()

ax2.set_xlabel('Length of data used in exponential fit [s]')
ax2.set_ylabel('Extracted $\\tau$ [s]')
ax2.set_xlim(0,200)
ax2.set_ylim(400,1800)
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