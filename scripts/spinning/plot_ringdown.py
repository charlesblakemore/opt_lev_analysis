import os, time, sys
import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
from iminuit import Minuit, describe

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet
import dill as pickle

import scipy.optimize as opti
import scipy.signal as signal

plt.rcParams.update({'font.size': 14})


dirname = '/processed_data/spinning/ringdown/20190626/'
paths, lengths = bu.find_all_fnames(dirname, ext='.p')

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def exp_decay(x, f0, tau, fopt):
    return f0 * np.exp(-1.0 * x / tau) + fopt


newpaths = [paths[1], paths[2]]
labels = ['Initial', 'Later']
fig, axarr = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3,1]}, \
                            sharex=True, figsize=(6,4),dpi=200)

fig2, ax2 = plt.subplots(1,1, #gridspec_kw = {'height_ratios':[1,1]}, \
                            sharex=True, figsize=(6,3) ,dpi=200)

kb = 1.3806e-23
T = 297
m0 = 18.0 * 1.66e-27

mbead = 85.0e-15 # convert picograms to kg
mbead_err = 1.6e-15
rhobead = 1550.0 # kg/m^3
rhobead_err = 80.0

rbead = ( (mbead / rhobead) / ((4.0/3.0)*np.pi) )**(1.0/3.0)
rbead_err = rbead * np.sqrt( (1.0/3.0)*(mbead_err/mbead)**2 + \
                                (1.0/3.0)*(rhobead_err/rhobead)**2 )

Ibead = 0.4 * (3.0 / (4.0 * np.pi))**(2.0/3.0) * mbead**(5.0/3.0) * rhobead**(-2.0/3.0)
Ibead_err = Ibead * np.sqrt( (5.0/3.0)*(mbead_err/mbead)**2 + \
                                (2.0/3.0)*(rhobead_err/rhobead)**2 )

print 'Optical torque estimate: ', Ibead * 20.0e3 / 1500.0

kappa = 6.32e11
kappa_err = 0.07e11

colors = bu.get_color_map(len(newpaths)*2 + 1, cmap='plasma')

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

    ndat = len(all_time)

    all_time_flat = all_time.flatten()
    all_freq_flat = all_freq.flatten()
    all_freq_err_flat = all_freq_err.flatten()

    print np.mean(all_freq_err_flat), np.std(all_freq_err_flat)

    fit_inds = all_time_flat < 500.0
    npts = np.sum(fit_inds)
    xdat = all_time_flat[fit_inds]
    ydat = all_freq_flat[fit_inds]
    yerr = all_freq_err_flat[fit_inds]

    p0 = [50000.0, -0.5e-3, 0]
    p0_2 = [-0.5e-3, 10.0, 0]


    def fit_fun(x, t0, tau, b):
        return f0 * np.exp(-1.0 * (x - t0) / tau) + (b * tau) * ( 1 - np.exp(-1.0 * (x - t0) / tau) )

    def chisquare_1d_2(t0, tau, b):
        resid = ydat - fit_fun(xdat, t0, tau, b)
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
             t0 = -10.0,
             tau = 1500.0, # set start parameter
             limit_tau = (500.0, 4000.0), # if you want to limit things
             #fix_a = "True", # you can also fix it
             b = 10.0,
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
    print m.values

    popt_modexp = [m.values['t0'], m.values['tau'], m.values['b']]

    prior_data = [m.values['t0'], np.mean(np.abs([minos['t0']['upper'], minos['t0']['lower']])), \
                    m.values['b'], np.mean(np.abs([minos['b']['upper'], minos['b']['lower']]))]

    tau_all = minos['tau']['min']
    tau_all_err = np.mean(np.abs([minos['tau']['upper'], minos['tau']['lower']]))

    plot_x = np.linspace(all_time_flat[0], all_time_flat[int(np.sum(fit_inds)-1)], 500)

    last_ind = np.sum(times < 500.0) + 1

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
        derp_inds = all_time_flat < 1000.0

        npts = np.sum(fit_inds)
        xdat = all_time_flat[fit_inds]
        ydat = all_freq_flat[fit_inds]
        yerr = all_freq_err_flat[fit_inds]
        #yerr = np.sqrt(ydat)
        #yerr = np.random.randn(npts) * np.std(yerr) + np.mean(yerr)

        def chisquare_1d(f0, tau, fopt):
            resid = ydat - exp_decay(xdat, f0, tau, fopt)
            return (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)

        def chisquare_1d_2(t0, tau, b):
            resid = ydat - fit_fun(xdat, t0, tau, b)
            prior1 = (t0 - prior_data[0])**2 / prior_data[1]**2
            prior2 = (b - prior_data[2])**2 / prior_data[3]**2
            return (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2) + prior1 + prior2

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
                 tau = 1500.0, # set start parameter
                 limit_tau = (500.0, 4000.0), # if you want to limit things
                 b = 2.0, # set start parameter
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

        if len(tau_many):
            if np.abs(m.values['tau'] - tau_many[-1]) / tau_many[-1] > 0.33:
                continue

        tau_many.append(minos['tau']['min'])
        tau_upper.append(minos['tau']['upper'])
        tau_lower.append(minos['tau']['lower'])
        #tau_err.append(m.errors['tau'])
        time_many.append(t_mean)

        p0 = [50000.0, 1500.0, 1000.0]
        popt = [m2.values['t0'], m2.values['tau'], m2.values['b']]

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

    ax2.plot(time_many, tau_many, color=color, label=label)
    ax2.fill_between(time_many, tau_many + tau_upper, tau_many + tau_lower, \
                            facecolor=color, edgecolor=color, linewidth=0, alpha=0.35)
    #axarr2[1].plot(time_many, 100.0 * (np.array(tau_many) - tau_many[-1]) / tau_many[-1], color=color)

    #print plot_x
    axarr[0].plot(all_time_flat, all_freq_flat*1e-3, color=color)#, label=label)
    # axarr[0].plot(plot_x, fit_fun(plot_x, *popt_exp), '--', lw=3, color=color, alpha=0.75, \
    #                 label=(label + ': $\\tau=$' + '{:0.1f} s'.format(-1.0/popt_exp[1])))
    axarr[0].plot(plot_x, fit_fun(plot_x, *popt_modexp)*1e-3, '--', lw=4, color=color, alpha=0.75, \
                    label=(label + ': $\\tau = ({:d} \\pm {:d})$ s'.format(int(tau_all), int(tau_all_err))))
    #axarr[1].plot(all_time_flat, all_freq_flat - fit_fun(all_time_flat, *popt_exp), color=color)
    axarr[1].plot(all_time_flat, all_freq_flat - fit_fun(all_time_flat, *popt_modexp), color=color)

    p = (kappa / np.sqrt(m0)) * (Ibead / tau_all)
    p_err = p * np.sqrt( (kappa_err / kappa)**2 + (Ibead_err / Ibead)**2 + (tau_all_err / tau_all)**2)

    print label + ': {:0.3g} +- {:0.3g} mbar'.format(p * 0.01, p_err * 0.01)

axarr[0].set_xlim(0,700)
axarr[0].set_ylim(30, 52)
axarr[1].set_ylim(-50, 50)
axarr[1].set_xlabel('Time [s]')
axarr[0].set_ylabel('Rot. Freq. [kHz]')
axarr[1].set_ylabel('Resid. [Hz]')
axarr[0].legend(loc='upper right', fontsize=10)
fig.tight_layout()

ax2.set_xlabel('Length of data used in exponential fit [s]')
ax2.set_ylabel('Extracted $\\tau$ [s]')
ax2.set_xlim(0,500)
ax2.legend(fontsize=10)
# axarr2[1].set_xlabel('Time [s]')
# axarr2[1].set_ylabel('$\\Delta \\tau$ [%]')
# axarr2[0].set_ylabel('$\\tau$ [s]')
# axarr2[0].legend(fontsize=10)

fig2.tight_layout()

fig.savefig('/home/charles/plots/20190626/spindowns.png')
fig.savefig('/home/charles/plots/20190626/spindowns.svg')

fig2.savefig('/home/charles/plots/20190626/spindown_time.png')
fig2.savefig('/home/charles/plots/20190626/spindown_time.svg')

plt.show()