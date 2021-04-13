import os, time, sys, io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tight_layout as tlt
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

n_core = 20

plt.rcParams.update({'font.size': 14})

date = '20200322'
date = '20200924'

# fig_base = '/home/cblakemore/plots/20190626/'
savefig = True
fig_base = '/home/cblakemore/plots/{:s}/spinning/'.format(date)
#fig_base = '/home/cblakemore/plots/spinsim/'
suffix = ''
# suffix = '_less-yrange'
#suffix = '_3_5e-6mbar_110kHz_real-noise'

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
manual_priors = False
fix_fterm = False

fit_end_time = 3000.0
exp_fit_end_time = 3000.0
two_point_end_time = 3000.0
tau_ylim = (1100, 1400)
# tau_ylim = (1850,2050)
both_two_point = False

err_adjust = 5.0

# newpaths = [#dirname + '100kHz_start_4_all.p', \
#             #dirname + '100kHz_start_5_all.p', \
#             #dirname + '100kHz_start_6_all.p', \
#             #dirname + '100kHz_start_7_all.p', \
#             #dirname + '100kHz_start_8_all.p', \
#             #dirname + '100kHz_start_9_all.p', \
#             #dirname + '100kHz_start_10_all.p', \
#             #dirname + '100kHz_start_11_all.p', \
#             #dirname + '100kHz_start_12_all.p', \
#             #dirname + '100kHz_start_13_all.p', \
#             #dirname + '100kHz_start_14_all.p', \
#             #dirname + '50kHz_start_1_all.p', \
#             #dirname + '50kHz_start_2_all.p', \
#             #dirname + '50kHz_start_3_all.p', \
#             #dirname + '110kHz_start_1_all.p', \
#             #dirname + '110kHz_start_2_all.p', \
#             #dirname + '110kHz_start_3_all.p', \
#             #dirname + '110kHz_start_4_all.p', \
#             #dirname + '110kHz_start_5_all.p', \
#             #dirname + '110kHz_start_6_all.p', \
#             dirname + '110kHz_start_2_coarse_all.p', \
#             dirname + '110kHz_start_3_coarse_all.p', \
#             dirname + '110kHz_start_5_coarse_all.p', \
#             dirname + '110kHz_start_6_coarse_all.p', \
#             ]

newpaths = [\
            # os.path.join(dirname, '110kHz_start_1_all.p'), \
            os.path.join(dirname, '110kHz_start_2_all.p'), \
            os.path.join(dirname, '110kHz_start_3_all.p'), \
           ]


sim_data = False
sim_path = '/data/old_trap_processed/spinsim_data/spindowns_processed/sim_110kHz_real-noise/'
sim_fig_base = '/home/cblakemore/plots/spinsim/'
sim_suffix = '_3_5e-6mbar_110kHz_real-noise'
paths, lengths = bu.find_all_fnames(sim_path, ext='.p')

sim_prior_data = [0.0, 1]

if sim_data:
    newpaths = paths[:50]

labels = []
for pathind, path in enumerate(newpaths):
    labels.append('Meas. {:d}'.format(pathind))





def gauss(x, A, mu, sigma, c):
    return A * np.exp( -1.0 * (x - mu)**2 / (2.8 * sigma**2)) + c


def ngauss(x, A, mu, sigma, c, n=2):
    return A * np.exp(-1.0*np.abs(x-mu)**n / (2.0*sigma**n)) + c

def fit_fun(x, A, mu, sigma):
    return ngauss(x, A, mu, sigma, 0, n=5)




#if manual_priors:
# fterm_dirname = '/data/old_trap_processed/spinning/ringdown/20191017/'
fterm_dirname = '/data/old_trap_processed/spinning/ringdown/20200322/'
fterm_paths = [fterm_dirname + 'term_velocity_check_1.npy', \
               fterm_dirname + 'term_velocity_check_2.npy', \
               #fterm_dirname + 'term_velocity_check_3.npy', \
               # fterm_dirname + 'term_velocity_check_4.npy', \
               # fterm_dirname + 'term_velocity_check_5.npy', \
               # fterm_dirname + 'term_velocity_check_6.npy', \
               # fterm_dirname + 'term_velocity_check_7.npy', \
              ]

all_fterm = []
for pathind, path in enumerate(fterm_paths):
    data = np.load(open(path, 'rb'))
    #plt.plot(data[1])
    #plt.show()
    all_fterm += list(data[1])
all_fterm = np.array(all_fterm)


fig_term, ax_term = plt.subplots(1,1,dpi=200)

vals, bin_edge, _ = ax_term.hist(all_fterm, density=True)
bins = bin_edge[:-1] + 0.5*(bin_edge[1] - bin_edge[0])

prior_popt, prior_pcov = opti.curve_fit(fit_fun, bins, vals, maxfev=10000,\
                                        p0=[1, np.mean(all_fterm), np.std(all_fterm)])

plot_x = np.linspace(np.mean(all_fterm)-np.std(all_fterm), \
                     np.mean(all_fterm)+np.std(all_fterm), 100)
plot_x_2 = np.linspace(np.mean(all_fterm) - 3.0*np.std(all_fterm), \
                       np.mean(all_fterm) + 3.0*np.std(all_fterm), 100)
ax_term.plot(plot_x, 0.5*np.max(vals)*np.ones_like(plot_x), color='r', ls='--')
ax_term.plot(plot_x_2, fit_fun(plot_x_2, *prior_popt), color='r', lw=2)
ax_term.axvline(np.mean(all_fterm), color='r', ls='--')
ax_term.set_xlabel('Terminal Rotation Freq. [Hz]')
ax_term.set_ylabel('Counts [arb.]')
# plt.plot(plot_x, fit_fun(plot_x, *prior_popt), lw=2, color='r', ls='--')
fig_term.tight_layout()
fig_term.savefig(fig_base + 'terminal_velocity.svg')
# plt.show()

prior_popt = np.abs(prior_popt)






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

fig3, ax3 = plt.subplots(1,1, #gridspec_kw = {'height_ratios':[1,1]}, \
                            sharex=True, figsize=(6,3) ,dpi=200)

kb = constants.Boltzmann
T = 297

## Assuming base pressure limited by water
m0 = 18.0 * constants.atomic_mass


mbead = bu.get_mbead(date)

rbead = bu.get_rbead(mbead)
Ibead = bu.get_Ibead(mbead)



print('Optical torque estimate: ', Ibead['val'] * 20.0e3 / 1500.0)

kappa = {}
kappa['val'] = 6.47e11
kappa['sterr'] = 0.06e11
kappa['syserr'] = 0.25e11

kappa_calc = bu.get_kappa(mbead)

#colors = bu.get_color_map(len(newpaths)*2 + 1, cmap='plasma')
colors = bu.get_color_map(len(newpaths), cmap='plasma')

two_point_times = []
two_point_estimates = []
two_point_errors = []
two_point_estimates_2 = []
two_point_errors_2 = []
linear_estimates = []
linear_errors = []

all_fits = []

for fileind, file in enumerate(newpaths):

    try:
        #if fileind < 3:
        #    fterm_prior = 7000.0
        #else:
        #fterm_prior = prior_popt[1]
        fterm_prior = np.mean(all_fterm)
        #fterm_prior_width = prior_popt[2]
        fterm_prior_width = np.std(all_fterm)
        #fterm_prior_width = 1000.0
    except:
        fterm_prior = 0.0
        fterm_prior_width = 1e9

    #fterm_prior = 7000.0
    #fterm_prior_width = 50.0

    print('PRIORS: ', fterm_prior, fterm_prior_width)

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

    print(f0, 'at', t0_line)



    all_time_flat = all_time.flatten()
    all_freq_flat = all_freq.flatten()
    all_freq_err_flat = all_freq_err.flatten() * err_adjust #* 500.0

    #ndat = len(all_time)
    two_point_times.append([])
    two_point_estimates.append([])
    two_point_errors.append([])
    two_point_estimates_2.append([])
    two_point_errors_2.append([])
    linear_estimates.append([])
    linear_errors.append([])
    ndat = 0
    last_freq = f0
    last_t = 0
    for i in range(len(all_time)):
        t_mean = np.mean(all_time[i])
        freq_mean = np.mean(all_freq[i])
        if t_mean < two_point_end_time:
            two_point_times[-1].append(t_mean)

            logterm = np.log((f0 - fterm_prior) / (freq_mean - fterm_prior))

            val = (t_mean - t0_line) / logterm
            two_point_estimates[-1].append(val)

            var = 0.0
            var += err_adjust*np.median(all_freq_err[i])**2 * \
                    (val / ((freq_mean - fterm_prior) * logterm) )**2
            var += fterm_prior_width**2 * ( (freq_mean - f0) * val / \
                                            ( (freq_mean - fterm_prior) * \
                                              (f0 - fterm_prior) * logterm ) )**2
            two_point_errors[-1].append(np.sqrt(var))

            if i != 0:
                logterm2 = np.log((last_freq - fterm_prior) / (freq_mean - fterm_prior))
                val2 = (t_mean - last_t) / logterm2
                two_point_estimates_2[-1].append(val2)
                if i == 1:
                    two_point_estimates_2[-1].append(val2)

                var2 = 0.0
                var2 += err_adjust*np.median(all_freq_err[i])**2 * \
                        (val2 / ((freq_mean - fterm_prior) * logterm2) )**2
                var2 += err_adjust*np.median(all_freq_err[i])**2 * \
                        (val2 / ((last_freq - fterm_prior) * logterm2) )**2
                var2 += fterm_prior_width**2 * ( (freq_mean - last_freq) * val2 / \
                                                ( (freq_mean - fterm_prior) * \
                                                  (last_freq - fterm_prior) * logterm2 ) )**2
                two_point_errors_2[-1].append(np.sqrt(var2))
                if i == 1:
                    two_point_errors_2[-1].append(np.sqrt(var2))


            fit_inds = all_time_flat <= all_time[i][-1]
            popt_line, pcov_line = \
                opti.curve_fit(line, all_time_flat[fit_inds], all_freq_flat[fit_inds], \
                                sigma=all_freq_err_flat[fit_inds], absolute_sigma=True, \
                                p0=[0, f0])
            val3 = (fterm_prior - f0) / popt_line[0]
            linear_estimates[-1].append(val3)

            var3 = 0.0
            var3 += val3**2 * (t_mean / val3)**4
            var3 += val3**2 * pcov_line[0,0] / popt_line[0]**2
            var3 += val3**2 * (fterm_prior_width / fterm_prior)**2
            linear_errors[-1].append(np.sqrt(var3))

        last_freq = freq_mean
        last_t = t_mean

        if t_mean > exp_fit_end_time:
            break
        ndat += 1


    #print times

    #print np.mean(all_freq_err_flat), np.std(all_freq_err_flat)
    print(datetime.utcfromtimestamp(int(data['t_init']*1e-9)).strftime('%Y-%m-%d %H:%M:%S'))

    #continue

    fit_inds = all_time_flat < fit_end_time
    npts = np.sum(fit_inds)
    xdat = all_time_flat[fit_inds]
    ydat = all_freq_flat[fit_inds]
    yerr = all_freq_err_flat[fit_inds]

    def fit_fun(t, t0, tau, fterm):
        return f0 * np.exp(-1.0 * (t - t0) / tau) + \
                    (fterm) * ( 1 - np.exp(-1.0 * (t - t0) / tau) )

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

    two_point_tau = np.mean(two_point_estimates[-1])
    two_point_tau_err = np.std(two_point_estimates[-1])
    print(two_point_tau)

    m=Minuit(chisquare_1d_2,
             #t0 = 0,
             #fix_t0 = t0_line, 
             #limit_t0 = (-5, 5),
             #tau = 2000.0, # set start parameter
             tau = two_point_tau,
             #limit_tau = (500.0, 4000.0), # if you want to limit things
             #fix_tau = True,
             #fix_a = "True", # you can also fix it
             fterm = fterm_prior,
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
    if sim_data:
        prior_data = sim_prior_data
    #prior_data[1] = 10.0

    plot_x = np.linspace(all_time_flat[0], all_time_flat[int(np.sum(fit_inds)-1)], 500)

    last_ind = np.sum(times < exp_fit_end_time) + 1


    def fit_ringdown(int_index):

        #print fterm_prior, fterm_prior_width

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
            fterm_init = prior_data[0]
            def chisquare_1d(tau, fterm):
                resid = ydat - fit_fun(xdat, t0_line, tau, fterm)
                prior = (fterm - prior_data[0])**2 / prior_data[1]**2
                chisq = (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)
                return chisq + prior

        elif manual_priors:
            fterm_init = fterm_prior
            def chisquare_1d(tau, fterm):
                resid = ydat - fit_fun(xdat, t0_line, tau, fterm)
                prior = (fterm - fterm_prior)**2 / fterm_prior_width**2
                chisq = (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)
                return chisq + prior

        else:
            fterm_init = 0.0
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
                 tau = two_point_tau, # set start parameter
                 #fix_tau = True,
                 #limit_tau = (500.0, 4000.0), # if you want to limit things
                 fterm = fterm_init, # set start parameter
                 fix_fterm = fix_fterm,
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

    # tau_fit_err = np.mean(np.abs([tau_upper[-1], tau_lower[-1]]))
    # popt_modexp = [t0_line, tau_many[-1], fterm_many[-1]]

    inds = np.array(two_point_times[-1]) < 500.0
    popt_modexp = [t0_line, np.mean(np.array(two_point_estimates[-1])[inds]), fterm_prior]
    tau_fit_err = np.mean([np.max(np.array(two_point_errors[-1])[inds]), \
                            np.median(np.array(two_point_errors[-1])[inds])])

    # inds = time_many < 500.0
    # popt_modexp = [t0_line, np.mean(tau_many[inds]), np.mean(fterm_many[inds])]
    # tau_fit_err = np.mean(np.abs([tau_upper[inds], tau_lower[inds]]))

    print('TAU     : ', popt_modexp[1])
    print('TAU ERR : ', tau_fit_err)
    print('FTERM   : ', popt_modexp[2])

    label = ('$\\tau_{{{:d}}} = ({:d} \\pm {:d})$ s'\
                                .format(fileind, int(popt_modexp[1]), int(tau_fit_err)))
    # label += ',    $f_{{opt, {:d}}} = {:d}$ Hz'.format(fileind, int(popt_modexp[2]))

    # label = ('$\\tau_{{{:d}}} = ({:d} \\pm {:d})$ s'\
    #                             .format(fileind, int(np.mean(two_point_estimates[-1])), \
    #                                     int(np.max(two_point_errors[-1]))))

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

    p = (kappa['val'] / np.sqrt(m0)) * (Ibead['val'] / popt_modexp[1])
    p_sterr = p * np.sqrt( (kappa['sterr'] / kappa['val'])**2 + \
                           (Ibead['sterr'] / Ibead['val'])**2 )
    p_syserr = p * np.sqrt( (kappa['syserr'] / kappa['val'])**2 + \
                            (Ibead['syserr'] / Ibead['val'])**2 + \
                            (tau_fit_err / popt_modexp[1])**2 )

    p2 = (kappa_calc['val'] / np.sqrt(m0)) * (Ibead['val'] / popt_modexp[1])
    p2_sterr = p2 * np.sqrt( (kappa_calc['sterr'] / kappa_calc['val'])**2 + \
                            (Ibead['sterr'] / Ibead['val'])**2 )
    p2_syserr = p2 * np.sqrt( (kappa_calc['syserr'] / kappa_calc['val'])**2 + \
                             (Ibead['syserr'] / Ibead['val'])**2 + \
                             (tau_fit_err / popt_modexp[1])**2 )

    #print 'kappa unc. - ', kappa_err / kappa
    #print 'Ibead unc. - ', Ibead_err / Ibead
    #print 'tau unc. - ', tau_all_err / tau_all

    print('PRESSURE: {:0.3g} +- {:0.3g} (st) +- {:0.3g} (sys) mbar'\
                .format(p * 0.01, p_sterr * 0.01, p_syserr * 0.01))#, p_err * 0.01)

    print('PRESSURE: {:0.3g} +- {:0.3g} (st) +- {:0.3g} (sys) mbar'\
                .format(p2 * 0.01, p2_sterr * 0.01, p2_syserr * 0.01))#, p_err * 0.01)

    all_fits.append( [popt_modexp[1], tau_fit_err, p2*0.01, p2_sterr*0.01, p2_syserr*0.01] )



min_len = 10000
for sublist in two_point_times:
    sublist_len = len(sublist)
    if sublist_len < min_len:
        min_len = sublist_len
for sublist_ind, sublist in enumerate(two_point_times):
    two_point_times[sublist_ind]       =  sublist[:min_len]
    two_point_estimates[sublist_ind]   = two_point_estimates[sublist_ind][:min_len]
    two_point_errors[sublist_ind]      = two_point_errors[sublist_ind][:min_len]
    two_point_estimates_2[sublist_ind] = two_point_estimates_2[sublist_ind][:min_len]
    two_point_errors_2[sublist_ind]    = two_point_errors_2[sublist_ind][:min_len]
    linear_estimates[sublist_ind]      = linear_estimates[sublist_ind][:min_len]
    linear_errors[sublist_ind]         = linear_errors[sublist_ind][:min_len]

two_point_times = np.array(two_point_times)
two_point_estimates = np.array(two_point_estimates)
two_point_errors = np.array(two_point_errors)
two_point_estimates_2 = np.array(two_point_estimates_2)
two_point_errors_2 = np.array(two_point_errors_2)
linear_estimates = np.array(linear_estimates)
linear_errors = np.array(linear_errors)

colors = bu.get_color_map(len(two_point_times), cmap='plasma')
for sublist_ind, sublist in enumerate(two_point_times):
    # lab1 = 'Two-point: {:d}'.format(sublist_ind)
    # lab2 = 'Linear: {:d}'.format(sublist_ind)
    # ax3.plot(two_point_times[sublist_ind], linear_estimates[sublist_ind], \
    #             color=colors[sublist_ind], label=lab2)
    # ax3.fill_between(two_point_times[sublist_ind], \
    #                  linear_estimates[sublist_ind]+linear_errors[sublist_ind], \
    #                  linear_estimates[sublist_ind]-linear_errors[sublist_ind], \
    #                  color=colors[sublist_ind], alpha=0.3)

    # lab1 = 'Meas. {:d}'.format(sublist_ind)
    lab1 = '$\\tau_{{{:d}}}$'.format(sublist_ind)

    if both_two_point:
        ax3.plot(two_point_times[sublist_ind], two_point_estimates_2[sublist_ind], \
                    color=colors[sublist_ind], ls='--', alpha=0.2)
        ax3.fill_between(two_point_times[sublist_ind], \
                         two_point_estimates_2[sublist_ind]+two_point_errors_2[sublist_ind], \
                         two_point_estimates_2[sublist_ind]-two_point_errors_2[sublist_ind], \
                         color=colors[sublist_ind], hatch='\\', alpha=0.05)

    ax3.plot(two_point_times[sublist_ind], two_point_estimates[sublist_ind], \
                color=colors[sublist_ind], ls='-', label=lab1)
    ax3.fill_between(two_point_times[sublist_ind], \
                     two_point_estimates[sublist_ind]+two_point_errors[sublist_ind], \
                     two_point_estimates[sublist_ind]-two_point_errors[sublist_ind], \
                     color=colors[sublist_ind], hatch='', alpha=0.3)

#ax3.set_title('Two-point Evaluations')
#ax3.set_title('Linear Fits')
ax3.set_xlabel('Median time $\\langle t_i \\rangle$ [s]')
ax3.set_ylabel('Extracted $\\tau$ [s]')
ax3.legend(fontsize=10, ncol=2)
#ax3.set_ylim(1700,2100)
ax3.set_ylim(tau_ylim[0], tau_ylim[1])

# buf3 = io.BytesIO()
# pickle.dump(fig3, buf3)
# buf3.seek(0)
# fig3z = pickle.load(buf3)
# fig3z.canvas.draw()
# tlt.get_renderer(fig3z)
# ax3z = fig3z.axes[0]

ax3.set_xlim(0,two_point_end_time)
fig3.tight_layout()

# ax3z.set_xlim(0,500)
# fig3z.tight_layout()

if savefig:
    if both_two_point:
        fig3.savefig(fig_base + 'spindown-time_two-point{:s}_both.svg'.format(suffix))
        # fig3z.savefig(fig_base + 'spindown-time_two-point{:s}_zoom_both.svg'.format(suffix))
    else:
        fig3.savefig(fig_base + 'spindown-time_two-point{:s}.svg'.format(suffix))
        # fig3z.savefig(fig_base + 'spindown-time_two-point{:s}_zoom.svg'.format(suffix))

# plt.show()

# print(two_point_estimates)
# for i in range(len(two_point_estimates)):
#     print(np.mean(two_point_estimates[i]))




all_fits = np.array(all_fits)


#plot_end_ind = np.argmin(np.abs(all_time_flat - fit_end_time))
plot_end_ind = np.argmin(np.abs(all_time_flat - 3000))
lower_yval = np.min([0.975 * all_freq_flat[plot_end_ind] * 1e-3, \
                     all_freq_flat[plot_end_ind] * 1e-3 - 2.5])

axarr[0].set_xlim(0,3000)
#axarr[0].set_xlim(0,fit_end_time)
#axarr[0].set_ylim(30, 52)
#axarr[0].set_ylim(72, 112)
axarr[0].set_ylim(lower_yval, 112)
#axarr[0].set_ylim(102, 112)
axarr[1].set_ylim(-2000, 2000)
#axarr[1].set_ylim(-12, 12)
axarr[1].set_xlabel('Time [s]')
axarr[0].set_ylabel('Rot. Freq. [kHz]')
axarr[1].set_ylabel('Resid. [Hz]')
if not sim_data:
    axarr[0].legend(loc='upper right', fontsize=10)
fig.tight_layout()

ax2.set_title('Successive Exp. Fits - $f_{term}$ constrained')
ax2.set_xlabel('Length of data used in exponential fit [s]')
ax2.set_ylabel('Extracted $\\tau$ [s]')

ax2.set_ylim(tau_ylim[0], tau_ylim[1])
if not sim_data:
    ax2.legend(fontsize=10, loc='upper right')
# axarr2[1].set_xlabel('Time [s]')
# axarr2[1].set_ylabel('$\\Delta \\tau$ [%]')
# axarr2[0].set_ylabel('$\\tau$ [s]')
# axarr2[0].legend(fontsize=10)

# buf2 = io.BytesIO()
# pickle.dump(fig2, buf2)
# buf2.seek(0)
# fig2z = pickle.load(buf2)
# fig2z.canvas.draw()
# tlt.get_renderer(fig2z)
# ax2z = fig2z.axes[0]

ax2.set_xlim(0,3000)
fig2.tight_layout()

# ax2z.set_xlim(0,500)
# fig2z.tight_layout()

if savefig:
    # fig.savefig('/home/charles/plots/20190626/spindowns.png')
    # fig.savefig('/home/charles/plots/20190626/spindowns.svg')
    fig.savefig(fig_base + 'spindowns{:s}.svg'.format(suffix))

    # fig2.savefig('/home/charles/plots/20190626/spindown_time.png')
    # fig2.savefig('/home/charles/plots/20190626/spindown_time.svg')
    fig2.savefig(fig_base + 'spindown_time{:s}.svg'.format(suffix))
    # fig2z.savefig(fig_base + 'spindown_time{:s}_zoom.svg'.format(suffix))




if sim_data:

    def fit_fun(x, A, mu, sigma):
        return gauss(x, A, mu, sigma, 0)

    fig3, ax3 = plt.subplots(1,1,figsize=(6,4),dpi=200)
    vals, bin_edge, _ = ax3.hist(all_fits[:,0])
    #vals = vals / np.max(vals)
    bins = bin_edge[:-1] + 0.5*(bin_edge[1] - bin_edge[0])

    p0 = [10.0, np.mean(all_fits[:,0]), 1.0]
    popt, pcov = opti.curve_fit(fit_fun, bins, vals, p0=p0, maxfev=10000)

    label = '$\\tau = ({:0.1f} \\pm {:0.1f})~$s'.format(popt[1],popt[2])

    plot_x = np.linspace(bins[0], bins[-1], 500)
    ax3.plot(plot_x, fit_fun(plot_x, *popt), ls='--', lw=2, color='r', label=label)
    ax3.set_xlabel('Extracted Damping Time [s]')
    ax3.set_ylabel('Count [arb. units]')
    ax3.set_ylim(0,12)
    ax3.legend(loc='upper right')
    fig3.tight_layout()

    fig3.savefig(sim_fig_base + 'tau_histogram{:s}.png'.format(suffix))
    fig3.savefig(sim_fig_base + 'tau_histogram{:s}.svg'.format(suffix))


    fig4, ax4 = plt.subplots(1,1,figsize=(6,4),dpi=200)
    vals2, bin_edge2, _ = ax4.hist(all_fits[:,2])
    #vals = vals / np.max(vals)
    bins2 = bin_edge2[:-1] + 0.5*(bin_edge2[1] - bin_edge2[0])


    p0 = [10, np.mean(all_fits[:,2]), 1e-7]
    popt2, pcov2 = opti.curve_fit(fit_fun, bins2, vals2, p0=p0, maxfev=10000)

    p_sv = bu.get_scivals(popt2[1])
    pe_sv = bu.get_scivals(np.abs(popt2[2]))

    err_exp_diff = p_sv[1] - pe_sv[1]
    pe_sv = (pe_sv[0] / (10.0**err_exp_diff), p_sv[1])

    label2 = '$ P = ({0} \\pm {1})$'.format('{:0.3f}'.format(p_sv[0]), '{:0.3f}'.format(pe_sv[0]) ) \
                + '$ \\times 10^{{{0}}}$ mbar'.format('{:d}'.format(p_sv[1])) 
    label3 = 'Expected: $3.5\\times10^{-6}~$ mbar'

    plot_x = np.linspace(bins2[0], bins2[-1], 500)
    ax4.plot(plot_x, fit_fun(plot_x, *popt2), ls='--', lw=2, color='r', label=label2)
    ax4.scatter([], [], color='w', label=label3)
    ax4.set_xlabel('Calculated Pressure [mbar]')
    ax4.set_ylabel('Count [arb. units]')
    ax4.set_xlim(3.48e-6, 3.52e-6)
    ax4.set_ylim(0,14)
    #ax4.set_xticks([3.49e-6, 3.495e-6, 3.50e-6, 3.505e-6, 3.51e-6])
    ax4.set_xticks([3.48e-6, 3.49e-6, 3.50e-6, 3.51e-6, 3.52e-6])
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax4.legend(loc='upper right')
    fig4.tight_layout()

    fig4.savefig(sim_fig_base + 'pressure_histogram{:s}.png'.format(suffix))
    fig4.savefig(sim_fig_base + 'pressure_histogram{:s}.svg'.format(suffix))

plt.show()