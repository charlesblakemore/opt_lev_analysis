import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
import scipy.optimize as opti
import scipy.interpolate as interp
import scipy.signal as signal
import matplotlib
import dill as pickle
import bead_util as bu

plt.rcParams.update({'font.size': 14})

#base_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/pramp_data/" 
base_path = '/processed_data/spinning/pramp_data/20190514'

plot_each_gas = True
plot_raw_data = True
plot_pressures = True

nbins_user = 200

m_He = 4.022
m_N2 = 28.014
m_Ar = 39.948
m_Kr = 83.798
m_Xe = 131.29
m_SF6 = 146.06

gases = {'He': [['50kHz_4Vpp_3_1', '50kHz_4Vpp_3_2', '50kHz_4Vpp_3_3'], False], \
         #'N2': [['50kHz_4Vpp_5', '50kHz_4Vpp_6', '50kHz_4Vpp_7'], True], \
         #'Ar': [['50kHz_4Vpp_2_1', '50kHz_4Vpp_2_2', '50kHz_4Vpp_2_3', '50kHz_4Vpp_2_4'], True], \
         #'Kr': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
         #'Xe': [['50kHz_4Vpp_2', '50kHz_4Vpp_3', '50kHz_4Vpp_4'], True], \
         #'SF6': [['50kHz_4Vpp_2', '50kHz_4Vpp_3', '50kHz_4Vpp_4'], True], \
        }

outdat = {'He': {'data': [], 'pmax': [], 'mass': m_He}, \
          'N2': {'data': [], 'pmax': [], 'mass': m_N2}, \
          'Ar': {'data': [], 'pmax': [], 'mass': m_Ar}, \
          'Kr': {'data': [], 'pmax': [], 'mass': m_Kr}, \
          'Xe': {'data': [], 'pmax': [], 'mass': m_Xe}, \
          'SF6': {'data': [], 'pmax': [], 'mass': m_SF6}}


def get_delta_phi(fname):
    delta_phi = np.load(fname + "_phi.npy")
    return delta_phi

def get_pressure(fname):
    pressures = np.load(fname + "_pressures.npy")
    return pressures


def phi_ffun(p, k, phi0):
    # negative arcsin function for phase lag fit. It's parameterized
    # such that param[0] (popt[0] from curve_fit) is basically pmax
    return -1.*np.arcsin(p/k) + phi0



def build_full_pressure(pressures, pirani_ind=0, highp_baratron_ind=1, \
                          baratron_ind=2, bara_lim=0.015, pirani_lim=5.0e-4, \
                          highp_bara_lim=0.001, plot=False, use_highp_bara=False):
    # Function to take the data from all three baratrons and combine
    # it sensibly. Since the baratrons have overlap and should both be
    # gas species independent, we use them as a reference. The low pressure
    # baratron is always used. Sometimes, high pressure data is taken from 
    # the pirani and sometimes from the other baratron.

    inds = np.array(range(len(pressures[:,0])))

    pirani_p = pressures[:,pirani_ind]
    bara_p = pressures[:,baratron_ind]
    bara_p2 = pressures[:,highp_baratron_ind]

    low_p = bara_p
    if use_highp_bara:
        high_p = bara_p2
    else:
        high_p = pirani_p

    low_p_good = low_p < bara_lim
    if use_highp_bara:
        high_p_good = high_p > highp_bara_lim
    else:
        high_p_good = high_p > pirani_lim

    overlap = low_p_good * high_p_good

    high_p_bad = np.invert(high_p_good)
    low_p_bad = np.invert(low_p_good)

    if use_highp_bara:
        overlap_avg = 0.5 * (low_p[overlap] + high_p[overlap])
        fac1 = np.mean(overlap_avg / low_p[overlap])
        low_p = low_p * fac1
        fac2 = np.mean(overlap_avg / high_p[overlap])
        high_p = high_p * fac2

    else:
        fac = np.mean(low_p[overlap] / high_p[overlap])
        high_p = high_p * fac

    low_p_only = low_p[high_p_bad]
    high_p_only = high_p[low_p_bad]

    avg_p_only = 0.5 * (low_p[overlap] + high_p[overlap])
    total_p = np.concatenate((low_p_only, avg_p_only, high_p_only))

    pres_func = interp.interp1d(inds, total_p, kind='quadratic')

    #pres_func_2 = interp.splrep(inds, total_p, s=5e-4)
    pres_func_2 = interp.splrep(inds, total_p, s=12e-4)
    pres_2 = interp.splev(inds, pres_func_2, der=0)

    if plot:
        plt.plot(inds, pirani_p, label='Pirani')
        plt.plot(inds, bara_p, label='0.02 Baratron')
        plt.plot(inds, bara_p2, label='0.1 Baratron')
        plt.plot(inds, pres_2)
        plt.legend()
        plt.show() 

    #return pres_func_2(inds)
    return pres_func(inds), pres_2




def analyze_file(fname, nbins=500, grad_thresh=10, use_highp_bara=True, plot_pressures=False, \
                 plot_raw_data=False):

    phases = get_delta_phi(fname)
    pressures = get_pressure(fname)

    pressures_real, pressures_smooth = build_full_pressure(pressures, plot=plot_pressures, \
                                                           use_highp_bara=use_highp_bara)

    # Compute the initial phase for offsetting so arcsin(phi0) = 0
    phi0 = np.mean(phases[:5])

    # Find where we lose lock by looking for sharp derivative
    raw_grad = np.gradient(np.unwrap(2.0 * phases))

    raw_grad_init = np.std(raw_grad[:int(0.01*len(raw_grad))])
    bad_inds = np.array(range(len(raw_grad)))[np.abs(raw_grad) > grad_thresh * raw_grad_init]
    
    lock_lost_ind = -1
    # Make sure we didn't just find an anomolous fluctuation
    for indind, ind in enumerate(bad_inds):
        if ind == bad_inds[-2]:
            lock_lost_ind = -1
            break
        delta = np.abs(ind - bad_inds[indind+1])
        if delta < 10:
            delta2 = np.abs(bad_inds[indind+1] - bad_inds[indind+2])
            if delta2 < 10:
                lock_lost_ind = ind
                break

    # Reconstruct phase difference of fundamental rotation by 
    # unwrapping data prior to losing lock, then using the raw
    # data after losing lock
    uphases = np.unwrap(2.0*phases) / 2.0

    init_offset = np.mean(uphases[:10])
    uphases -= init_offset

    uphases[lock_lost_ind:] = phases[lock_lost_ind:]

    sort_inds = np.argsort(pressures_real)

    lock_lost_ind_sort = sort_inds[lock_lost_ind]

    pressures_real_sorted = pressures_real[sort_inds]
    uphases_sorted = uphases[sort_inds]

    if plot_raw_data:
        plt.scatter(pressures_real_sorted, uphases_sorted, s=100)
        plt.axvline(pressures_real_sorted[lock_lost_ind_sort])
        plt.show()

    fit_pressures = pressures_real_sorted[:lock_lost_ind_sort-2]
    fit_uphases = uphases_sorted[:lock_lost_ind_sort-2]

    fit_pressures_2, fit_uphases_2, fit_errs = bu.rebin(fit_pressures, fit_uphases, nbins=50)
    
    zero_inds = np.where(fit_errs == 0.0)
    non_zero_inds = np.invert(zero_inds)
    fit_errs[zero_inds] += np.mean( fit_errs[non_zero_inds] )

    p0 = [1.1*pressures_real_sorted[lock_lost_ind_sort-2], 0]

    if np.sum(np.isnan(fit_uphases_2)):
        pphi, covphi = curve_fit(phi_ffun, fit_pressures, fit_uphases, p0 = p0, \
                                 bounds=([0.005, -0.5], [1.5*pressures_real_sorted[lock_lost_ind_sort-2], 0.5]), \
                                 maxfev=10000)

    else:
        #print p0
        #plt.plot(fit_pressures_2, fit_uphases_2)
        #plt.show()
        #print fit_errs
        #print phi_ffun(fit_pressures_2, *p0)
        pphi, covphi = curve_fit(phi_ffun, fit_pressures_2, fit_uphases_2, sigma=fit_errs, p0 = p0, \
                                 bounds=([0.005, -0.5], [1.5*pressures_real_sorted[lock_lost_ind_sort-2], 0.5]), \
                                 absolute_sigma=False, maxfev=10000)

    uphases_sorted -= pphi[1]

    cut_inds = pressures_real_sorted < 1.2*pphi[0]

    pressures_cut = pressures_real_sorted[cut_inds]
    uphases_cut = uphases_sorted[cut_inds]

    rand_phase_ind = np.argmin( np.abs(pressures_cut - pphi[0]) )
    #print rand_phase_ind
    #print filname
    #plt.plot(pressures_cut, uphases_cut)
    #plt.show()

    pressures_out_1, uphases_out_1, errs_out_1 = bu.rebin(pressures_cut[:rand_phase_ind], \
                                                          uphases_cut[:rand_phase_ind], \
                                                          nbins=nbins)

    if rand_phase_ind > 2.0*nbins:
        pressures_out_2 = signal.decimate(pressures_cut[rand_phase_ind:], \
                                          int(rand_phase_ind / nbins))
        uphases_out_2 = signal.decimate(uphases_cut[rand_phase_ind:], \
                                        int(rand_phase_ind / nbins))
    else:
        pressures_out_2 = pressures_cut[rand_phase_ind:]
        uphases_out_2 = pressures_cut[rand_phase_ind:]
    
    return np.array([np.concatenate((pressures_out_1, pressures_out_2)), \
                     np.concatenate((uphases_out_1, uphases_out_2))]), \
                     pphi[0]


'''
gas_keys = gases.keys()
for gas in gas_keys:
    fils = gases[gas][0]
    use_highp_bara = gases[gas][1]

    maxp = 0

    for fil in fils:
        filname = base_path + '/' + gas + '/' + fil
        dat, pmax = analyze_file(filname, nbins=nbins_user, use_highp_bara=use_highp_bara, \
                                 grad_thresh=5, plot_raw_data=plot_raw_data, \
                                 plot_pressures=plot_pressures)

        outdat[gas]['data'].append(dat)
        outdat[gas]['pmax'].append(pmax)

        cmax = np.max(dat[0])
        if cmax > maxp:
            maxp = cmax


    if plot_each_gas:
        line_p = np.linspace(-0.5 * pmax, 1.5 * pmax, 100) 
        min_line = np.ones_like(line_p) * (-0.5)
        plt.plot(line_p, min_line, '--', color='k', lw=2, alpha=0.6)

        for filind, fil in enumerate(fils):
            color = 'C' + str(filind)

            dat = outdat[gas]['data'][filind]
            pmax = outdat[gas]['pmax'][filind]

            lab_str = '$p_{\mathrm{max}}$ = %0.3f' % pmax

            fitp = np.linspace(0, np.max(dat[0]), 100)
            fit = np.array(phi_ffun(fitp, pmax, 0))
            plt.scatter(dat[0], dat[1] / np.pi, edgecolors=color, facecolors='none', alpha=0.5)
            plt.plot(fitp, fit / np.pi, '-', color=color, lw=3, label=lab_str)

        mean_pmax = np.mean(outdat[gas]['pmax'])
        err_pmax = np.std(outdat[gas]['pmax'])
        title_str = gas + (': $<p_{\mathrm{max}}> = %0.3f \pm %0.3f$' % (mean_pmax, err_pmax))

        plt.xlim(-0.05*maxp, 1.05*maxp)
        plt.xlabel('Pressure [mbar]')
        plt.ylabel('Phase offset [$\pi$ rad]')
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.suptitle(title_str, fontsize=16)
        plt.subplots_adjust(top=0.91)
        plt.show()

pickle.dump(outdat, open(base_path + '/all_data.p', 'wb'))



def proportional(x, a, b):
    return a * x + b

def inverse_sqrt(x, a, b):
    return a * (1.0 / np.sqrt(x)) + b

pmax_He = np.mean(outdat['He']['pmax'])

pmax_vec = []
pmax_err_vec = []

mass_vec = []

pmax_ratios = []
mass_ratios = []
for gas in gas_keys:
    
    #if gas == 'N2':
    #    continue

    pmax_vec.append( np.mean(outdat[gas]['pmax']) )
    pmax_err_vec.append( np.std(outdat[gas]['pmax']) )

    mass_vec.append( outdat[gas]['mass'] )

    pmax_ratios.append( np.mean(outdat[gas]['pmax']) / pmax_He )
    mass_ratios.append( np.sqrt(m_He / outdat[gas]['mass']) )

popt, pcov = opti.curve_fit(proportional, mass_ratios, pmax_ratios, p0=[1, 0])

print 'Proportional offset: %0.4f' % popt[1]

xplot = np.linspace(0, 1.1*np.max(mass_ratios), 100)
yplot = proportional(xplot, *popt)

plt.scatter(mass_ratios, pmax_ratios - popt[1], s=50)
plt.plot(xplot, yplot, '--', color='r', lw=4)
plt.xlim(0, 1.1*np.max(mass_ratios))
plt.xlabel('$\sqrt{m_{\mathrm{He}} / m_0}$')
plt.ylim(0, 1.1*np.max(pmax_ratios))
plt.ylabel('$p_{\mathrm{max}} / p_{\mathrm{max},He}$')
plt.tight_layout()


popt2, pcov2 = opti.curve_fit(inverse_sqrt, mass_vec, pmax_vec, [0.1, 0])

print 'Inverse sqrt offset: %0.4f' % popt2[1]

ann_str = 'Systematic offset of %0.4f mbar removed' % popt2[1]

xplot2 = np.linspace(0, 1.1*np.max(mass_vec), 100)
xplot2[0] += 1e-9
yplot2 = inverse_sqrt(xplot2, popt2[0], 0)
#yplot3 = inverse_sqrt(mass_vec, 0.1)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.errorbar(mass_vec, pmax_vec - popt2[1], yerr=pmax_err_vec, fmt='o', ms=5)
ax.plot(xplot2, yplot2, '--', color='r', lw=4, alpha=0.5)
ax.text(0.98, 0.9, ann_str, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

ax.set_xlim(0, 1.1*np.max(mass_vec))
ax.set_xlabel('$m_0$ [amu]')
ax.set_ylim(0, 1.1*np.max(pmax_vec))
ax.set_ylabel('$p_{\mathrm{max}}$ [mbar]')
plt.tight_layout()



plt.show()
    
'''










