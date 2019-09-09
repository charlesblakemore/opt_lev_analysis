import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
import scipy.optimize as opti
import scipy.interpolate as interp
import scipy.signal as signal
import scipy.stats as stats
import matplotlib
import dill as pickle
import bead_util as bu

plt.rcParams.update({'font.size': 14})

#base_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/pramp_data/" 
base_path = '/processed_data/spinning/pramp_data/20190626'
base_dipole_path = '/processed_data/spinning/wobble/20190626'

base_plot_path = '/home/charles/plots/20190626/pramp'

kb = 1.3806e-23
T = 297

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

print Ibead
print Ibead_err

dipole_units = 1.6e-19 * 1e-6

plot_each_gas = False
plot_raw_data = False
plot_pressures = False

nbins_user = 200
grad_thresh = 10

gases = {'He': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
         'N2': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
         'Ar': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
         'Kr': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3', ], True], \
         'Xe': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
         'SF6': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
        }



keys = ['data', 'pmax', 'pmax_err', 'p_outgassing', 'dipole', 'dipole_err', \
        'mass', 'rot_freq', 'rot_freq_err', 'rot_amp', 'rot_amp_err']

outdat = {}
for gas in gases.keys():
    outdat[gas] = {}
    for key in keys:
        outdat[gas][key] = []



outgassing_dir = '/processed_data/spinning/pramp_data/20190626/outgassing/'
files, lengths = bu.find_all_fnames(outgassing_dir, ext='.txt')
rates = []
for filename in files:
    file_obj = open(filename, 'rb')
    lines = file_obj.readlines()
    file_obj.close()
    rate = float(lines[2])
    rates.append(rate)
outgassing_rate = np.mean(rates)


def get_delta_phi(fname):
    delta_phi = np.load(fname + "_phi.npy")
    return delta_phi

def get_delta_phi_err(fname):
    delta_phi_err = np.load(fname + "_phi_err.npy")
    return delta_phi_err

def get_pressure(fname):
    pressures = np.load(fname + "_pressures_mbar.npy")
    return pressures

def get_time(fname):
    time = np.load(fname + "_time.npy")
    return time

def get_field_data(fname):
    field_amp = np.load(fname + '_field_amp.npy')
    field_amp_err = np.load(fname + '_field_amp_err.npy')
    field_freq = np.load(fname + '_field_freq.npy')
    field_freq_err = np.load(fname + '_field_freq_err.npy')
    return {'field_amp': field_amp, 'field_amp_err': field_amp_err, \
            'field_freq': field_freq, 'field_freq_err': field_freq_err}

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
                 plot_raw_data=False, find_dipole=True):

    phases = get_delta_phi(fname)
    phase_errs = get_delta_phi_err(fname)
    pressures = get_pressure(fname)
    times = get_time(fname)
    field_data = get_field_data(fname)

    rot_freq = np.mean(field_data['field_freq'])
    rot_freq_err = np.std(field_data['field_freq'])
    rot_amp = np.mean(field_data['field_amp'])
    rot_amp_err = np.std(field_data['field_amp'])

    plt.figure()
    plt.plot(field_data['field_amp'])
    plt.figure()
    plt.plot(field_data['field_freq'])
    plt.show()

    pressures_real, pressures_smooth = build_full_pressure(pressures, plot=plot_pressures, \
                                                           use_highp_bara=use_highp_bara)

    sort_inds = np.argsort(pressures_real)

    phases = phases[sort_inds]
    phase_errs = phase_errs[sort_inds]
    pressures_real = pressures_real[sort_inds]
    #plt.errorbar(pressures_real, phases, yerr=phase_errs, ms=2)
    #plt.show()

    # Compute the initial phase for offsetting so arcsin(phi0) = 0
    phi0 = np.mean(phases[:5])

    # Find where we lose lock by looking for sharp derivative
    raw_grad = np.gradient(np.unwrap(2.0 * phases))
    #plt.show()

    init_ind = int(np.max([10.0, 0.01*len(raw_grad)]))

    raw_grad_init = np.std(raw_grad[:init_ind])
    raw_grad -= np.mean(raw_grad[:init_ind])

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

    p_outgassing = (times[lock_lost_ind] - times[0]) * 1e-9 * outgassing_rate

    # Reconstruct phase difference of fundamental rotation by 
    # unwrapping data prior to losing lock, then using the raw
    # data after losing lock
    uphases = np.unwrap(2.0*phases) / 2.0

    init_offset = np.mean(uphases[:10])
    uphases -= init_offset

    uphases[lock_lost_ind:] = phases[lock_lost_ind:]

    if plot_raw_data:
        plt.scatter(pressures_real, uphases, s=100)
        plt.axvline(pressures_real[lock_lost_ind])
        plt.show()

    fit_pressures = pressures_real[:lock_lost_ind-2]
    fit_uphases = uphases[:lock_lost_ind-2]
    fit_errs = phase_errs[:lock_lost_ind-2]

    fit_pressures_2, fit_uphases_2, fit_errs_2 = \
            bu.rebin(fit_pressures, fit_uphases, errs=fit_errs, nbins=50)

    #plt.errorbar(pressures_real, uphases, yerr=phase_errs)
    #plt.errorbar(fit_pressures_2, fit_uphases_2, yerr=fit_errs_2)
    #plt.axvline(pressures_real[lock_lost_ind-2])
    #plt.show()
    
    zero_inds = np.where(fit_errs_2 == 0.0)
    non_zero_inds = np.invert(zero_inds)
    fit_errs_2[zero_inds] += np.sqrt(np.mean( fit_errs_2[non_zero_inds]**2 ))

    p0 = [1.1*pressures_real[lock_lost_ind-2], 0]

    if not np.sum(np.isnan(fit_uphases_2)):
        fit_pressures = fit_pressures_2
        fit_uphases = fit_uphases_2
        fit_errs = fit_errs_2

    pphi, covphi = curve_fit(phi_ffun, fit_pressures, fit_uphases, sigma=fit_errs, p0 = p0, \
                             bounds=([0.005, -0.5], [1.5*pressures_real[lock_lost_ind-2], 0.5]), \
                             maxfev=10000)

    param_arr = np.linspace(pphi[0]*0.99, pphi[0]*1.01, 200)
    def nll(param):
        inds = fit_pressures < param
        resid = np.abs(fit_uphases[inds]-pphi[1] - phi_ffun(fit_pressures[inds], param, 0))
        return (1. / (np.sum(inds) - 1)) * np.sum(resid**2 / fit_errs[inds]**2)

    pmax, pmax_err, min_chi = bu.minimize_nll(nll, param_arr, plot=False)

    uphases -= pphi[1]

    cut_inds = pressures_real< 1.2*pmax

    pressures_cut = pressures_real[cut_inds]
    uphases_cut = uphases[cut_inds]

    rand_phase_ind = np.argmin( np.abs(pressures_cut - pphi[0]) )
    #print rand_phase_ind
    #print filname
    #plt.plot(pressures_cut, uphases_cut)
    #plt.show()

    pressures_out_1, uphases_out_1, errs_out_1 = bu.rebin(pressures_cut[:rand_phase_ind], \
                                                          uphases_cut[:rand_phase_ind], \
                                                          nbins=nbins)

    pressures_out_2 = pressures_cut[rand_phase_ind:]
    uphases_out_2 = uphases_cut[rand_phase_ind:]
    
    print 'outgassing ratio: ', p_outgassing / pmax

    return np.array([np.concatenate((pressures_out_1, pressures_out_2)), \
                     np.concatenate((uphases_out_1, uphases_out_2))]), \
                     pmax, pmax_err, p_outgassing, rot_freq, rot_freq_err, \
                     rot_amp, rot_amp_err



gas_keys = gases.keys()
for gas in gas_keys:
    fils = gases[gas][0]
    use_highp_bara = gases[gas][1]

    maxp = 0

    for filind, fil in enumerate(fils):
        filname = base_path + '/' + gas + '/' + fil
        dat, pmax, pmax_err, p_outgassing, rot_freq, rot_freq_err, rot_amp, rot_amp_err = \
                analyze_file(filname, nbins=nbins_user, use_highp_bara=use_highp_bara, \
                             grad_thresh=grad_thresh, plot_raw_data=plot_raw_data, \
                             plot_pressures=plot_pressures)

        mass_filename = base_path + '/' + gas + '/rga-m0_%i.mass' % (filind + 1)
        mass_arr = np.load(open(mass_filename, 'rb'))

        dipole_filename = base_dipole_path + '/' + gas + '_pramp_' + str(filind+1) + '.dipole'
        dipole_dat = np.load(open(dipole_filename, 'rb'))
        dipole = dipole_dat[0] / dipole_units
        dipole_err = dipole_dat[1] / dipole_units

        # Include 1% systematic from field amplitude uncertainty
        pmax_err = np.sqrt(pmax_err**2 + (0.01*pmax)**2)

        outdat[gas]['data'].append(dat)
        outdat[gas]['pmax'].append(pmax)
        outdat[gas]['pmax_err'].append(pmax_err)
        outdat[gas]['p_outgassing'].append(p_outgassing)

        outdat[gas]['dipole'].append(dipole)
        outdat[gas]['dipole_err'].append(dipole_err)
        outdat[gas]['mass'].append(mass_arr)

        outdat[gas]['rot_freq'].append(rot_freq)
        outdat[gas]['rot_freq_err'].append(rot_freq_err)
        outdat[gas]['rot_amp'].append(rot_amp)
        outdat[gas]['rot_amp_err'].append(rot_amp_err)

        cmax = np.max(dat[0])
        if cmax > maxp:
            maxp = cmax


    if plot_each_gas:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line_p = np.linspace(-0.5 * pmax, 1.5 * pmax, 100) 
        min_line = np.ones_like(line_p) * (-0.5)
        ax.plot(line_p, min_line, '--', color='k', lw=2, alpha=0.6)

        for filind, fil in enumerate(fils):
            color = 'C' + str(filind)

            dat = outdat[gas]['data'][filind]
            pmax = outdat[gas]['pmax'][filind]

            lab_str = '$p_{\mathrm{max}}$ = %0.3f' % pmax

            fitp = np.linspace(0, np.max(dat[0]), 100)
            fit = np.array(phi_ffun(fitp, pmax, 0))
            ax.scatter(dat[0], dat[1] / np.pi, edgecolors=color, facecolors='none', alpha=0.5)
            ax.plot(fitp, fit / np.pi, '-', color=color, lw=3, label=lab_str)

        mean_pmax = np.mean(outdat[gas]['pmax'])
        err_pmax = np.std(outdat[gas]['pmax'])
        title_str = gas + (': $<p_{\mathrm{max}}> = %0.3f \pm %0.3f$' % (mean_pmax, err_pmax))

        ax.set_xlim(-0.05*maxp, 1.05*maxp)
        ax.set_xlabel('Pressure [mbar]')
        ax.set_ylabel('Phase offset [$\pi$ rad]')
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.suptitle(title_str, fontsize=16)
        fig.subplots_adjust(top=0.91)

        fig_path = base_plot_path + ('/%s_pramp.png' % gas)
        bu.make_all_pardirs(fig_path)
        plt.savefig(fig_path)
        plt.show()



pickle.dump(outdat, open(base_path + '/all_data.p', 'wb'))



def proportional(x, a, b):
    return a * x + b

def inverse_sqrt(x, a, b):
    return a * (1.0 / np.sqrt(x)) #+ b

pmax_He = np.mean(outdat['He']['pmax'])

pmax_vec = []
pmax_err_vec = []

pmax_norm_vec = []
pmax_norm_err_vec = []

raw_mass_vec = []
mass_vec = []
mass_err_vec = []
mass_vec_2 = []
mass_err_vec_2 = []
mass_vec_3 = []
mass_err_vec_3 = []

rot_freq_vec = []
rot_freq_err_vec = []
rot_amp_vec = []
rot_amp_err_vec = []
for gas in gas_keys:
    
    #if gas == 'N2':
    #    continue

    rot_freq_vec.append( np.mean(outdat[gas]['rot_freq']) )
    rot_freq_err_vec.append( np.sqrt( 1.0 / (len(outdat[gas]['rot_freq']) - 1)  * \
                            np.sum(np.array(outdat[gas]['rot_freq_err'])**2) ) )
    rot_amp_vec.append( np.mean(outdat[gas]['rot_amp']) )
    rot_amp_err_vec.append( np.sqrt( 1.0 / (len(outdat[gas]['rot_amp']) - 1)  * \
                            np.sum(np.array(outdat[gas]['rot_amp_err'])**2) ) )

    gas_pmax_vec = np.array(outdat[gas]['pmax'])
    gas_pmax_err_vec = np.array(outdat[gas]['pmax_err'])
    gas_dipole_vec = np.array(outdat[gas]['dipole'])
    gas_dipole_err_vec = np.array(outdat[gas]['dipole_err'])

    gas_norm_vec = gas_pmax_vec / gas_dipole_vec

    pmax_vec.append( np.mean(outdat[gas]['pmax']) )
    pmax_err_vec.append( np.std(outdat[gas]['pmax']) )

    pmax_norm_vec.append( np.mean( gas_norm_vec ) )

    gas_norm_err_vec = gas_norm_vec * np.sqrt((gas_dipole_err_vec / gas_dipole_vec)**2 + \
                                                (gas_pmax_err_vec / gas_pmax_vec)**2)
    err_val = np.sqrt( (1.0 / (len(gas_norm_vec)-1)) * np.sum(gas_norm_err_vec**2) ) 

    pmax_norm_err_vec.append( err_val )

    #raw_mass_vec.append()

    mass_arr = np.array(outdat[gas]['mass'])
    for ind in [1,3,5]:
        mass_arr[:,ind] = mass_arr[:,ind]**2
    mean_masses = np.mean(np.array(outdat[gas]['mass']), axis=0)
    mass_vec.append( mean_masses[2] )
    mass_err_vec.append( np.sqrt(mean_masses[3]) )

mass_vec = np.array(mass_vec)
mass_err_vec = np.array(mass_err_vec)



# popt, pcov = opti.curve_fit(proportional, mass_ratios, pmax_ratios, p0=[1, 0])

# print 'Proportional offset: %0.4f' % popt[1]

# xplot = np.linspace(0, 1.1*np.max(mass_ratios), 100)
# yplot = proportional(xplot, *popt)

# plt.scatter(mass_ratios, pmax_ratios - popt[1], s=50)
# plt.plot(xplot, yplot, '--', color='r', lw=4)
# plt.xlim(0, 1.1*np.max(mass_ratios))
# plt.xlabel('$\sqrt{m_{\mathrm{He}} / m_0}$')
# plt.ylim(0, 1.1*np.max(pmax_ratios))
# plt.ylabel('$p_{\mathrm{max}} / p_{\mathrm{max},He}$')
# plt.tight_layout()


# popt2, pcov2 = opti.curve_fit(inverse_sqrt, mass_vec, pmax_vec, [0.1, 0])

# ann_str = 'Systematic offset of %0.4f mbar removed' % popt2[1]

# xplot2 = np.linspace(0, 1.1*np.max(mass_vec), 100)
# xplot2[0] += 1e-9
# yplot2 = inverse_sqrt(xplot2, popt2[0], 0)
# #yplot3 = inverse_sqrt(mass_vec, 0.1)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.errorbar(mass_vec, pmax_vec - popt2[1], xerr=mass_err_vec, yerr=pmax_err_vec, fmt='o', ms=5)
# ax.plot(xplot2, yplot2, '--', color='r', lw=4, alpha=0.5)
# #ax.text(0.98, 0.9, ann_str, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

# ax.set_xlim(0, 1.1*np.max(mass_vec))
# ax.set_xlabel('$m_0$ [amu]')
# ax.set_ylim(0, 1.1*np.max(pmax_vec))
# ax.set_ylabel('$p_{\mathrm{max}}$ [mbar]')
# plt.tight_layout()




popt3, pcov3 = opti.curve_fit(inverse_sqrt, mass_vec, pmax_norm_vec, [0.1 / 100, 0])


pmax_norm_vec = np.array(pmax_norm_vec)
pmax_norm_err_vec = np.array(pmax_norm_err_vec)

mass_vec = np.array(mass_vec)
mass_err_vec = np.array(mass_err_vec)

def cost(param):
    resid = np.abs(inverse_sqrt(mass_vec, param, 0) - pmax_norm_vec)
    norm = 1. / (len(mass_vec) - 1)
    tot_var = pmax_norm_err_vec**2 + pmax_norm_vec**2 * (mass_err_vec / mass_vec)**2
    return norm * np.sum( resid**2 / tot_var)


# x0 = [popt3[0]]
# res = opti.minimize(cost, x0)

param_arr = np.linspace(0.98*popt3[0], 1.02*popt3[0], 200)
param, param_err, min_chi = bu.minimize_nll(cost, param_arr, plot=True)

# convert from units of mbar * amu^1/2 / (e * um) to Pa * kg^1/2 / (C * m)
conv_fac = (100.0) * np.sqrt(1.6605e-27) * (1.0 / dipole_units)
param_si = param * conv_fac
param_si_err = param_err * conv_fac

rot_freq = np.mean(rot_freq_vec)
rot_amp =  np.mean(rot_amp_vec)

kappa = param_si * 2.0 * np.pi * rot_freq / rot_amp

print mass_vec
print rot_freq_err_vec

print param_si_err / param_si
print np.median(rot_freq_err_vec) / rot_freq
print np.median(rot_amp_err_vec) / rot_amp

kappa_err = kappa * np.sqrt( (np.max(rot_freq_err_vec) / rot_freq)**2 + \
                             (np.max(rot_amp_err_vec) / rot_amp)**2 + \
                             (param_si_err / param_si)**2 )

r_drag = (1.0 / kappa)**(0.25) * ((27.0 * kb * T) / (32.0 * np.pi))**(0.5 * 0.25) 
r_drag_err = r_drag * np.sqrt( 0.25 * (kappa_err / kappa)**2 + 0.125 * (5.0 / T)**2 )

print 'Kappa: {:0.3g} +- {:0.3g}'.format(kappa, kappa_err)
print 'Rdrag: {:0.3g} +- {:0.3g}'.format(r_drag, r_drag_err)
print 'Rbead: {:0.3g} +- {:0.3g}'.format(rbead, rbead_err)

# ann_str = 'Systematic offset of %0.4f mbar removed' % popt3[1]

#xplot3 = np.linspace(0, 1.1*np.max(mass_vec), 100)
xplot3 = np.linspace(0, 150, 100)
xplot3[0] += 1e-9
#yplot3 = inverse_sqrt(xplot3, popt3[0], 0)
yplot3 = inverse_sqrt(xplot3, param, 0)

k_sv = bu.get_scivals(kappa)
ke_sv = bu.get_scivals(kappa_err)

r_sv = bu.get_scivals(r_drag)
re_sv = bu.get_scivals(r_drag_err)

err_exp_diff = k_sv[1] - ke_sv[1]
ke_sv = (ke_sv[0] / (10.0**err_exp_diff), k_sv[1])

err_exp_diff = r_sv[1] - re_sv[1]
re_sv = (re_sv[0] / (10.0**err_exp_diff), r_sv[1])

label = '$ \\kappa = ({0} \\pm {1})$'.format('{:0.1f}'.format(k_sv[0]), '{:0.1f}'.format(ke_sv[0]) ) \
            + '$ \\times 10^{{{0}}}$ '.format('{:d}'.format(k_sv[1])) \
            + '$\\mathrm{J}^{1/2} \\mathrm{m}^{-4}$'

label2 = 'min($ \\chi ^2 / N_{\\mathrm{DOF}}$)=' + '{0}'.format('{:0.2f}'.format(min_chi))

if r_sv[1] == -6:
    label3 = '$ \\rightarrow \\mathrm{r}_{\mathrm{drag}} = $' \
            + '${0} \\pm {1}$'.format('{:0.2f}'.format(r_sv[0]), '{:0.2f}'.format(re_sv[0])) \
            + ' $\\mu \\mathrm{m}$'
else:
    label3 = '$ \\rightarrow \\mathrm{r}_{\mathrm{drag}} = $' \
            + '$({0} \\pm {1})$'.format('{:0.2f}'.format(r_sv[0]), '{:0.2f}'.format(re_sv[0])) \
            + '$\\times 10^{{{0}}}$'.format('{:d}'.format(r_sv[1]+6)) \
            + ' $\\mu \\mathrm{m}$'


fig3, axarr3 = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3,1]}, \
                            sharex=True, figsize=(6,4),dpi=200)

axarr3[0].errorbar(mass_vec, pmax_norm_vec*1e3, xerr=mass_err_vec,\
                yerr=pmax_norm_err_vec*1e3, fmt='o', ms=4, color='k')
axarr3[0].plot(xplot3, yplot3*1e3, '--', color='r', lw=2, alpha=0.5, label=label)
axarr3[0].plot([1], [1], color='w', label=label2)
axarr3[0].plot([1], [1], color='w', label=' ')
axarr3[0].plot([1], [1], color='w', label=label3)

resid_vec = pmax_norm_vec-inverse_sqrt(mass_vec, param, 0)
# axarr3[1].errorbar(mass_vec, (resid_vec)/np.array(pmax_norm_err_vec), \
#                    xerr=mass_err_vec,yerr=np.ones_like(pmax_norm_err_vec), fmt='o', ms=5)
axarr3[1].errorbar(mass_vec, resid_vec*1e3, xerr=mass_err_vec, \
                       yerr=pmax_norm_err_vec*1e3, fmt='o', ms=4, color='k')
# axarr3[1].errorbar(mass_vec, 1e2*resid_vec/np.array(pmax_norm_vec), xerr=mass_err_vec, \
#                         yerr=1e2*np.array(pmax_norm_err_vec)/np.array(pmax_norm_vec), fmt='o', ms=4)
axarr3[1].plot(xplot3, np.zeros_like(xplot3), '--', color='r', alpha=0.5)
#ax3.text(0.98, 0.9, ann_str, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

#ax3.set_xlim(0, 1.1*np.max(mass_vec))
axarr3[0].set_xlim(0, 150)
axarr3[1].set_xlabel('$m_{0,\mathrm{eff}}$ [amu]')
axarr3[0].set_ylim(0, 1.1*np.max(pmax_norm_vec)*1e3)
#axarr3[0].set_ylabel(r'$p_{\mathrm{max}} / d$'+'\n[$10^{-3}$ mbar / ($e \cdot \mu m$)]')
axarr3[0].set_ylabel(r'$p_{\mathrm{max}} / d$'+'\n[$\mu$bar / ($e \cdot \mu m$)]')
#axarr3[1].set_ylabel('Resid. [$\sigma$]')
#axarr3[1].set_ylabel('Resid. [%]')
axarr3[1].set_ylabel('Resid.')
#axarr3[1].set_ylabel('Resid\n[mbar / ($e \cdot \mu m$)]')

#handles, labels = axarr3[0].get_legend_handles_labels()


axarr3[0].legend(loc='upper right', fontsize=10)
plt.tight_layout()

fig_path1 = base_plot_path + '/all_pramp_fit.png'
fig_path2 = base_plot_path + '/all_pramp_fit.pdf'
fig_path3 = base_plot_path + '/all_pramp_fit.svg'
bu.make_all_pardirs(fig_path1)
fig3.savefig(fig_path1)
fig3.savefig(fig_path2)
fig3.savefig(fig_path3)






plt.show()
    
    










