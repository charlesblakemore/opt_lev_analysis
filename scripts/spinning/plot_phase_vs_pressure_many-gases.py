import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
import scipy.optimize as opti
import scipy.interpolate as interp
import scipy.signal as signal
import scipy.stats as stats
import scipy.constants as constants
import matplotlib
import dill as pickle
import bead_util as bu
import rga_util as ru
from iminuit import Minuit, describe

plt.rcParams.update({'font.size': 14})

debug = False

#base_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/pramp_data/" 

# base_path = '/data/old_trap_processed/spinning/pramp_data/20190626'
# base_dipole_path = '/data/old_trap_processed/spinning/wobble/20190626/'
# mbead = 85.0e-15 # convert picograms to kg
# mbead_err = 1.6e-15
# ind_offset = 1

# base_plot_path = '/home/cblakemore/plots/20190626/pramp'


date = '20190626'
base_path = '/data/old_trap_processed/spinning/pramp_data/{:s}'.format(date)
base_dipole_path = '/data/old_trap_processed/spinning/wobble/{:s}/'.format(date)
mbead = bu.get_mbead(date)
ind_offset = 1
gases_to_consider = ['He', 'N2']
#ind_offset = 3

savefig = True
base_plot_path = '/home/cblakemore/plots/20191017/pramp/combined_reverse'.format(date)
#base_plot_path = '/home/cblakemore/plots/{:s}/pramp'.format(date)

# base_path = '/data/old_trap_processed/spinning/pramp_data/20190905'
# base_dipole_path = '/data/old_trap_processed/spinning/wobble/20190905/'
# mbead = 84.2e-15 # convert picograms to kg
# mbead_err = 1.5e-15
# ind_offset = 1
# #ind_offset = 3

# base_plot_path = '/home/cblakemore/plots/20190905/pramp'

colors = bu.get_color_map(3, cmap='plasma')


include_other_beads = True
other_paths = ['/data/old_trap_processed/spinning/pramp_data/20190905', \
               '/data/old_trap_processed/spinning/pramp_data/20191017']
other_markers = ['x', '+']
other_linestyles = [':', '-.']
other_linestyles = [(0, (1, 1)), (0, (3, 1, 1, 1))]
other_dipole_paths = ['/data/old_trap_processed/spinning/wobble/20190905/', \
                      '/data/old_trap_processed/spinning/wobble/20191017/']
other_masses = []
for path in other_paths:
    date_o = path.split('/')[-1]
    mbead_o = bu.get_mbead(date_o)
    other_masses.append(mbead_o)
other_gases = [['He', 'N2'], ['He', 'N2']]


kb = constants.Boltzmann
T = 297

rbead = bu.get_rbead(mbead)
kappa_calc = bu.get_kappa(mbead)

# print Ibead
# print Ibead_err

dipole_units = constants.e * 1e-6

plot_each_gas = False
plot_raw_data = False
plot_pressures = False

make_example_pramp = True
example_figname = '/home/cblakemore/plots/20190626/pramp/example_pramp.svg'
example_gases = ['He', 'Ar', 'SF6']

nbins_user = 200
grad_thresh = 10

gases = {'He': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
         'N2': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
         'Ar': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
         'Kr': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3', ], True], \
         'Xe': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
         'SF6': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
        }

# gases = {'He': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
#          'N2': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
#          #'N2': [['50kHz_4Vpp_1'], True], \
#          #'Ar': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
#          #'Kr': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3', ], True], \
#          #'Xe': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
#          #'SF6': [['50kHz_4Vpp_1', '50kHz_4Vpp_2', '50kHz_4Vpp_3'], True], \
#         }


keys = ['data', 'pmax', 'pmax_sterr', 'pmax_syserr', 'p_outgassing', 'dipole', 'dipole_sterr', \
        'dipole_syserr', 'mass', 'rot_freq', 'rot_freq_err', 'rot_amp', 'rot_amp_err', \
        'filenames']

outdat = {}
other_outdat = []
for gas in gases.keys():
    outdat[gas] = {}
    for key in keys:
        outdat[gas][key] = []
for pathind, path in enumerate(other_paths):
    other_dict = {}
    for gas in other_gases[pathind]:
        other_dict[gas] = {}
        for key in keys:
            other_dict[gas][key] = []
    other_outdat.append(other_dict)


fig_ex, ax_ex = plt.subplots(1,1,figsize=(6,3),dpi=200)
temp_colors = bu.get_color_map(len(example_gases), cmap='plasma')[::-1]
example_colors = {}
for gasind, gas in enumerate(example_gases):
    example_colors[gas] = temp_colors[gasind]


outgassing_dir = '/data/old_trap_processed/spinning/pramp_data/20190626/outgassing/'
files, lengths = bu.find_all_fnames(outgassing_dir, ext='.txt')
rates = []
for filename in files:
    file_obj = open(filename, 'rb')
    lines = file_obj.readlines()
    file_obj.close()
    rate = float(lines[2])
    rates.append(rate)
outgassing_rate = np.mean(rates)
# outgassing_rate = 0.0


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

    # plt.figure()
    # plt.plot(field_data['field_amp'])
    # plt.figure()
    # plt.plot(field_data['field_freq'])
    # plt.show()

    pressures_real, pressures_smooth = build_full_pressure(pressures, plot=plot_pressures, \
                                                           use_highp_bara=use_highp_bara)

    sort_inds = np.argsort(pressures_real)

    phases = phases[sort_inds]
    phase_errs = phase_errs[sort_inds]
    pressures_real = pressures_real[sort_inds]
    #plt.errorbar(pressures_real, phases, yerr=phase_errs, ms=2)
    #plt.show()

    # phases_start = phases[:50]
    # phases_start = np.unwrap( 2.0 * phases_start ) / 2.0
    # plt.plot(phases[:50])
    # plt.plot(phases_start)
    # plt.show()

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
    
    # print 'pmax uncertainty: ', pmax_err / pmax

    return np.array([np.concatenate((pressures_out_1, pressures_out_2)), \
                     np.concatenate((uphases_out_1, uphases_out_2))]), \
                     pmax, pmax_err, p_outgassing, rot_freq, rot_freq_err, \
                     rot_amp, rot_amp_err



gas_keys = gases.keys()
gas_keys.sort(key = lambda x: ru.raw_mass[x])
for gas in gas_keys:
    fils = gases[gas][0]
    use_highp_bara = gases[gas][1]

    maxp = 0

    for filind, fil in enumerate(fils):
        filname = base_path + '/' + gas + '/' + fil
        #print 'Main file: ', filname
        dat, pmax, pmax_err, p_outgassing, rot_freq, rot_freq_err, rot_amp, rot_amp_err = \
                analyze_file(filname, nbins=nbins_user, use_highp_bara=use_highp_bara, \
                             grad_thresh=grad_thresh, plot_raw_data=plot_raw_data, \
                             plot_pressures=plot_pressures)

        mass_filename = base_path + '/' + gas + '/rga-m0_%i.mass' % (filind + ind_offset)
        mass_arr = np.load(open(mass_filename, 'rb'), allow_pickle=True)

        dipole_filename = base_dipole_path + '/' + gas + '_pramp_' + str(filind + ind_offset) + '.dipole'
        dipole_dat = np.load(open(dipole_filename, 'rb'))
        dipole = dipole_dat[0] / dipole_units
        dipole_sterr = dipole_dat[1] / dipole_units
        dipole_syserr = dipole_dat[2] / dipole_units

        # print np.sqrt(dipole_sterr**2 + dipole_syserr**2) / dipole

        # Include 1% systematic from baratron
        # pmax_err = np.sqrt(pmax_err**2 + (0.01*pmax)**2)

        outdat[gas]['filenames'].append(filname)
        outdat[gas]['data'].append(dat)
        outdat[gas]['pmax'].append(pmax)
        outdat[gas]['pmax_sterr'].append(pmax_err)
        outdat[gas]['pmax_syserr'].append(0.01 * pmax)
        outdat[gas]['p_outgassing'].append(p_outgassing)

        outdat[gas]['dipole'].append(dipole)
        outdat[gas]['dipole_sterr'].append(dipole_sterr)
        outdat[gas]['dipole_syserr'].append(dipole_syserr)
        outdat[gas]['mass'].append(mass_arr)

        outdat[gas]['rot_freq'].append(rot_freq)
        outdat[gas]['rot_freq_err'].append(rot_freq_err)
        outdat[gas]['rot_amp'].append(rot_amp)
        outdat[gas]['rot_amp_err'].append(rot_amp_err)

        cmax = np.max(dat[0])
        if cmax > maxp:
            maxp = cmax


    if make_example_pramp and (gas in example_gases):

        if gas == 'He':
            line_p = np.linspace(-0.5 * pmax, 1.5 * pmax, 100) 
            min_line = np.ones_like(line_p) * (-0.5)
            ax_ex.plot(line_p, min_line, '--', color='k', lw=2, alpha=0.6)
            ax_ex.set_xlim(-0.05*maxp, 1.05*maxp)

        dat = outdat[gas]['data'][filind]
        pmax = outdat[gas]['pmax'][filind]

        lab_str = gas + ': $P_{{\\rm max }}$ = {:0.3f} mbar'.format(pmax)

        color = example_colors[gas]

        fitp = np.linspace(0, np.max(dat[0]), 100)
        fit = np.array(phi_ffun(fitp, pmax, 0))
        ax_ex.scatter(dat[0], dat[1] / np.pi, edgecolors=color, facecolors='none', alpha=0.5)
        ax_ex.plot(fitp, fit / np.pi, '-', color=color, lw=3, label=lab_str)


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

            lab_str = '$P_{\mathrm{max}}$ = %0.3f' % pmax

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


    if include_other_beads:
        for baseind, base in enumerate(other_paths):
            if gas not in other_gases[baseind]:
                continue
            base_dipole = other_dipole_paths[baseind]
            print base
            for filind, fil in enumerate(fils):
                other_filname = base + '/' + gas + '/' + fil
                #print 'Other file: ', other_filname
                dat_o, pmax_o, pmax_err_o, p_outgassing_o, rot_freq_o, \
                    rot_freq_err_o, rot_amp_o, rot_amp_err_o = \
                        analyze_file(other_filname, nbins=nbins_user, use_highp_bara=use_highp_bara, \
                                     grad_thresh=grad_thresh, plot_raw_data=plot_raw_data, \
                                     plot_pressures=plot_pressures)

                mass_filename_o = base + '/' + gas + '/rga-m0_%i.mass' % (filind + ind_offset)
                mass_arr_o = np.load(open(mass_filename_o, 'rb'), allow_pickle=True)
                if debug:
                    print
                    print 'NAME'
                    print mass_filename_o
                    print

                dipole_filename_o = base_dipole + '/' + gas + '_pramp_' + str(filind + ind_offset) + '.dipole'
                dipole_dat_o = np.load(open(dipole_filename_o, 'rb'))
                dipole_o = dipole_dat_o[0] / dipole_units
                dipole_sterr_o = dipole_dat_o[1] / dipole_units
                dipole_syserr_o = dipole_dat_o[2] / dipole_units

                # print np.sqrt(dipole_sterr_o**2 + dipole_syserr_o**2) / dipole_o

                # Include 1% systematic from field amplitude uncertainty
                #pmax_err_o = np.sqrt(pmax_err_o**2 + (0.01*pmax_o)**2)

                other_outdat[baseind][gas]['filenames'].append(other_filname)
                other_outdat[baseind][gas]['data'].append(dat_o)
                other_outdat[baseind][gas]['pmax'].append(pmax_o)
                other_outdat[baseind][gas]['pmax_sterr'].append(pmax_err_o)
                other_outdat[baseind][gas]['pmax_syserr'].append(0.05 * pmax_o)
                other_outdat[baseind][gas]['p_outgassing'].append(p_outgassing_o)

                other_outdat[baseind][gas]['dipole'].append(dipole_o)
                other_outdat[baseind][gas]['dipole_sterr'].append(dipole_sterr_o)
                other_outdat[baseind][gas]['dipole_syserr'].append(dipole_syserr_o)
                other_outdat[baseind][gas]['mass'].append(mass_arr_o)

                other_outdat[baseind][gas]['rot_freq'].append(rot_freq_o)
                other_outdat[baseind][gas]['rot_freq_err'].append(rot_freq_err_o)
                other_outdat[baseind][gas]['rot_amp'].append(rot_amp_o)
                other_outdat[baseind][gas]['rot_amp_err'].append(rot_amp_err_o)



if make_example_pramp:
    ax_ex.set_xlabel('Pressure [mbar]')
    ax_ex.set_ylabel('$\\phi_{{\\rm eq}}$ [$\\pi$ rad]')
    ax_ex.legend(fontsize=10)
    fig_ex.tight_layout()

    #fig_path = base_plot_path + ('/%s_pramp.png' % gas)
    #bu.make_all_pardirs(fig_path)
    fig_ex.savefig(example_figname)
    plt.show()

pickle.dump(outdat, open(base_path + '/all_data.p', 'wb'))



def proportional(x, a, b):
    return a * x + b

def inverse_sqrt(x, a, b):
    return a * (1.0 / np.sqrt(x)) #+ b

#pmax_He = np.mean(outdat['He']['pmax'])



def process_outdat(outdat_dict):
    pmax_vec = []
    pmax_err_vec = []

    pmax_norm_vec = []
    pmax_norm_sterr_vec = []
    pmax_norm_syserr_vec = []

    raw_mass_vec = []
    mass_vec = []
    mass_err_vec = []

    rot_freq_vec = []
    rot_freq_err_vec = []
    rot_amp_vec = []
    rot_amp_err_vec = []
    for gas in gas_keys:

        if gas not in outdat_dict.keys():
            continue
        
        #if gas == 'N2':
        #    continue

        rot_freq_vec.append( np.mean(outdat_dict[gas]['rot_freq']) )
        rot_freq_err_vec.append( np.sqrt( 1.0 / (len(outdat_dict[gas]['rot_freq']) - 1)  * \
                                np.sum(np.array(outdat_dict[gas]['rot_freq_err'])**2) ) )
        rot_amp_vec.append( np.mean(outdat_dict[gas]['rot_amp']) )
        rot_amp_err_vec.append( np.sqrt( 1.0 / (len(outdat_dict[gas]['rot_amp']) - 1)  * \
                                np.sum(np.array(outdat_dict[gas]['rot_amp_err'])**2) ) )

        gas_pmax_vec = np.array(outdat_dict[gas]['pmax'])
        gas_pmax_sterr_vec = np.array(outdat_dict[gas]['pmax_sterr'])
        gas_pmax_syserr_vec = np.array(outdat_dict[gas]['pmax_syserr'])

        gas_dipole_vec = np.array(outdat_dict[gas]['dipole'])
        gas_dipole_sterr_vec = np.array(outdat_dict[gas]['dipole_sterr'])
        gas_dipole_syserr_vec = np.array(outdat_dict[gas]['dipole_syserr'])

        gas_norm_vec = gas_pmax_vec / gas_dipole_vec

        pmax_vec.append( np.mean(outdat_dict[gas]['pmax']) )
        pmax_err_vec.append( np.std(outdat_dict[gas]['pmax']) )

        pmax_norm_vec.append( np.mean( gas_norm_vec ) )

        gas_norm_sterr_vec = gas_norm_vec * np.sqrt((gas_dipole_sterr_vec / gas_dipole_vec)**2 + \
                                                    (gas_pmax_sterr_vec / gas_pmax_vec)**2 )
        gas_norm_syserr_vec = gas_norm_vec * np.sqrt((gas_dipole_syserr_vec / gas_dipole_vec)**2 + \
                                                     (gas_pmax_syserr_vec / gas_pmax_vec)**2 )

        sterr_val = np.sqrt( 1.0 / np.sum(1.0 / gas_norm_sterr_vec**2) )
        syserr_val = np.sqrt( 1.0 / np.sum(1.0 / gas_norm_syserr_vec**2) )

        pmax_norm_sterr_vec.append( sterr_val )
        pmax_norm_syserr_vec.append( syserr_val )

        #print 'stat: ', sterr_val / np.mean(gas_norm_vec)
        #print ' sys: ', syserr_val / np.mean(gas_norm_vec)

        #raw_mass_vec.append()

        mass_arr = np.array(outdat_dict[gas]['mass'])
        if debug:
            print
            print outdat_dict[gas]['filenames']
            print mass_arr
            print 
        mean_mass, mean_mass_err = bu.weighted_mean(mass_arr[:,0], mass_arr[:,1])
        mass_vec.append( mean_mass )
        mass_err_vec.append( mean_mass_err )

    mass_vec = np.array(mass_vec)
    mass_err_vec = np.array(mass_err_vec)

    # print 'stat: ', np.array(pmax_norm_sterr_vec) / np.array(pmax_norm_vec)
    # print ' sys: ', np.array(pmax_norm_syserr_vec) / np.array(pmax_norm_vec)

    return {'pmax_vec': np.array(pmax_vec), 'pmax_err_vec': np.array(pmax_err_vec), \
            'pmax_norm_vec': np.array(pmax_norm_vec), \
            'pmax_norm_sterr_vec': np.array(pmax_norm_sterr_vec), \
            'pmax_norm_syserr_vec': np.array(pmax_norm_syserr_vec), \
            'raw_mass_vec': np.array(raw_mass_vec), 'mass_vec': np.array(mass_vec), \
            'mass_err_vec': np.array(mass_err_vec), \
            'rot_freq_vec': np.array(rot_freq_vec), 'rot_freq_err_vec': np.array(rot_freq_err_vec), \
            'rot_amp_vec': np.array(rot_amp_vec), 'rot_amp_err_vec': np.array(rot_amp_err_vec)}





proc_outdat = process_outdat(outdat)

if debug:
    print
    print 'OTHERS'
    print 

if include_other_beads:
    other_processed = []
    for baseind, base in enumerate(other_paths):
        other_processed.append(process_outdat(other_outdat[baseind]))



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








def fit_kappa(proc_outdat_dict, mbead, plot_chi2=False):

    rbead = bu.get_rbead(mbead)

    mass_vec = proc_outdat_dict['mass_vec']
    mass_err_vec = proc_outdat_dict['mass_err_vec']
    pmax_norm_vec = proc_outdat_dict['pmax_norm_vec']
    pmax_norm_err_vec = proc_outdat_dict['pmax_norm_sterr_vec']

    pmax_norm_vec_upper = pmax_norm_vec + proc_outdat_dict['pmax_norm_syserr_vec']
    pmax_norm_vec_lower = pmax_norm_vec - proc_outdat_dict['pmax_norm_syserr_vec']

    popt3, pcov3 = opti.curve_fit(inverse_sqrt, mass_vec, pmax_norm_vec, [0.1 / 100, 0])

    def cost(param):
        resid = np.abs(inverse_sqrt(mass_vec, param, 0) - pmax_norm_vec)
        norm = 1. / (len(mass_vec) - 1)
        tot_var = pmax_norm_err_vec**2 + pmax_norm_vec**2 * (mass_err_vec / mass_vec)**2
        return norm * np.sum( resid**2 / tot_var)

    def cost_upper(param):
        resid = np.abs(inverse_sqrt(mass_vec, param, 0) - pmax_norm_vec_upper)
        norm = 1. / (len(mass_vec) - 1)
        tot_var = pmax_norm_err_vec**2 + pmax_norm_vec_upper**2 * (mass_err_vec / mass_vec)**2
        return norm * np.sum( resid**2 / tot_var)

    def cost_lower(param):
        resid = np.abs(inverse_sqrt(mass_vec, param, 0) - pmax_norm_vec_lower)
        norm = 1. / (len(mass_vec) - 1)
        tot_var = pmax_norm_err_vec**2 + pmax_norm_vec_lower**2 * (mass_err_vec / mass_vec)**2
        return norm * np.sum( resid**2 / tot_var)

    #param_arr = np.linspace(0.98*popt3[0], 1.02*popt3[0], 200)
    #param, param_err, min_chi = bu.minimize_nll(cost, param_arr, plot=plot_chi2)

    m=Minuit(cost,
             param = popt3[0], # set start parameter
             #fix_param = "True", # you can also fix it
             #limit_param = (0.0, 10000.0),
             errordef = 1,
             print_level = 0, 
             pedantic=False)
    m.migrad(ncall=500000)
    minos = m.minos()

    m_upper=Minuit(cost_upper,
                   param = popt3[0], # set start parameter
                   #fix_param = "True", # you can also fix it
                   #limit_param = (0.0, 10000.0),
                   errordef = 1,
                   print_level = 0, 
                   pedantic=False)
    m_upper.migrad(ncall=500000)

    m_lower=Minuit(cost_lower,
                   param = popt3[0], # set start parameter
                   #fix_param = "True", # you can also fix it
                   #limit_param = (0.0, 10000.0),
                   errordef = 1,
                   print_level = 0, 
                   pedantic=False)
    m_lower.migrad(ncall=500000)

    param = {}
    param['val'] = minos['param']['min']
    param['sterr'] = np.mean(np.abs([minos['param']['upper'], minos['param']['lower']]))

    param['syserr'] = np.mean(np.abs(m.values['param'] - \
                                np.array([m_upper.values['param'], m_lower.values['param']])))

    min_chi = m.fval

    # convert from units of mbar * amu^1/2 / (e * um) to Pa * kg^1/2 / (C * m)
    conv_fac = (100.0) * np.sqrt(1.6605e-27) * (1.0 / dipole_units)
    param_si = param['val'] * conv_fac
    param_si_sterr = param['sterr'] * conv_fac
    param_si_syserr = param['syserr'] * conv_fac

    rot_freq = np.mean(proc_outdat_dict['rot_freq_vec'])
    rot_amp =  np.mean(proc_outdat_dict['rot_amp_vec'])

    kappa = {}
    kappa['val'] = param_si * 2.0 * np.pi * rot_freq / rot_amp

    #print mass_vec
    #print rot_freq_err_vec

    #print param_si_err / param_si
    #print np.median(rot_freq_err_vec) / rot_freq
    #print np.median(rot_amp_err_vec) / rot_amp

    kappa['sterr'] = kappa['val'] * np.sqrt( (np.max(proc_outdat_dict['rot_freq_err_vec']) / rot_freq)**2 + \
                                 (np.max(proc_outdat_dict['rot_amp_err_vec']) / rot_amp)**2 + \
                                 (param_si_sterr / param_si)**2 )
    kappa['syserr'] = kappa['val'] * np.sqrt( (param_si_syserr / param_si)**2 )


    sigma = {}
    sigma['val'] = (1.0 / (kappa['val'] * rbead['val']**4)) * \
                        np.sqrt( (9.0 * kb * T) / (32.0 * np.pi) )
    sigma['sterr'] = sigma['val'] * np.sqrt( (kappa['sterr']/kappa['val'])**2 + \
                                    16.0 * (rbead['sterr']/rbead['val'])**2 )
    sigma['syserr'] = sigma['val'] * np.sqrt( (kappa['syserr']/kappa['val'])**2 + \
                                    16.0 * (rbead['syserr']/rbead['val'])**2 )

    kappa_calc = bu.get_kappa(mbead, verbose=True)

    sigma2 = {}
    sigma2['val'] = kappa_calc['val'] / kappa['val']
    sigma2['sterr'] = sigma2['val'] * np.sqrt( (kappa['sterr']/kappa['val'])**2 + \
                                    (kappa_calc['sterr']/kappa_calc['val'])**2 )
    sigma2['syserr'] = sigma2['val'] * np.sqrt( (kappa['syserr']/kappa['val'])**2 + \
                                    (kappa_calc['syserr']/kappa_calc['val'])**2 )



    print
    print 'Kappa (meas) : {:0.4g} +- {:0.4g} (st) +- {:0.4g} (sys)'\
                        .format(kappa['val'], kappa['sterr'], kappa['syserr'])
    print 'Kappa (calc) : {:0.4g} +- {:0.4g} (st) +- {:0.4g} (sys)'\
                        .format(kappa_calc['val'], kappa_calc['sterr'], kappa_calc['syserr'])
    print 'Rbead        : {:0.4g} +- {:0.4g} (st) +- {:0.4g} (sys)'\
                        .format(rbead['val'], rbead['sterr'], rbead['syserr'])
    print 'mbead        : {:0.4g} +- {:0.4g} (st) +- {:0.4g} (sys)'\
                        .format(mbead['val'], mbead['sterr'], mbead['syserr'])
    print 'sigma        : {:0.4g} +- {:0.4g} (st) +- {:0.4g} (sys)'\
                        .format(sigma['val'], sigma['sterr'], sigma['syserr'])
    print 'sigma2       : {:0.4g} +- {:0.4g} (st) +- {:0.4g} (sys)'\
                        .format(sigma2['val'], sigma2['sterr'], sigma2['syserr'])
    print 'min chi: {:0.2g}'.format(min_chi)

    return {'kappa': kappa, \
            'kappa_calc': kappa_calc, \
            'sigma': sigma, \
            'param': param}





kappa_dict = fit_kappa(proc_outdat, mbead)


if include_other_beads:
    other_kappa_dicts = []
    for baseind, base in enumerate(other_paths):
        kappa_dict_o = fit_kappa(other_processed[baseind], other_masses[baseind])
        other_kappa_dicts.append(kappa_dict_o)


# ann_str = 'Systematic offset of %0.4f mbar removed' % popt3[1]

#xplot3 = np.linspace(0, 1.1*np.max(mass_vec), 100)
xplot = np.linspace(0, 150, 100)
xplot[0] += 1e-9
#yplot3 = inverse_sqrt(xplot3, popt3[0], 0)
yplot = inverse_sqrt(xplot, kappa_dict['param']['val'], 0)

k_sv = bu.get_scivals(kappa_dict['kappa']['val'])
ke_sv = bu.get_scivals(np.sqrt(kappa_dict['kappa']['sterr']**2 + \
                                kappa_dict['kappa']['syserr']**2))

#r_sv = bu.get_scivals(kappa_dict['rdrag'][0])
#re_sv = bu.get_scivals(kappa_dict['rdrag'][1])
#re_sv = bu.get_scivals(np.sqrt(kappa_dict['rdrag'][1]**2 + kappa_dict['rdrag'][2]**2))

err_exp_diff = k_sv[1] - ke_sv[1]
ke_sv = (ke_sv[0] / (10.0**err_exp_diff), k_sv[1])

#err_exp_diff = r_sv[1] - re_sv[1]
#re_sv = (re_sv[0] / (10.0**err_exp_diff), r_sv[1])

label = '$ \\kappa_3 = ({0} \\pm {1})$'.format('{:0.2f}'.format(k_sv[0]), '{:0.2f}'.format(ke_sv[0]) ) \
            + '$ \\times 10^{{{0}}}$ '.format('{:d}'.format(k_sv[1])) \
            + '$\\mathrm{J}^{1/2} \\mathrm{m}^{-4}$'

label2 = '$ \\rightarrow \\sigma_1 = {:0.2f} \\pm {:0.2f} $'\
                .format(kappa_dict['sigma']['val'], \
                        np.sqrt(kappa_dict['sigma']['sterr']**2 + kappa_dict['sigma']['syserr']**2))

# label2 = 'min($ \\chi ^2 / N_{\\mathrm{DOF}}$)=' + '{0}'.format('{:0.2f}'.format(min_chi))

# if r_sv[1] == -6:
#     label3 = '$ \\rightarrow \\mathrm{r}_{\mathrm{drag}} = $' \
#             + '${0} \\pm {1}$'.format('{:0.2f}'.format(r_sv[0]), '{:0.2f}'.format(re_sv[0])) \
#             + ' $\\mu \\mathrm{m}$'
# else:
#     label3 = '$ \\rightarrow \\mathrm{r}_{\mathrm{drag}} = $' \
#             + '$({0} \\pm {1})$'.format('{:0.2f}'.format(r_sv[0]), '{:0.2f}'.format(re_sv[0])) \
#             + '$\\times 10^{{{0}}}$'.format('{:d}'.format(r_sv[1]+6)) \
#             + ' $\\mu \\mathrm{m}$'



#print mass_vec, mass_err_vec, pmax_norm_vec, pmax_norm_err_vec

fig3, axarr3 = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3,1]}, \
                            sharex=True, figsize=(6,4),dpi=200)

axarr3[0].errorbar(proc_outdat['mass_vec'], proc_outdat['pmax_norm_vec']*1e3, \
                    xerr=proc_outdat['mass_err_vec'], \
                    yerr=np.sqrt(proc_outdat['pmax_norm_sterr_vec']**2 + \
                                 proc_outdat['pmax_norm_syserr_vec']**2)*1e3, \
                    fmt='o', ms=4, color=colors[0])
axarr3[0].plot(xplot, yplot*1e3, '--', color=colors[0], lw=2, alpha=0.65, label=label)
#axarr3[0].plot([1], [1], color='w', label=label2)


resid_vec = proc_outdat['pmax_norm_vec'] - inverse_sqrt(proc_outdat['mass_vec'], \
                                                        kappa_dict['param']['val'], 0)

axarr3[1].errorbar(proc_outdat['mass_vec'], resid_vec*1e3, xerr=proc_outdat['mass_err_vec'], \
                       yerr=np.sqrt(proc_outdat['pmax_norm_sterr_vec']**2 + \
                                 proc_outdat['pmax_norm_syserr_vec']**2)*1e3, \
                       fmt='o', ms=4, color=colors[0])

if include_other_beads:
    for pathind, other_proc_outdat in enumerate(other_processed):
        k_sv_o = bu.get_scivals(other_kappa_dicts[pathind]['kappa']['val'])
        ke_sv_o = bu.get_scivals(np.sqrt(other_kappa_dicts[pathind]['kappa']['sterr']**2 + \
                                            other_kappa_dicts[pathind]['kappa']['syserr']**2))

        xplot2 = np.linspace(0, 28, 100)
        yplot2 = inverse_sqrt(xplot2, other_kappa_dicts[pathind]['param']['val'], 0)

        err_exp_diff_o = k_sv_o[1] - ke_sv_o[1]
        ke_sv_o = (ke_sv_o[0] / (10.0**err_exp_diff_o), k_sv_o[1])

        label3 = '$ \\kappa_{0} = ({1} \\pm {2})$'.format(str(2-pathind), '{:0.2f}'.format(k_sv_o[0]), '{:0.2f}'.format(ke_sv_o[0]) ) \
                    + '$ \\times 10^{{{0}}}$ '.format('{:d}'.format(k_sv_o[1])) \
                    + '$\\mathrm{J}^{1/2} \\mathrm{m}^{-4}$'

        label4 = '$ \\rightarrow \\sigma_{:d} = {:0.2f} \\pm {:0.2f} $'\
                .format(pathind+2, other_kappa_dicts[pathind]['sigma']['val'], \
                        np.sqrt(other_kappa_dicts[pathind]['sigma']['sterr']**2 + \
                                other_kappa_dicts[pathind]['sigma']['syserr']**2))

        #axarr3[0].plot([1], [1], color='w', label=' ')
        axarr3[0].errorbar(other_proc_outdat['mass_vec'], \
                            other_proc_outdat['pmax_norm_vec']*1e3, \
                            xerr=other_proc_outdat['mass_err_vec'], \
                            yerr=np.sqrt(other_proc_outdat['pmax_norm_sterr_vec']**2 + \
                                         other_proc_outdat['pmax_norm_syserr_vec']**2)*1e3, \
                            ms=4, color=colors[pathind+1], \
                            fmt=other_markers[pathind], \
                            #label=label3 \
                            )

        axarr3[0].plot(xplot2, yplot2*1e3, color=colors[pathind+1], \
                        linestyle=other_linestyles[pathind], lw=2, alpha=0.65, label=label3)
        #axarr3[0].plot([1], [1], color='w', label=label4)

        other_resid_vec = other_proc_outdat['pmax_norm_vec'] - \
                                inverse_sqrt(other_proc_outdat['mass_vec'], \
                                                other_kappa_dicts[pathind]['param']['val'], 0)
        axarr3[1].errorbar(other_proc_outdat['mass_vec'], other_resid_vec*1e3, \
                            xerr=other_proc_outdat['mass_err_vec'], \
                            yerr=np.sqrt(other_proc_outdat['pmax_norm_sterr_vec']**2 + \
                                         other_proc_outdat['pmax_norm_syserr_vec']**2)*1e3, \
                            ms=4, color=colors[pathind+1], \
                            fmt=other_markers[pathind])
        # axarr3[1].plot(xplot, np.zeros_like(xplot), '--', \
        #                 color=other_colors[pathind], alpha=0.5)


# axarr3[1].errorbar(mass_vec, (resid_vec)/np.array(pmax_norm_err_vec), \
#                    xerr=mass_err_vec,yerr=np.ones_like(pmax_norm_err_vec), fmt='o', ms=5)

# axarr3[1].errorbar(mass_vec, 1e2*resid_vec/np.array(pmax_norm_vec), xerr=mass_err_vec, \
#                         yerr=1e2*np.array(pmax_norm_err_vec)/np.array(pmax_norm_vec), fmt='o', ms=4)
axarr3[1].plot(xplot, np.zeros_like(xplot), '--', color='k', alpha=0.5)
#ax3.text(0.98, 0.9, ann_str, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)


for label, mass in [('He', 4), ('N$_2$', 28), ('Ar', 40), \
                ('Kr', 83.06), ('Xe', 126.16), ('SF$_6$', 145.11)]:
    axarr3[0].text(mass, 1.6 / np.sqrt(mass) - 0.1, label, fontdict={'size': 12}, \
                    horizontalalignment='center', verticalalignment='bottom')


#ax3.set_xlim(0, 1.1*np.max(mass_vec))
axarr3[0].set_xlim(0, 150)
axarr3[1].set_xlabel('$m_{0,\mathrm{eff}}$ [amu]')
axarr3[0].set_ylim(0, 1.1*np.max(proc_outdat['pmax_norm_vec'])*1e3)
#axarr3[0].set_ylabel(r'$p_{\mathrm{max}} / d$'+'\n[$10^{-3}$ mbar / ($e \cdot \mu m$)]')
axarr3[0].set_ylabel(r'$P_{\mathrm{max}} / d$'+'\n[$\mu$bar / ($e \cdot \mu m$)]')
#axarr3[1].set_ylabel('Resid. [$\sigma$]')
#axarr3[1].set_ylabel('Resid. [%]')
axarr3[1].set_ylabel('Resid.')
#axarr3[1].set_ylabel('Resid\n[mbar / ($e \cdot \mu m$)]')

#handles, labels = axarr3[0].get_legend_handles_labels()


axarr3[0].legend(loc='upper right', fontsize=10)
plt.tight_layout()

if savefig:
    fig_path1 = base_plot_path + '/all_pramp_fit.png'
    fig_path2 = base_plot_path + '/all_pramp_fit.pdf'
    fig_path3 = base_plot_path + '/all_pramp_fit.svg'
    bu.make_all_pardirs(fig_path1)
    fig3.savefig(fig_path1)
    fig3.savefig(fig_path2)
    fig3.savefig(fig_path3)






plt.show()
    
    










