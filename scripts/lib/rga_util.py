import sys, os, time

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.optimize as opti

import bead_util as bu

plt.rcParams.update({'font.size': 14})

verbose = 0

raw_mass = {'He': 4.0, 'N2': 28, 'Ar': 40, 'Kr': 83.79, 'Xe': 131.29, 'SF6': 146}

gases = {'He': {'He4': 4.0}, \
         'H2O': {'H2O16': 18}, \
         'N2': {'N2_14': 28}, \
         'O2': {'O2_16': 32}, \
         'Ar': {'Ar40': 40}, \
         'Kr': {'Kr78': 78, 'Kr80': 80, 'Kr82': 82, \
                    'Kr83': 83, 'Kr84': 84, 'Kr86': 86}, \
         'Xe': {'Xe128': 128, 'Xe129': 129, 'Xe130': 130, \
                    'Xe131': 131, 'Xe132': 132, 'Xe134': 134, \
                    'Xe136': 136}, \
         'SF6': {'S32F6': 146, 'S34F6': 148}, \
         }


rga_sensitivities = {'He': 0.14, \
                     'H2O': 1.0, \
                     'N2': 1.0, \
                     'O2': 1.0, \
                     'Ar': 1.2, \
                     'Kr': 1.7, \
                     'Xe': 3.0, \
                     'SF6': 2.3, \
                     }

pumping_speed = {'He': 2, \
                 'H2O': 0.1, \
                 'N2': 1.0, \
                 'O2': 1.0, \
                 'Ar': 1.0, \
                 'Kr': 1.0, \
                 'Xe': 1.0, \
                 'SF6': 1.0, \
                 }


ions = {'He4': {'He+': 4}, \
       'H2O16': {'H2O+': 18, 'H3O+': 19, 'OH+': 17}, \
       'N2_14': {'N2+': 28, 'N2+2': 14}, \
       'O2_16': {'O2+': 32, 'O2+2': 16}, \
       'Ar40': {'Ar+': 40, 'Ar+2': 20}, \
       'Kr78': {'Kr78+': 78, 'Kr78+2': 39}, \
       'Kr80': {'Kr80+': 80, 'Kr80+2': 40}, \
       'Kr82': {'Kr82+': 82, 'Kr82+2': 41}, \
       'Kr83': {'Kr83+': 83, 'Kr83+2': 41.5}, \
       'Kr84': {'Kr84+': 84, 'Kr84+2': 42}, \
       'Kr86': {'Kr86+': 86, 'Kr86+2': 43}, \
       'Xe128': {'Xe128+': 128, 'Xe128+2': 64}, \
       'Xe129': {'Xe129+': 129, 'Xe129+2': 64.5}, \
       'Xe130': {'Xe130+': 130, 'Xe130+2': 65}, \
       'Xe131': {'Xe131+': 131, 'Xe131+2': 65.5}, \
       'Xe132': {'Xe132+': 132, 'Xe132+2': 66}, \
       'Xe134': {'Xe134+': 134, 'Xe134+2': 67}, \
       'Xe136': {'Xe136+': 136, 'Xe136+2': 68}, \
       'S32F6': {'SF5+': 127, 'SF4+': 108, 'SF3+': 89, 'SF2+': 70, 'SF+': 51, 'S+': 32, \
                    'SF5+2': 63.5, 'SF4+2': 54, 'SF3+2': 44.5, 'SF2+2': 35, 'F+': 19}, \
       'S34F6': {'SF5+': 129, 'SF4+': 110, 'SF3+': 91, 'SF2+': 72, 'SF+': 53, 'S+': 34, \
                    'SF5+2': 64.5, 'SF4+2': 55, 'SF3+2': 45.5, 'SF2+2': 36, 'F+': 19} \
       }



gases_to_label = \
        {'He': {'He$^+$': 3.9, \
                'H$_2$O$^+$': 18, \
                'N$_2^+$': 28, \
                'O$_2^+$': 32, \
                'Ar$^+$': 40, \
                '$^{84}$Kr$^+$': 84, \
                '$^{132}$Xe$^+$': 132, \
                }, \
         'N2': {'He$^+$': 3.9, \
                'H$_2$O$^+$': 18, \
                'N$_2^+$': 28, \
                'N$_2^{(2{+})}$': 14, \
                'O$_2^+$': 32, \
                'Ar$^+$': 40, \
                '$^{84}$Kr$^+$': 84, \
                '$^{132}$Xe$^+$': 132, \
                }, \
         'Ar': {'He$^+$': 3.9, \
                'H$_2$O$^+$': 18, \
                'N$_2^+$': 28, \
                'O$_2^+$': 32, \
                'Ar$^+$': 40, \
                'Ar$^{2{+}}$': 20, \
                '$^{84}$Kr$^+$': 84, \
                '$^{132}$Xe$^+$': 132, \
                }, \
         'Kr': {'He$^+$': 3.9, \
                'H$_2$O$^+$': 18, \
                'N$_2^+$': 28, \
                'O$_2^+$': 32, \
                'Ar$^+$': 40, \
                '$^{82}$Kr$^+$': 82, \
                '$^{84}$Kr$^+$': 84, \
                '$^{84}$Kr$^{(2{+})}$': 42, \
                '$^{132}$Xe$^+$': 132, \
                }, \
         'Xe': {'He$^+$': 3.9, \
                'H$_2$O$^+$': 18, \
                'N$_2^+$': 28, \
                'O$_2^+$': 32, \
                'Ar$^+$': 40, \
                '$^{84}$Kr$^+$': 84, \
                '$^{129}$Xe$^+$': 129, \
                '$^{136}$Xe$^+$': 136, \
                '$^{131}$Xe$^{(2{+})}$': 65.5, \
                #'$^{132}$Xe$^+$': 132, \
                #'$^{134}$Xe$^+$': 134, \
                }, \
         'SF6': {'He$^+$': 3.9, \
                'H$_2$O$^+$': 18, \
                'N$_2^+$': 28, \
                'Ar$^+$': 40, \
                '$^{84}$Kr$^+$': 84, \
                #'$^{132}$Xe$^+$': 132, \
                '$^{32}$SF$_5^+$': 127, \
                '$^{34}$SF$_5^+$': 129, \
                #'SF$_5^{2+}$': 63.5, \
                'SF$_4^+$': 108, \
                'SF$_4^{(2{+})}$': 54, \
                'SF$_3^+$': 89, \
                'SF$_3^{(2{+})}$': 44.5, \
                'SF$_2^+$': 70, \
                'SF$_2^{(2{+})}$': 35, \
                'SF$^+$': 51, \
                'S$^+$': 32, \
                }, \

        }



def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0*(x-mu)**2 / (2.0*sigma**2)) + c


def ngauss(x, A, mu, sigma, n):
    return A * np.exp(-1.0*np.abs(x-mu)**n / (2.0*sigma)**n)

def ngauss_many(x, A_vec, mu_vec, sigma_vec, n_vec, N):
    try:
        Npts = len(x)
    except:
        x = [x]
        Npts = 1
    x_arr = np.tile(x, reps=(N,1))#, axis=0)
    A_arr = np.tile(A_vec, reps=(Npts,1)).T
    mu_arr = np.tile(mu_vec, reps=(Npts,1)).T
    sigma_arr = np.tile(sigma_vec, reps=(Npts,1)).T
    n_arr = np.tile(n_vec, reps=(Npts,1)).T

    vals = A_arr * np.exp( -1.0*np.abs(x_arr - mu_arr)**n_arr / \
                                        (2.0*sigma_arr)**n_arr)
    sum_vec = np.sum(vals, axis=0)
    #for i in range(N):
    #    sum_vec += A_vec[i] * np.exp( -1.0*np.abs(x-mu_vec[i])**n_vec[i] / \
    #                                    (2.0*sigma_vec[i])**n_vec[i] ) 
    return sum_vec

def fit_wrapper(x, N, *args):
    A_vec, mu_vec, sigma_vec, n_vec = \
            args[0][:N], args[0][N:2*N], args[0][2*N:3*N], args[0][3*N:4*N]
    return ngauss_many(x, A_vec, mu_vec, sigma_vec, n_vec, N)

def fit_wrapper_2(x, N, *args):
    A_vec, mu_vec = list(args[0][:N]), list(args[0][N:2*N])
    sigma_vec = list(np.ones_like(A_vec)*0.275- np.linspace(0,1,len(A_vec))*0.075)
    n_vec = list(np.ones_like(A_vec)*5 - np.linspace(0,1,len(A_vec))*2.5)
    return ngauss_many(x, A_vec, mu_vec, sigma_vec, n_vec, N)

def fit_wrapper_3(x, N, *args):
    A_vec, mu_vec, n_vec = \
        list(args[0][:N]), list(args[0][N:2*N]), list(args[0][2*N:3*N])
    sigma_vec = list(np.ones_like(A_vec)*0.25- np.linspace(0,1,len(A_vec))*0.075)
    return ngauss_many(x, A_vec, mu_vec, sigma_vec, n_vec, N)


def extract_mass_vec(lines):

    for line_ind, line in enumerate(lines):
        if line[0:6] == '"Scan"':
            mass_line = line
            mass_line_ind = line_ind
            break

    nscans = len(lines) - 1 - mass_line_ind

    mass_strs = mass_line[7:].split('\t')
    mass_strs.pop(-1)
    mass_strs.pop(-1)
    extract = lambda x: float(x[6:-1])

    masses = np.array(list(map(extract, mass_strs)))

    return {'mass_vec': masses, 'nscans': nscans, \
            'mass_line_ind': mass_line_ind}



def extract_scans(lines, mass_line_ind, nscans):

    def debug(str_in):
        print(str_in)
        return float(str_in)

    badscan = 0
    scans = []
    pressures = []
    for i in range(nscans):

        scan_line = lines[mass_line_ind + 1 + i]
        scan_strs = scan_line.split('\t')

        scan_strs.pop(0)
        scan_strs.pop(-1)

        if scan_strs[-1] == '':
            badscan += 1
            continue

        pp_arr = np.abs(np.array(list(map(float, scan_strs[:-1]))))
        #pp_arr = np.abs(np.array(map(debug, scan_strs[:-1])))
        scans.append(pp_arr)

        try:
            pressures.append(float(scan_strs.pop(-1)))
        except:
            pressures.append(np.sum(pp_arr))

    scans = np.array(scans)

    return {'nscans': nscans-badscan, 'scans': scans, 'pressures': pressures}



def get_rga_data(rga_data_file, many_scans=True, last_nscans=1000, scan_ind=0, \
                 plot=True, plot_many=False, plot_last_scans=False, plot_nscans=1, \
                 gases_to_extract=[], ions_to_ignore=[], plot_extraction=False, \
                 fit_scale=1e8, save_fig=False, show=True, fig_base='', before=False):

    print('Processing:')
    print(rga_data_file)

    file_obj = open(rga_data_file)
    lines = file_obj.readlines()

    mass_data = extract_mass_vec(lines)
    mass_vec = mass_data['mass_vec'] + (1.0/8)

    scan_data = extract_scans(lines, mass_data['mass_line_ind'], \
                              mass_data['nscans'])
    nscans = scan_data['nscans']

    plot_x = mass_vec
    if many_scans:
        start_ind = nscans - last_nscans
        if start_ind < 0:
            start_ind = 0
        plot_y = np.mean(scan_data['scans'][start_ind:,:], axis=0)
        plot_errs = np.std(scan_data['scans'][start_ind:,:], axis=0)
        pressure = np.mean(scan_data['pressures'][start_ind:])
    else:
        plot_y = scan_data['scans'][scan_ind]
        plot_errs = np.zeros_like(plot_y) #* 0.1 * np.min(plot_y)
        pressure = scan_data['pressures'][scan_ind]

    if plot:
        title_str = 'Total Pressure: %0.3g mbar' % pressure

        fig, ax = plt.subplots(1,1,dpi=150,figsize=(10,3))
        ax.errorbar(plot_x, plot_y, yerr=plot_errs)
        ax.fill_between(plot_x, plot_y, np.ones_like(plot_y)*1e-9,\
                        alpha=0.5)
        ax.set_ylim(1e-9, 2*np.max(plot_y))
        ax.set_xlim(0,int(np.max(plot_x)))
        ax.set_yscale('log')
        ax.set_xlabel('Mass [amu]')
        ax.set_ylabel('Partial Pressure [mbar]')
        fig.suptitle(title_str)
        plt.tight_layout()
        plt.subplots_adjust(top=0.87)
        if save_fig:
            fig.savefig(fig_base+'single-scan.png')
        if not plot_many and show:
            plt.show()
        fig.clf()

    if plot_many:
        title_str = 'Total Pressure: %0.3g mbar' % pressure

        if plot_last_scans:
            ncolors = plot_nscans
        else:
            ncolors = nscans

        colors = bu.get_color_map(ncolors, cmap='inferno')
        fig2, ax2 = plt.subplots(1,1,dpi=150,figsize=(10,3))
        for i in range(nscans):
            newind = i - (nscans-plot_nscans)
            if plot_last_scans:
                if newind < 0:
                    continue
            ax2.plot(plot_x, scan_data['scans'][i], label=str(newind), 
                    color=colors[newind])
        ax2.set_ylim(1e-9, 2*np.max(plot_y))
        ax2.set_xlim(0,int(np.max(plot_x)))
        ax2.set_yscale('log')
        ax2.set_xlabel('Mass [amu]')
        ax2.set_ylabel('Partial Pressure [mbar]')
        fig2.suptitle(title_str)
        plt.tight_layout()
        plt.subplots_adjust(top=0.87)
        plt.legend(ncol=2)
        if save_fig:
            fig2.savefig(fig_base+'many-scans.png')
        if show:
            plt.show()
        fig2.clf()

    if len(gases_to_extract):

        if plot_extraction:
            title_str = 'Total Pressure: %0.3g mbar' % pressure

            fig_ex, ax_ex = plt.subplots(1,1,dpi=150,figsize=(10,3))
            ax_ex.errorbar(plot_x, plot_y, yerr=plot_errs)
            ax_ex.fill_between(plot_x, plot_y, np.ones_like(plot_y)*1e-9,\
                            alpha=0.5)
            ax_ex.set_yscale('log')
            ax_ex.set_xlabel('Mass [amu]')
            ax_ex.set_ylabel('Partial Pressure [mbar]')
            fig_ex.suptitle(title_str)
            plt.tight_layout()
            plt.subplots_adjust(top=0.87)


            title_str = 'Total Pressure: %0.3g mbar' % pressure

            fig_ex2, ax_ex2 = plt.subplots(1,1,dpi=150,figsize=(10,3))
            ax_ex2.errorbar(plot_x, plot_y, yerr=plot_errs)
            ax_ex2.fill_between(plot_x, plot_y, np.ones_like(plot_y)*1e-9,\
                            alpha=0.5)
            ax_ex2.set_xlabel('Mass [amu]')
            ax_ex2.set_ylabel('Partial Pressure [mbar]')
            fig_ex2.suptitle(title_str)
            plt.tight_layout()
            plt.subplots_adjust(top=0.87)

        gas_pressures = {}
        mq_arr = []
        amp_arr = []
        N_param = 0
        for gas in gases_to_extract:
            isotope_dict = gases[gas]

            isotopes = list(isotope_dict.keys())
            isotopes.sort()

            isotope_pressures = {}
            for isotope in isotopes:
                m0 = isotope_dict[isotope]

                ion_dict = ions[isotope]
                d_ions = list(ion_dict.keys())
                d_ions.sort()

                isotope_pressure = 0
                for ion in d_ions:
                    if ion in ions_to_ignore:
                        continue
                    mq = ion_dict[ion]
                    peak_pts = np.abs(plot_x - mq) <= 1
                    peak_pos = np.argmin(np.abs(plot_x - mq))

                    amp_arr.append(plot_y[peak_pos])
                    mq_arr.append(mq)



        print('Fitting desired peaks in RGA spectrum...') 
        sys.stdout.flush()

        fit_amps = []
        fit_mus = []
        fit_ns = []
        fit_sigmas = []

        mq_done = []

        mq_arr = np.array(mq_arr)
        gas_pressures = {}
        for gas_ind, gas in enumerate(gases_to_extract):
            bu.progress_bar(gas_ind, len(gases_to_extract), suffix='gases')
            isotope_dict = gases[gas]

            isotopes = list(isotope_dict.keys())
            isotopes.sort()

            isotope_pressures = {}
            for isotope in isotopes:
                m0 = isotope_dict[isotope]

                ion_dict = ions[isotope]
                d_ions = list(ion_dict.keys())
                d_ions.sort()

                isotope_pressure = 0
                isotope_pressure_var = 0
                for ion in d_ions:
                    if ion in ions_to_ignore:
                        continue
                    mq = ion_dict[ion]

                    if mq in mq_done:
                        continue

                    neighbor_peaks = np.abs(mq - mq_arr) < 1.75

                    if np.sum(neighbor_peaks) == 1:
                        start = time.time()
                        mq_done.append(mq)
                        peak_pts = np.abs(plot_x - mq) <= 1
                        n_peak_pts = np.sum(peak_pts)
                        peak_pos = np.argmin(np.abs(plot_x - mq))
                        #alpha = 0.333
                        #peak_window = signal.tukey(n_peak_pts, alpha=alpha)

                        p0 = [np.max(plot_y[peak_pts])*fit_scale, mq, 0.375, 4]
                        bounds = ([1e-10*fit_scale, mq-0.5, 0.15, 2], [1e-4*fit_scale, mq+0.5, .6, 6])
                        popt, pcov = opti.curve_fit(ngauss, plot_x[peak_pts], plot_y[peak_pts]*fit_scale, \
                                                    p0=p0, bounds=bounds, maxfev=100000)

                        fit_amps.append(popt[0])
                        fit_mus.append(popt[1])
                        fit_sigmas.append(popt[2])
                        fit_ns.append(popt[3])
                        stop = time.time()
                        print('regular ngauss fit: ', stop-start)


                    elif np.sum(neighbor_peaks) > 1:
                        start = time.time()
                        mq_subarr = mq_arr[neighbor_peaks]
                        more = True
                        new_neighbor_peaks = np.copy(neighbor_peaks)
                        while more:
                            npeaks = np.sum(new_neighbor_peaks)
                            for mq_val in mq_subarr:
                                c_neighbor_peaks = np.abs(mq_val - mq_arr) < 1.75
                                new_neighbor_peaks = np.logical_or(new_neighbor_peaks, c_neighbor_peaks)
                            mq_subarr = mq_arr[new_neighbor_peaks]
                            if np.sum(new_neighbor_peaks) == npeaks:
                                more = False

                        for mq_val in mq_subarr:
                            mq_done.append(mq_val)
                        amp_subarr = []
                        sigma_subarr = []
                        n_subarr = []
                        for mass in mq_subarr:
                            peak_pos = np.argmin(np.abs(plot_x - mass))
                            amp_subarr.append(plot_y[peak_pos]*fit_scale)
                            sigma_subarr.append(0.3)
                            n_subarr.append(4)

                        N_param = len(mq_subarr)
                        params0 = amp_subarr + list(mq_subarr) + sigma_subarr + n_subarr

                        amp_bounds_u = np.ones_like(amp_subarr) * 1e-4 * fit_scale
                        amp_bounds_l = np.ones_like(amp_subarr) * 1e-10 * fit_scale

                        mq_bounds_u = np.array(mq_subarr) + .5
                        mq_bounds_l = np.array(mq_subarr) - .5

                        sigma_bounds_u = np.ones_like(sigma_subarr) * 0.5
                        sigma_bounds_l = np.ones_like(sigma_subarr) * 0.15

                        n_bounds_u = np.ones_like(n_subarr) * 6
                        n_bounds_l = np.ones_like(n_subarr) * 2

                        bounds = (list(amp_bounds_l) + list(mq_bounds_l) \
                                    + list(sigma_bounds_l) + list(n_bounds_l), \
                                  list(amp_bounds_u) + list(mq_bounds_u) \
                                    + list(sigma_bounds_u) + list(n_bounds_u))

                        rel_to_max = plot_y / np.max(plot_y)
                        rel_to_max = plot_y - plot_y
                        fit_sigma = (1.0 - rel_to_max + 1e-2)**(1.0)

                        print()
                        print(mq_subarr)
                        print(params0)
                        popt, pcov = opti.curve_fit(lambda x, *params: fit_wrapper(x, N_param, params), \
                                                plot_x, plot_y * fit_scale, p0 = params0, maxfev=100000, \
                                                bounds=bounds, sigma=fit_sigma, verbose=verbose)
                        print(popt)

                        fit_amps += list(popt[:N_param])
                        fit_mus += list(popt[N_param:2*N_param])
                        fit_sigmas += list(popt[2*N_param:3*N_param])
                        fit_ns += list(popt[3*N_param:4*N_param])
                        stop = time.time()
                        print('many ngauss fit: ', stop-start)

        print('Done!')
        sys.stdout.flush()

        popt_all = fit_amps + fit_mus + fit_sigmas + fit_ns
        N_param_all = len(fit_amps)

        pressure_fun = lambda x: fit_wrapper(x, N_param_all, popt_all)

        mq_arr = np.array(mq_arr)
        gas_pressures = {}
        for gas in gases_to_extract:
            isotope_dict = gases[gas]

            isotopes = list(isotope_dict.keys())
            isotopes.sort()

            isotope_pressures = {}
            for isotope in isotopes:
                m0 = isotope_dict[isotope]

                ion_dict = ions[isotope]
                d_ions = list(ion_dict.keys())
                d_ions.sort()

                isotope_pressure = 0
                isotope_pressure_var = 0
                for ion in d_ions:
                    if ion in ions_to_ignore:
                        continue
                    mq = ion_dict[ion]
                    ind = np.argmin(np.abs(np.array(fit_mus) - mq))
                    mq_mu = fit_mus[ind]
                    peak_ind = np.argmin(np.abs(plot_x - mq_mu))

                    overall_fit_pressure = pressure_fun(mq_mu) / fit_scale
                    isotope_fit_pressure = fit_amps[ind] / fit_scale

                    isotope_pressure += isotope_fit_pressure
                    isotope_pressure_var += np.abs(overall_fit_pressure - plot_y[peak_ind])**2 \
                                                + plot_errs[peak_ind]**2

                isotope_pressures[isotope] = (isotope_pressure, np.sqrt(isotope_pressure_var))

            gas_pressures[gas] = isotope_pressures



        if plot_extraction:

            ax_ex.set_ylim(1e-9, 2*np.max(plot_y))
            ax_ex.set_xlim(0,int(np.max(plot_x)))

            ax_ex2.set_ylim(0, 1.2*np.max(plot_y))
            ax_ex2.set_xlim(0,int(np.max(plot_x)))

            if save_fig:
                fig_ex.savefig(fig_base+'log.png')
                fig_ex2.savefig(fig_base+'linear.png')

            ax_ex.plot(plot_x, fit_wrapper(plot_x, N_param_all, popt_all) / fit_scale, \
                        '--', color='r', lw=2)
            ax_ex2.plot(plot_x, fit_wrapper(plot_x, N_param_all, popt_all) / fit_scale, \
                        '--', color='r', lw=2)

            if save_fig:
                fig_ex.savefig(fig_base+'extraction-log.png')
                fig_ex2.savefig(fig_base+'extraction-linear.png')
            if show:
                plt.show()
            fig_ex.clf()
            fig_ex2.clf()

    elif not len(gases_to_extract):
        gas_pressures = {}



    return {'mass_vec': plot_x, 'partial_pressures': plot_y, \
            'errs': plot_errs, 'pressure': pressure, 'gas_pressures': gas_pressures}






def get_leak_m0(main_gas, gas_pp1, gas_pp2, remove_neg_diffs=False, sens_err=0.1):

    extracted_gas_keys = list(gas_pp1.keys())
    diffs = {}
    max_diff = 0

    for key in extracted_gas_keys:

        diffs[key] = {}
        isotopes = list(gas_pp1[key].keys())
        isotopes.sort()
        for isotope in isotopes:
            diff = gas_pp2[key][isotope][0] - gas_pp1[key][isotope][0]
            diff_err = np.sqrt(gas_pp2[key][isotope][1]**2 + gas_pp1[key][isotope][1]**2)
            if diff < 0 and remove_neg_diffs:
                diff = 0
                diff_err = 0.0
            diffs[key][isotope] = (diff, diff_err)
            if diff > max_diff:
                max_diff = diff
                max_err = diff_err
                max_gas = isotope
                max_species = key
                max_mass = gases[key][isotope]

    print()
    print()

    str_to_print = 'Gas Fraction by amu, normalized to %s...' % max_gas
    print(str_to_print)

    # Normalize and compute "total" change in pressure to determine mole fraction
    diffs_norm_cal = {}
    total = 0
    total_var = 0
    total_low = 0
    total_high = 0
    for key in extracted_gas_keys:
        sens_fac = rga_sensitivities[max_species] / rga_sensitivities[key]
        sens_fac *= pumping_speed[key] / pumping_speed[max_species]
        diffs_norm_cal[key] = {}
        isotopes = list(diffs[key].keys())
        isotopes.sort()
        for isotope in isotopes:
            m0 = gases[key][isotope]
            diff = np.abs(diffs[key][isotope][0])
            diff_err = np.abs(diffs[key][isotope][1])
            if diff == 0:
                diff_norm_cal = 0
                diff_norm_cal_err = 0
            else:
                diff_norm_cal = sens_fac * diff / max_diff
                diff_norm_cal_err = diff_norm_cal * \
                        np.sqrt( (diff_err / diff)**2 + (max_err / max_diff)**2 + sens_err**2 )
            total += diff_norm_cal
            total_var += diff_norm_cal**2 * diff_norm_cal_err**2
            diffs_norm_cal[key][isotope] = (diff_norm_cal, diff_norm_cal_err)


    # calculate effective m0
    total_err = np.sqrt(total_var)
    m0_raw = 0

    m0_eff = 0
    m0_eff_var = 0

    m0_eff_sqrt = 0
    m0_eff_sqrt_var = 0

    m0_eff_sqrt_INV = 0
    m0_eff_sqrt_INV_var = 0

    for key in extracted_gas_keys:
        isotopes = list(diffs_norm_cal[key].keys())
        for isotope in isotopes:
            diff = np.abs(diffs_norm_cal[key][isotope][0])
            diff_err = np.abs(diffs_norm_cal[key][isotope][1])
            if diff == 0:
                continue
            fraction = np.abs(diff) / total
            fraction_err = (np.abs((total-diff)/total) * \
                            fraction * np.sqrt((diff_err / diff)**2 + (total_err / total)**2))[0]
            m0 = gases[key][isotope]

            # m0_eff += fraction * m0
            m0_eff_sqrt += fraction * np.sqrt(m0)
            # m0_eff_sqrt_INV += fraction / np.sqrt(m0)

            if key != main_gas:
                # m0_eff_var += fraction_err**2 * m0**2
                m0_eff_sqrt_var += fraction_err**2 * m0
                # m0_eff_sqrt_INV_var += fraction_err**2 / m0

            print(isotope, fraction, fraction_err)          

    # m0_eff_err = np.sqrt(m0_eff_var)

    m0_eff_2 = m0_eff_sqrt**2
    m0_eff_err_2 = np.sqrt(m0_eff_2**2 * (4.0 * m0_eff_sqrt_var / m0_eff_sqrt**2))

    # m0_eff_3 = 1.0 / m0_eff_sqrt_INV**2
    # m0_eff_err_3 = np.sqrt(m0_eff_3**2 * (4.0 * m0_eff_sqrt_INV_var / m0_eff_sqrt_INV**2))

    print()
    print('Effective mass for %s: %0.2f +- %0.2f' % (max_species, m0_eff_2, m0_eff_err_2))
    # print '                     : %0.2f +- %0.2f' % (m0_eff_2, m0_eff_err_2)
    # print '                     : %0.2f +- %0.2f' % (m0_eff_3, m0_eff_err_3)

    return np.array([m0_eff_2, m0_eff_err_2])#, m0_eff_2, m0_eff_err_2, m0_eff_3, m0_eff_err_3]

