import os
import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu

import scipy.optimize as opti

import itertools

from iminuit import Minuit, describe

plt.rcParams.update({'font.size': 14})


#date = '20190626'
#date = '20190905'
date = '20191017'
#gases = ['He', 'N2', 'Ar', 'Kr', 'Xe', 'SF6']
#gases = ['He', 'N2']
gases = ['He', 'N2']
inds = [1, 2, 3]

# date = '20190905'
# gases = ['He', 'N2']
# inds = [1, 2, 3]


dipole_savebase = '/data/old_trap_processed/calibrations/dipoles/'


# base_path = '/processed_data/spinning/wobble/20190626/'
# base_path = '/processed_data/spinning/wobble/20190626/long_wobble/'
base_path = '/data/old_trap_processed/spinning/wobble/{:s}/'.format(date)

base_plot_path = '/home/cblakemore/plots/{:s}/pramp/'.format(date)
bu.make_all_pardirs(base_plot_path)
savefig = False

baselen = len(base_path)

# gas = 'N2'
# paths = [base_path + '%s_pramp_1/' % gas, \
#          base_path + '%s_pramp_2/' % gas, \
#          base_path + '%s_pramp_3/' % gas, \
#         ]

path_dict = {}
for meas in itertools.product(gases, inds):
    gas, pramp_ind = meas
    if gas not in list(path_dict.keys()):
        path_dict[gas] = []
    path_dict[gas].append(base_path + '{:s}_pramp_{:d}/'.format(gas, pramp_ind))


#paths = [base_path]
#one_path = True
one_path = False

base_path = '/data/old_trap_processed/spinning/wobble/'

date = '20200727'
# meas = 'wobble_slow_2'

# date = '20200924'
# meas = 'dipole_meas/initial'

meas_list = [\
             'wobble_fast', \
             'wobble_large-step_many-files', \
             'wobble_slow', \
             'wobble_slow_2', \
             'wobble_slow_after'
            ]

paths = []
for meas in meas_list:
    path = os.path.join(base_path, date, meas)
    paths.append(path)
npaths = len(paths)

gases = ['XX']
path_dict = {'XX': paths}

print(paths)
input()



Ibead = bu.get_Ibead(date=date, verbose=True)

def sqrt(x, A, x0, b):
    return A * np.sqrt(x-x0) + b


for gas in gases:
    fig, ax = plt.subplots(1,1)
    paths = path_dict[gas]
    for pathind, path in enumerate(paths):

        parts = path.split('/')
        if not len(parts[-1]):
            meas = parts[-2]
        else:
            meas = parts[-1]

        dipole_filename = os.path.join(dipole_savebase, \
                                       '{:s}_{:s}.dipole'.format(date, meas))
        print(dipole_filename)
        input()

        color = 'C' + str(pathind)

        files, lengths = bu.find_all_fnames(path, ext='.npy')
        if one_path:
            colors = bu.get_color_map(len(files), cmap='inferno')

        popt_arr = []
        pcov_arr = []
        max_field = 0

        A_arr = []
        A_sterr_arr = []
        A_syserr_arr = []

        x0_arr = []
        x0_err_arr = []

        for fileind, file in enumerate(files):
            if one_path:
                color = colors[fileind]
            field_strength, field_err, wobble_freq, wobble_err = np.load(file)

            sorter = np.argsort(field_strength)
            field_strength = field_strength[sorter]
            field_err = field_err[sorter]
            wobble_freq = wobble_freq[sorter]
            wobble_err = wobble_err[sorter]

            wobble_freq *= (2 * np.pi)
            wobble_err *= (2 * np.pi)
            # plt.errorbar(field_strength, wobble_freq, xerr=field_err, yerr=wobble_err)
            # plt.show()



            #field_strength = 100.0 * field_strength * 2.0 

            # try:
            def fitfun(x, A, x0):
                return sqrt(x, A, x0, 0)

            popt, pcov = opti.curve_fit(fitfun, field_strength, wobble_freq, p0=[10,0])

            #wobble_err *= 40

            def cost(A, x0, xscale=1.0):
                resid = np.abs(sqrt(xscale * field_strength, A, x0, 0) - wobble_freq)
                norm = 1. / (len(field_strength) - 1)
                tot_var = wobble_err**2 #+ wobble_freq**2 * (field_err / field_strength)**2
                return norm * np.sum( resid**2 / tot_var)

            m=Minuit(cost,
                     A = popt[0], # set start parameter
                     #fix_A = "True", # you can also fix it
                     #limit_pA = (0.0, 10000.0),
                     x0 = popt[1],
                     xscale = 1.0,
                     fix_xscale = "True",
                     errordef = 1,
                     print_level = 1, 
                     pedantic=False)
            m.migrad(ncall=500000)
            minos = m.minos()

            m_upper=Minuit(cost,
                     A = popt[0], # set start parameter
                     #fix_A = "True", # you can also fix it
                     #limit_pA = (0.0, 10000.0),
                     x0 = popt[1],
                     xscale = 1.01,
                     fix_xscale = "True",
                     errordef = 1,
                     print_level = 0, 
                     pedantic=False)   
            m_upper.migrad()         
            m_lower=Minuit(cost,
                     A = popt[0], # set start parameter
                     #fix_A = "True", # you can also fix it
                     #limit_pA = (0.0, 10000.0),
                     x0 = popt[1],
                     xscale = 0.99,
                     fix_xscale = "True",
                     errordef = 1,
                     print_level = 0, 
                     pedantic=False)
            m_lower.migrad()

            bootstrap_errs = np.abs(m.values['A'] - \
                                    np.array([m_upper.values['A'], m_lower.values['A']]))
            A_syserr_arr.append(np.mean(bootstrap_errs))

            A_arr.append(minos['A']['min'])
            A_sterr_arr.append(np.mean(np.abs([minos['A']['upper'], minos['A']['lower']])))

            x0_arr.append(minos['x0']['min'])
            x0_err_arr.append(np.mean(np.abs([minos['x0']['upper'], minos['x0']['lower']])))


            # except:
            #     fig2 = plt.figure(2)
            #     plt.plot(field_strength, wobble_freq)
            #     fig2.show()
            #     plt.figure(1)
            #     continue

            # popt_arr.append(popt)
            # pcov_arr.append(pcov)

            ax.errorbar(field_strength*1e-3, wobble_freq, color=color,\
                         yerr=wobble_err)
            # if one_path:
            #     plot_x = np.linspace(0, np.max(field_strength), 100)
            #     plot_x[0] = 1.0e-9 * plot_x[1]
            #     plot_y = sqrt(plot_x, A_arr[-1], x0_arr[-1], 0)

            #     plt.plot(plot_x*1e-3, plot_y, '--', lw=2, color=color)

            max_field = np.max([np.max(field_strength), max_field])

        A_arr = np.array(A_arr)
        A_sterr_arr = np.array(A_sterr_arr)
        A_syserr_arr = np.array(A_syserr_arr)

        x0_arr = np.array(x0_arr)
        x0_err_arr = np.array(x0_err_arr)


        if not one_path:
            A_val = np.sum( A_arr / (A_sterr_arr**2 + A_syserr_arr**2)) / \
                        np.sum( 1.0 / (A_sterr_arr**2 + A_syserr_arr**2))
            A_sterr = np.sqrt( 1.0 / np.sum( 1.0 / A_sterr_arr**2) )
            A_syserr = np.sqrt( 1.0 / np.sum( 1.0 / A_syserr_arr**2) )

            x0_val = np.sum( x0_arr / x0_err_arr**2) / np.sum( 1.0 / x0_err_arr**2 )
            x0_err = np.sqrt( 1.0 / np.sum( 1.0 / x0_err_arr**2) )

            plot_x = np.linspace(0, max_field, 100)
            plot_x[0] = 1.0e-9 * plot_x[1]
            plot_y = sqrt(plot_x, A_val, x0_val, 0)

            # 1e-3 to account for 
            d = A_val**2 * Ibead['val']
            d_sterr = d * np.sqrt( (A_sterr/A_val)**2 + (Ibead['sterr']/Ibead['val'])**2 )
            d_syserr = d * np.sqrt( (A_syserr/A_val)**2 + (Ibead['syserr']/Ibead['val'])**2 )

            print(A_sterr / A_val, Ibead['sterr'] / Ibead['val'])

            d_scaled = d * (1.0 / 1.602e-19) * 1e6
            d_sterr_scaled = d_sterr * (1.0 / 1.602e-19) * 1e6
            d_syserr_scaled = d_syserr * (1.0 / 1.602e-19) * 1e6
            # if len(labels):
            #     prefix = labels[pathind]
            # else:
            #     prefix = ''

            label = '${:0.1f} \\pm {:0.1f} (st) \\pm {:0.1f} (sys) \\, \\, e \\cdot \\mu$m'\
                            .format(d_scaled, d_sterr_scaled, d_syserr_scaled)

            ax.plot(plot_x*1e-3, plot_y, '--', lw=2, color=color, label=label)

            np.save(open(dipole_filename, 'wb'), [d, d_sterr, d_syserr])

        # if one_path:
        #     d_vec = (np.array(popt_arr)[:,0])**2 * Ibead['val']
        #     d_vec_scaled = d_vec * (1.0 / 1.602e-19) * 1e6

        #     time_vec = np.linspace(0, len(d_vec)-1, len(d_vec)) * 990

        #     plt.figure(3)
        #     plt.plot(time_vec*(1./3600), d_vec_scaled)
        #     plt.xlabel('Time [hrs]')
        #     plt.ylabel('Dipole moment [$e \cdot \mu m$]')
        #     plt.tight_layout()

        #     np.save(open(dipole_filename, 'wb'), [np.mean(d_vec), np.std(d_vec)])

    ax.set_xlabel('Field [kV/m]')
    ax.set_ylabel('$\omega_{\phi}$ [rad/s]')

    ax.legend(fontsize=12)
    plt.tight_layout()

    plot_name = '%s_wobble.png' % gas
    plot_save_path = os.path.join(base_plot_path, plot_name)
    plot_name_2 = '%s_wobble.svg' % gas
    plot_save_path_2 = os.path.join(base_plot_path, plot_name)
    if savefig:
        fig.savefig(plot_save_path)
        fig.savefig(plot_save_path_2)

    plt.show()

