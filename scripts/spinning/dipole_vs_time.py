import os, sys
import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu

import scipy.optimize as opti

import itertools

from iminuit import Minuit, describe

plt.rcParams.update({'font.size': 14})






#################################################
#################################################
####                                         ####
####   THIS SCRIPT DOESN'T FUNCTION YET. I   ####
####   WANTED TO DISENTANGLE THE TWO         ####
####   DISTINCT USE CASES OF THE SIDEBAND    ####
####   FITTING: EITHER GETTING A DIPOLE      ####
####   FOR SAVING TO THE CALIBRATIONS DIR    ####
####   OR OBSERVING FLUCTUATIONS IN THE      ####
####   MEASURED DIPOLE MOMENT. THIS ONE IS   ####
####   FOR THE LATTER OBJECTIVE.             ####
####                                         ####
#################################################
#################################################


print("This script doesn't work yet, fix it!")
sys.exit()







dipole_savebase = '/data/old_trap_processed/calibrations/dipoles/'
save_dipole = True

savefig = True


#one_path = True
one_path = False

remove_outliers = True

include_field_prior = True
# field_prior = 0.0
field_prior = 100.0  # V/m (set to infinity to turn off prior)

# hard_lower_limit = 0.0
hard_lower_limit = 3000.0  # rad/s


polarizability = 0.0
# polarizability = 1e-3    # (C * m) / (V / m)




# ### Old path definition(s) for spinning rotor paper

# #date = '20190626'
# #date = '20190905'
# date = '20191017'
# #gases = ['He', 'N2', 'Ar', 'Kr', 'Xe', 'SF6']
# #gases = ['He', 'N2']
# gases = ['He', 'N2']
# inds = [1, 2, 3]

# # date = '20190905'
# # gases = ['He', 'N2']
# # inds = [1, 2, 3]

# #paths = [base_path]

# base_plot_path = '/home/cblakemore/plots/{:s}/pramp/'.format(date)

# # base_path = '/processed_data/spinning/wobble/20190626/'
# # base_path = '/processed_data/spinning/wobble/20190626/long_wobble/'
# base_path = '/data/old_trap_processed/spinning/wobble/{:s}/'.format(date)

# baselen = len(base_path)

# # gas = 'N2'
# # paths = [base_path + '%s_pramp_1/' % gas, \
# #          base_path + '%s_pramp_2/' % gas, \
# #          base_path + '%s_pramp_3/' % gas, \
# #         ]

# path_dict = {}
# for meas in itertools.product(gases, inds):
#     gas, pramp_ind = meas
#     if gas not in list(path_dict.keys()):
#         path_dict[gas] = []
#     path_dict[gas].append(base_path + '{:s}_pramp_{:d}/'.format(gas, pramp_ind))





### New path definition for basic dipole data. Still conforms to above
### because I haven't changed the underlying script

base_path = '/data/old_trap_processed/spinning/wobble/'

date = '20200727'
meas_list = [\
             'wobble_fast', \
             # 'wobble_large-step_many-files', \
             # 'wobble_slow', \
             # 'wobble_slow_2', \
             # 'wobble_slow_after'
            ]

base_plot_path = '/home/cblakemore/plots/{:s}/'.format(date)
savefig = True


# date = '20200924'
# meas_list = [\
#              '', \
#             ]

paths = []
for meas in meas_list:
    path = os.path.join(base_path, date, meas)
    paths.append(path)
npaths = len(paths)

keys = [date]
path_dict = {date: paths}







if savefig:
    bu.make_all_pardirs(base_plot_path)


Ibead = bu.get_Ibead(date=date, verbose=True)

def sqrt(x, A, x0, b):
    return A * np.sqrt(x-x0) + b


for key in keys:
    paths = path_dict[key]
    all_fig, all_ax = plt.subplots(1,1,figsize=(7,4))
    colors = bu.get_color_map(len(paths))

    for pathind, path in enumerate(paths):
        fig, ax = plt.subplots(1,1,figsize=(7,4))

        parts = path.split('/')
        if not len(parts[-1]):
            meas = parts[-2]
        else:
            meas = parts[-1]

        # color = 'C' + str(pathind)
        color = colors[pathind]

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
            ndata = len(field_strength)

            sorter = np.argsort(field_strength)
            field_strength = field_strength[sorter]
            field_err = field_err[sorter]
            wobble_freq = wobble_freq[sorter]
            wobble_err = wobble_err[sorter]

            wobble_freq *= (2 * np.pi)
            wobble_err *= (2 * np.pi)

            inds = wobble_freq > hard_lower_limit
            if remove_outliers:
                for i in range(ndata):

                    if (i == 0) or (i == ndata - 1):
                        continue

                    cond1 = (np.abs(wobble_freq[i] - wobble_freq[i+1]) / 
                                                wobble_freq[i+1]) > 0.2
                    cond2 =  (np.abs(wobble_freq[i] - wobble_freq[i-1]) / 
                                                wobble_freq[i-1]) > 0.2
                    if cond1 and cond2:
                        inds[i] = False

            field_strength = field_strength[inds]
            field_err = field_err[inds]
            wobble_freq = wobble_freq[inds]
            wobble_err = wobble_err[inds]

            def fitfun(x, A, x0):
                return sqrt(x, A, x0, 0)

            popt, pcov = opti.curve_fit(fitfun, field_strength, wobble_freq, p0=[10,0])

            #wobble_err *= 40
            if polarizability:
                print("Polarizability fitting hasn't been implemented yet")
                sys.exit()
                # def cost(d, alpha, x0, xscale=1.0):
                #     resid = np.abs(sqrt(xscale * field_strength, A, x0, 0) - wobble_freq)
                #     norm = 1. / (len(field_strength) - 2)
                #     tot_var = wobble_err**2 + wobble_freq**2 * (field_err / field_strength)**2
                #     return norm * np.sum( resid**2 / tot_var) + x0**2 / actual_field_prior**2

                # m=Minuit(cost,
                #          d = popt[0], # set start parameter
                #          #fix_A = "True", # you can also fix it
                #          #limit_pA = (0.0, 10000.0),
                #          alpha = 0.0,
                #          x0 = popt[1],
                #          xscale = 1.0,
                #          fix_xscale = "True",
                #          pedantic=False)
                # m.errordef = 1.0
                # m.print_level = 0
                # m.migrad(ncall=500000)
                # minos = m.minos()

                # m_upper=Minuit(cost,
                #          A = popt[0], # set start parameter
                #          #fix_A = "True", # you can also fix it
                #          #limit_pA = (0.0, 10000.0),
                #          x0 = popt[1],
                #          xscale = 1.01,
                #          fix_xscale = "True",
                #          errordef = 1,
                #          print_level = 0, 
                #          pedantic=False)   
                # m_upper.migrad()         
                # m_lower=Minuit(cost,
                #          A = popt[0], # set start parameter
                #          #fix_A = "True", # you can also fix it
                #          #limit_pA = (0.0, 10000.0),
                #          x0 = popt[1],
                #          xscale = 0.99,
                #          fix_xscale = "True",
                #          errordef = 1,
                #          print_level = 0, 
                #          pedantic=False)
                # m_lower.migrad()

            else:
                if include_field_prior:
                    actual_field_prior = np.sqrt(np.sum(field_err**2) + field_prior**2)
                    def cost(A, x0, xscale=1.0):
                        resid = np.abs(sqrt(xscale * field_strength, A, x0, 0) - wobble_freq)
                        norm = 1. / (len(field_strength) - 2)
                        tot_var = wobble_err**2 + wobble_freq**2 * (field_err / field_strength)**2
                        return norm * np.sum( resid**2 / tot_var) + x0**2 / actual_field_prior**2
                else:
                    def cost(A, x0, xscale=1.0):
                        resid = np.abs(sqrt(xscale * field_strength, A, x0, 0) - wobble_freq)
                        norm = 1. / (len(field_strength) - 2)
                        tot_var = wobble_err**2 + wobble_freq**2 * (field_err / field_strength)**2
                        return norm * np.sum( resid**2 / tot_var)

                m=Minuit(cost,
                         A = popt[0], # set start parameter
                         #fix_A = "True", # you can also fix it
                         #limit_pA = (0.0, 10000.0),
                         x0 = popt[1],
                         xscale = 1.0,
                         fix_xscale = "True",
                         errordef = 1,
                         print_level = 0, 
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


            ax.errorbar(field_strength*1e-3, wobble_freq, color=color,\
                         yerr=wobble_err)
            if one_path:
                popt_arr.append(popt)
                pcov_arr.append(pcov)
                plot_x = np.linspace(0, np.max(field_strength), 100)
                plot_x[0] = 1.0e-9 * plot_x[1]
                plot_y = sqrt(plot_x, A_arr[-1], x0_arr[-1], 0)

                plt.plot(plot_x*1e-3, plot_y, '--', lw=2, color=color)

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
            # A_syserr = np.sqrt( 1.0 / np.sum( 1.0 / A_syserr_arr**2) )
            A_syserr = np.mean(A_syserr_arr)

            x0_val = np.sum( x0_arr / x0_err_arr**2) / np.sum( 1.0 / x0_err_arr**2 )
            x0_err = np.sqrt( 1.0 / np.sum( 1.0 / x0_err_arr**2) )

            plot_x = np.linspace(0, max_field, 100)
            plot_x[0] = 1.0e-9 * plot_x[1]
            plot_y = sqrt(plot_x, A_val, x0_val, 0)

            d = A_val**2 * Ibead['val']
            d_sterr = d * np.sqrt( (2.0*A_sterr/A_val)**2 \
                                    + (Ibead['sterr']/Ibead['val'])**2 )
            d_syserr = d * np.sqrt( (2.0*A_syserr/A_val)**2 \
                                    + (Ibead['syserr']/Ibead['val'])**2 )

            # print(A_sterr / A_val, Ibead['sterr'] / Ibead['val'])

            d_scaled = d * (1.0 / 1.602e-19) * 1e6
            d_sterr_scaled = d_sterr * (1.0 / 1.602e-19) * 1e6
            d_syserr_scaled = d_syserr * (1.0 / 1.602e-19) * 1e6

            label = '${:0.1f}\\pm{:0.1f}(st)\\pm{:0.1f}(sys)\\,\\,e\\cdot\\mu$m'\
                            .format(d_scaled, d_sterr_scaled, d_syserr_scaled)

            ax.plot(plot_x*1e-3, plot_y, '--', lw=2, color=color, label=label)

            if save_dipole:        
                dipole_filename = os.path.join(dipole_savebase, \
                                       '{:s}_{:s}.dipole'.format(date, meas))
                print('Saving dipole to:')
                print('    {:s}'.format(dipole_filename))
                print()
                np.save(open(dipole_filename, 'wb'), [d, d_sterr, d_syserr])

        # elif one_path:
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
        ax.set_ylabel('$\\omega_{\\phi}$ [rad/s]')

        ax.legend(fontsize=12)
        plt.tight_layout()

        plot_name = '{:s}_{:s}_dipole_meas.svg'.format(date, meas)
        plot_save_path = os.path.join(base_plot_path, plot_name)
        if savefig:
            print()
            print('Saving figure to:')
            print('    {:s}'.format(plot_save_path))
            fig.savefig(plot_save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

