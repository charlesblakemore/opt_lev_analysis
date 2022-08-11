import os, sys, re
import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu

import scipy.optimize as opti

import itertools

from iminuit import Minuit, describe

plt.rcParams.update({'font.size': 14})



dipole_savebase = '/data/old_trap_processed/calibrations/dipoles/'
save_dipole = True

savefig = True


remove_outliers = True

include_field_prior = True
# field_prior = 0.0
field_prior = 10.0  # V/m (set to infinity to turn off prior)

# hard_lower_limit = 0.0
hard_lower_limit = 3000.0  # rad/s


polarizability = 0.0
# polarizability = 1e-3    # (C * m) / (V / m)



### New path definition for basic dipole data. Still conforms to above
### because I haven't changed the underlying script

base_path = '/data/old_trap_processed/spinning/wobble/'

date = '20200727'
meas_list = [\
             'wobble_fast', \
             # 'wobble_large-step_many-files', \
             'wobble_slow', \
             'wobble_slow_2', \
             # 'wobble_slow_after'
            ]

# date = '20200924'
# meas_list = [\
#              'dipole_meas/initial', \
#             ]


# date = '20201030'
# meas_list = [\
#              'dipole_meas/initial', \
#              'dipole_meas/final', \
#             ]

base_plot_path = '/home/cblakemore/plots/{:s}/'.format(date)
savefig = False
show = False

marker_alpha = 0.75
marker_zorder = 5

line_alpha = 1.0
line_zorder = 4


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
    date_fig, date_ax = plt.subplots(1,1,figsize=(7,4))
    meas_colors = bu.get_colormap(len(paths), cmap='plasma')

    date_max_field = 0

    for pathind, path in enumerate(paths):
        meas_fig, meas_ax = plt.subplots(1,1,figsize=(7,4))

        parts = path.split('/')
        if not len(parts[-1]):
            meas = parts[-2]
        else:
            meas = parts[-1]

        files, lengths = bu.find_all_fnames(path, ext='.npy')
        wobble_colors = bu.get_colormap(len(files), cmap='viridis')

        popt_arr = []
        pcov_arr = []
        meas_max_field = 0

        A_arr = []
        A_stunc_arr = []
        A_sysunc_arr = []

        x0_arr = []
        x0_unc_arr = []

        for fileind, file in enumerate(files):
            fig, ax = plt.subplots(1,1,figsize=(7,4))

            wobble_name = os.path.basename(file).split('.')[0]
            wobble_ind = re.search(r"\d+", wobble_name)[0]

            field_strength, field_unc, wobble_freq, wobble_unc = np.load(file)
            ndata = len(field_strength)

            sorter = np.argsort(field_strength)
            field_strength = field_strength[sorter]
            field_unc = field_unc[sorter]
            wobble_freq = wobble_freq[sorter]
            wobble_unc = wobble_unc[sorter]

            wobble_freq *= (2 * np.pi)
            wobble_unc *= (2 * np.pi)

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
            field_unc = field_unc[inds]
            wobble_freq = wobble_freq[inds]
            wobble_unc = wobble_unc[inds]

            def fitfun(x, A, x0):
                return sqrt(x, A, x0, 0)

            ### Do a quick "pre-fit" to help Minuit converge later
            try:
                popt, pcov = opti.curve_fit(fitfun, field_strength, \
                                            wobble_freq, p0=[10,0], maxfev=10000)
            except:
                plt.figure()
                plt.plot(field_strength, wobble_freq)
                plt.show()

                input()

                sys.exit()

            #wobble_unc *= 40
            if polarizability:
                print("Polarizability fitting hasn't been implemented yet")
                sys.exit()
                # def cost(d, alpha, x0, xscale=1.0):
                #     resid = np.abs(sqrt(xscale * field_strength, A, x0, 0) - wobble_freq)
                #     norm = 1. / (len(field_strength) - 2)
                #     tot_var = wobble_unc**2 + wobble_freq**2 * (field_unc / field_strength)**2
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
                    actual_field_prior = np.sqrt(np.sum(field_unc**2) + field_prior**2)
                    def cost(A, x0, xscale=1.0):
                        resid = np.abs(sqrt(xscale * field_strength, A, x0, 0) - wobble_freq)
                        norm = 1. / (len(field_strength) - 2)
                        tot_var = wobble_unc**2 + wobble_freq**2 * (field_unc / field_strength)**2
                        return norm * np.sum( resid**2 / tot_var) + x0**2 / actual_field_prior**2
                else:
                    def cost(A, x0, xscale=1.0):
                        resid = np.abs(sqrt(xscale * field_strength, A, x0, 0) - wobble_freq)
                        norm = 1. / (len(field_strength) - 2)
                        tot_var = wobble_unc**2 + wobble_freq**2 * (field_unc / field_strength)**2
                        return norm * np.sum( resid**2 / tot_var)

                m=Minuit(cost,
                         A = popt[0], # set start parameter
                         #fix_A = "True", # you can also fix it
                         #limit_pA = (0.0, 10000.0),
                         x0 = popt[1],
                         xscale = 1.0)
                m.fixed['xscale'] = True
                m.errordef = 1.0
                m.print_level = 0
                m.migrad(ncall=500000)
                m.minos()

                m_upper=Minuit(cost,
                         A = popt[0], # set start parameter
                         #fix_A = "True", # you can also fix it
                         #limit_pA = (0.0, 10000.0),
                         x0 = popt[1],
                         xscale = 1.01)  
                m_upper.fixed['xscale'] = True 
                m_upper.errordef = 1.0
                m_upper.print_level = 0
                m_upper.migrad()

                m_lower=Minuit(cost,
                         A = popt[0], # set start parameter
                         #fix_A = "True", # you can also fix it
                         #limit_pA = (0.0, 10000.0),
                         x0 = popt[1],
                         xscale = 0.99)
                m_lower.fixed['xscale'] = True
                m_lower.errordef = 1.0
                m_lower.print_level = 0
                m_lower.migrad()

            bootstrap_uncs = np.abs(m.values['A'] - \
                                    np.array([m_upper.values['A'], m_lower.values['A']]))

            ### Get the answers from Minuit
            # A = minos['A']['min']
            A = m.values['A']
            A_sysunc = np.mean(bootstrap_uncs)
            # A_stunc = np.mean(np.abs([minos['A']['upper'], minos['A']['lower']]))
            A_stunc = np.mean(np.abs( [m.merrors['A'].upper, \
                                       m.merrors['A'].lower] ))
            # x0 = minos['x0']['min']
            x0 = m.values['x0']
            # x0_unc = np.mean(np.abs([minos['x0']['upper'], minos['x0']['lower']]))
            x0_unc = np.mean(np.abs( [m.merrors['x0'].upper, \
                                      m.merrors['x0'].lower] ))

            ### Add the answers to the array for all the data
            A_arr.append(A)
            A_sysunc_arr.append(A_sysunc)
            A_stunc_arr.append(A_stunc)
            x0_arr.append(x0)
            x0_unc_arr.append(x0_unc)

            ### Compute the dipole moment
            d = A**2 * Ibead['val']
            d_stunc = d * np.sqrt( (2.0*A_stunc/A)**2 \
                                    + (Ibead['sterr']/Ibead['val'])**2 )
            d_sysunc = d * np.sqrt( (2.0*A_sysunc/A)**2 \
                                    + (Ibead['syserr']/Ibead['val'])**2 )

            ### Scale it for labeling
            d_scaled = d * (1.0 / 1.602e-19) * 1e6
            d_stunc_scaled = d_stunc * (1.0 / 1.602e-19) * 1e6
            d_sysunc_scaled = d_sysunc * (1.0 / 1.602e-19) * 1e6

            label = '${:0.1f}\\pm{:0.1f}(st)\\pm{:0.1f}(sys)\\,\\,e\\cdot\\mu$m'\
                            .format(d_scaled, d_stunc_scaled, d_sysunc_scaled)

            plot_x = np.linspace(0, np.max(field_strength), 100)
            # plot_x[0] = 1.0e-9 * plot_x[1]
            plot_y = sqrt(plot_x, A, x0, 0)

            ### Plot it
            ax.errorbar(field_strength*1e-3, wobble_freq, color='C0', \
                         mec='none', yerr=wobble_unc, fmt='o', \
                         alpha=marker_alpha, zorder=marker_zorder)
            ax.plot(plot_x*1e-3, plot_y, '--', lw=2, color='red', \
                    alpha=line_alpha, label=label, zorder=line_zorder)        
            ax.set_xlabel('Field [kV/m]')
            ax.set_ylabel('$\\omega_{\\phi}$ [rad/s]')

            ax.set_xlim(0.0, ax.get_xlim()[1])
            ax.set_ylim(0.0, ax.get_ylim()[1])

            ax.grid()
            ax.legend(fontsize=12)
            fig.tight_layout()

            if savefig:
                plot_name = '{:s}_dipole_meas_{:s}_{:s}.svg'\
                                .format(date, meas, wobble_ind)
                plot_save_path = os.path.join(base_plot_path, plot_name)
                print()
                print('Saving figure to:')
                print('    {:s}'.format(plot_save_path))
                fig.savefig(plot_save_path)

            if not show:
                plt.close(fig)

            if save_dipole:        
                dipole_filename = \
                    os.path.join(dipole_savebase, \
                        '{:s}_{:s}_{:s}.dipole'.format(date, meas, wobble_ind))
                print('Saving dipole to:')
                print('    {:s}'.format(dipole_filename))
                print()
                np.save(open(dipole_filename, 'wb'), [d, d_stunc, d_sysunc])

            meas_max_field = np.max([np.max(field_strength), meas_max_field])

            meas_ax.errorbar(field_strength*1e-3, wobble_freq, \
                             color=wobble_colors[fileind], mec='none',\
                             yerr=wobble_unc, fmt='o', \
                             alpha=marker_alpha, zorder=marker_zorder)

            date_ax.errorbar(field_strength*1e-3, wobble_freq, \
                             color=meas_colors[pathind], mec='none',\
                             yerr=wobble_unc, fmt='o', \
                             alpha=marker_alpha, zorder=marker_zorder)

        A_arr = np.array(A_arr)
        A_stunc_arr = np.array(A_stunc_arr)
        A_sysunc_arr = np.array(A_sysunc_arr)

        x0_arr = np.array(x0_arr)
        x0_unc_arr = np.array(x0_unc_arr)

        A_val = np.sum( A_arr / (A_stunc_arr**2 + A_sysunc_arr**2)) / \
                    np.sum( 1.0 / (A_stunc_arr**2 + A_sysunc_arr**2))
        A_stunc = np.sqrt( 1.0 / np.sum( 1.0 / A_stunc_arr**2) )
        # A_sysunc = np.sqrt( 1.0 / np.sum( 1.0 / A_sysunc_arr**2) )
        A_sysunc = np.mean(A_sysunc_arr)

        x0_val = np.sum( x0_arr / x0_unc_arr**2) / np.sum( 1.0 / x0_unc_arr**2 )
        x0_unc = np.sqrt( 1.0 / np.sum( 1.0 / x0_unc_arr**2) )

        meas_plot_x = np.linspace(0, meas_max_field, 100)
        # meas_plot_x[0] = 1.0e-9 * plot_x[1]
        meas_plot_y = sqrt(meas_plot_x, A_val, x0_val, 0)

        d = A_val**2 * Ibead['val']
        d_stunc = d * np.sqrt( (2.0*A_stunc/A_val)**2 \
                                + (Ibead['sterr']/Ibead['val'])**2 )
        d_sysunc = d * np.sqrt( (2.0*A_sysunc/A_val)**2 \
                                + (Ibead['syserr']/Ibead['val'])**2 )

        # print(A_stunc / A_val, Ibead['sterr'] / Ibead['val'])

        d_scaled = d * (1.0 / 1.602e-19) * 1e6
        d_stunc_scaled = d_stunc * (1.0 / 1.602e-19) * 1e6
        d_sysunc_scaled = d_sysunc * (1.0 / 1.602e-19) * 1e6

        label = '${:0.1f}\\pm{:0.1f}(st)\\pm{:0.1f}(sys)\\,\\,e\\cdot\\mu$m'\
                        .format(d_scaled, d_stunc_scaled, d_sysunc_scaled)

        meas_ax.plot(meas_plot_x*1e-3, meas_plot_y, ls='--', lw=2, \
                     color='r', label=label, \
                     alpha=line_alpha, zorder=line_zorder)

        date_ax.plot(meas_plot_x*1e-3, meas_plot_y, ls='--', lw=2, \
                     color=meas_colors[pathind], label=label, \
                     alpha=line_alpha, zorder=line_zorder)

        meas_ax.set_xlabel('Field [kV/m]')
        meas_ax.set_ylabel('$\\omega_{\\phi}$ [rad/s]')

        meas_ax.set_xlim(0.0, meas_ax.get_xlim()[1])
        meas_ax.set_ylim(0.0, meas_ax.get_ylim()[1])

        meas_ax.grid()
        meas_ax.legend(fontsize=12)
        meas_fig.tight_layout()

        if savefig:
            plot_name = '{:s}_dipole_meas_{:s}.svg'.format(date, meas)
            plot_save_path = os.path.join(base_plot_path, plot_name)
            print()
            print('Saving figure to:')
            print('    {:s}'.format(plot_save_path))
            meas_fig.savefig(plot_save_path)

        if not show:
            plt.close(meas_fig)


    date_ax.set_xlabel('Field [kV/m]')
    date_ax.set_ylabel('$\\omega_{\\phi}$ [rad/s]')

    date_ax.set_xlim(0.0, date_ax.get_xlim()[1])
    date_ax.set_ylim(0.0, date_ax.get_ylim()[1])

    date_ax.grid()
    date_ax.legend(fontsize=12)
    date_fig.tight_layout()

    if savefig:
        plot_name = '{:s}_dipole_meas.svg'.format(date)
        plot_save_path = os.path.join(base_plot_path, plot_name)
        print()
        print('Saving figure to:')
        print('    {:s}'.format(plot_save_path))
        date_fig.savefig(plot_save_path)

    if show:
        plt.show()
    else:
        plt.close(date_fig)

