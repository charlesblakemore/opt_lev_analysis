import os
import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu

import scipy.optimize as opti

plt.rcParams.update({'font.size': 14})



#base_path = '/processed_data/spinning/wobble/20190626/'
base_path = '/processed_data/spinning/wobble/20190626/long_wobble/'

base_plot_path = '/home/charles/plots/20190626/pramp/'

baselen = len(base_path)

#labels = ['First: ', 'Repeat: ', 'Next Day: ']
labels = []

gas = 'SF6'
paths = [base_path + '%s_pramp_1/' % gas, \
         base_path + '%s_pramp_2/' % gas, \
         base_path + '%s_pramp_3/' % gas ,\
        ]

paths = [base_path]
one_path = True


mbead = 85.0e-15 # convert picograms to kg
rhobead = 1550.0 # kg/m^3

rbead = ( (mbead / rhobead) / ((4.0/3.0)*np.pi) )**(1.0/3.0)
Ibead = 0.4 * mbead * rbead**2

print rbead
print Ibead

def sqrt(x, A, x0, b):
    return A * np.sqrt(x-x0)# + b


fig = plt.figure(1)
ax = fig.add_subplot(111)
for pathind, path in enumerate(paths):

    dipole_filename = path[:-1] + '.dipole'

    color = 'C' + str(pathind)

    files, lengths = bu.find_all_fnames(path, ext='.npy')
    if one_path:
        colors = bu.get_color_map(len(files), cmap='inferno')

    popt_arr = []
    pcov_arr = []
    max_field = 0

    for fileind, file in enumerate(files):
        if one_path:
            color = colors[fileind]
        field_strength, field_err, wobble_freq, wobble_err = np.load(file)

        wobble_freq *= (2 * np.pi)

        #field_strength = 100.0 * field_strength * 2.0 

        try:
            popt, pcov = opti.curve_fit(sqrt, field_strength, wobble_freq, \
                                        p0=[10,0,0], sigma=wobble_err)
        except:
            fig2 = plt.figure(2)
            plt.plot(field_strength, wobble_freq)
            fig2.show()
            plt.figure(1)
            continue


        print
        print popt
        print

        popt_arr.append(popt)
        pcov_arr.append(pcov)

        ax.errorbar(field_strength*1e-3, wobble_freq, color=color,\
                     yerr=wobble_err)
        if one_path:
            plot_x = np.linspace(0, np.max(field_strength), 100)
            plot_x[0] = 1.0e-9 * plot_x[1]
            plot_y = sqrt(plot_x, *popt)

            plt.plot(plot_x*1e-3, plot_y, '--', lw=2, color=color)

        max_field = np.max([np.max(field_strength), max_field])

    if not one_path:
        popt = np.mean(np.array(popt_arr), axis=0)
        popt_err = np.std(np.array(popt_arr), axis=0)
        pcov = np.mean(np.array(pcov_arr), axis=0)

        plot_x = np.linspace(0, max_field, 100)
        plot_x[0] = 1.0e-9 * plot_x[1]
        plot_y = sqrt(plot_x, *popt)

        # 1e-3 to account for 
        d = (popt[0])**2 * Ibead
        d_err = (popt_err[0])**2 * Ibead

        d_scaled = d * (1.0 / 1.602e-19) * 1e6
        d_err_scaled = d_err * (1.0 / 1.602e-19) * 1e6

        if len(labels):
            prefix = labels[pathind]
        else:
            prefix = ''

        label = prefix + ('%0.1f' % d_scaled) + '$\pm$' \
                    + ('%0.2f' % d_err_scaled) + ' $e \cdot \mu \mathrm{m}$'

        ax.plot(plot_x*1e-3, plot_y, '--', lw=2, color=color, label=label)

        np.save(open(dipole_filename, 'wb'), [d, d_err])

    if one_path:
        d_vec = (np.array(popt_arr)[:,0])**2 * Ibead
        d_vec_scaled = d_vec * (1.0 / 1.602e-19) * 1e6

        time_vec = np.linspace(0, len(d_vec)-1, len(d_vec)) * 990

        plt.figure(3)
        plt.plot(time_vec*(1./3600), d_vec_scaled)
        plt.xlabel('Time [hrs]')
        plt.ylabel('Dipole moment [$e \cdot \mu m$]')
        plt.tight_layout()

        np.save(open(dipole_filename, 'wb'), [np.mean(d_vec), np.std(d_vec)])

plt.figure(1)
ax.set_xlabel('Field [kV/m]')
ax.set_ylabel('$\omega_{\phi}$ [rad/s]')

ax.legend(fontsize=12)
plt.tight_layout()

plot_name = '%s_wobble.png' % gas
plot_save_path = os.path.join(base_plot_path, plot_name)
fig.savefig(plot_save_path)

plt.show()

