import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opti

plt.rcParams.update({'font.size': 14})



base_path = '/processed_data/spinning/wobble/20190626/'
baselen = len(base_path)

files = [base_path+'initial/wobble_0001.npy', \
         base_path+'slow/wobble_0001.npy', \
         base_path+'slow_later/wobble_0001.npy']



labels = ['First: ', 'Repeat: ', 'Next Day: ']



mbead = 85.0e-15 # convert picograms to kg
rhobead = 1550.0 # kg/m^3

rbead = ( (mbead / rhobead) / ((4.0/3.0)*np.pi) )**(1.0/3.0)
Ibead = 0.4 * mbead * rbead**2

print rbead
print Ibead

def sqrt(x, A, x0, b):
    return A * np.sqrt(x-x0) + b

for fileind, file in enumerate(files):
    field_strength, field_err, wobble_freq, wobble_err = np.load(file)

    wobble_freq *= (2 * np.pi)

    field_strength = 100.0 * field_strength * 2.0 

    popt, pcov = opti.curve_fit(sqrt, field_strength, wobble_freq, \
                                p0=[10,0,0], sigma=wobble_err)
    print
    print popt
    print

    plot_x = np.linspace(0, np.max(field_strength), 100)
    plot_x[0] = 1.0e-9 * plot_x[1]
    plot_y = sqrt(plot_x, *popt)

    # 1e-3 to account for 
    d = (popt[0])**2 * Ibead
    d_scaled = d * (1.0 / 1.602e-19) * 1e6

    label = labels[fileind] + ('%0.1f' % d_scaled) + ' $e \cdot \mu \mathrm{m}$'

    plt.plot(plot_x*1e-3, plot_y, '--', lw=2, color='r')
    plt.errorbar(field_strength*1e-3, wobble_freq, \
                 yerr=wobble_err, label=label)
    plt.xlabel('Field [kV/m]')
    plt.ylabel('$\omega_{\phi}$ [rad/s]')

plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

