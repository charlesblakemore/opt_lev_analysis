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
base_path = '/processed_data/spinning/pramp_data/20190626'
base_plot_path = '/home/charles/plots/20190626/pramp/example_pramp'

def phi_ffun(p, k, phi0):
    # negative arcsin function for phase lag fit. It's parameterized
    # such that param[0] (popt[0] from curve_fit) is basically pmax
    return -1.*np.arcsin(p/k) + phi0

alldat = pickle.load(open(base_path + '/all_data.p', 'rb'))

He_pmax = alldat['He']['pmax'][0]
Ar_pmax = alldat['Ar']['pmax'][0]
SF6_pmax = alldat['SF6']['pmax'][0]

He_dat = alldat['He']['data'][0]
Ar_dat = alldat['Ar']['data'][0]
SF6_dat = alldat['SF6']['data'][0]

fig = plt.figure(figsize=(6,4),dpi=200)
ax = fig.add_subplot(111)
line_p = np.linspace(-0.5 * He_pmax, 1.5 * He_pmax, 100) 
min_line = np.ones_like(line_p) * (-0.5)
ax.plot(line_p, min_line, '--', color='k', lw=2, alpha=0.6)

He_sv = bu.get_scivals(He_pmax)
Ar_sv = bu.get_scivals(Ar_pmax)
SF6_sv = bu.get_scivals(SF6_pmax)

He_lab_str = 'He: $p_{\\mathrm{max}}$' \
                + '$ = {0} \\times 10^{{{1}}}~$ mbar'.format('{:0.2f}'.format(He_sv[0]), \
                                                            '{:d}'.format(He_sv[1]))
He_lab_str = 'He: $p_{\\mathrm{max}}$' + '$={:0.2g}$ mbar'.format(He_pmax)

Ar_lab_str = 'Ar: $p_{\\mathrm{max}}$' \
                + '$ = {0} \\times 10^{{{1}}}~$ mbar'.format('{:0.2f}'.format(Ar_sv[0]), \
                                                            '{:d}'.format(Ar_sv[1]))
Ar_lab_str = 'Ar: $p_{\\mathrm{max}}$' + '$={:0.3f}$ mbar'.format(Ar_pmax)

SF6_lab_str = 'SF$_6$: $p_{\\mathrm{max}}$' \
                + '$ = {0} \\times 10^{{{1}}}~$ mbar'.format('{:0.2f}'.format(SF6_sv[0]), \
                                                            '{:d}'.format(SF6_sv[1]))
SF6_lab_str = 'SF$_6$: $p_{\\mathrm{max}}$' + '$={:0.3f}$ mbar'.format(SF6_pmax)

stuff = [[He_pmax, He_dat, He_lab_str, 'C0'],\
         [Ar_pmax, Ar_dat, Ar_lab_str, 'C1'],\
         [SF6_pmax, SF6_dat, SF6_lab_str, 'C2'] ]

maxp = 0
colors = bu.get_color_map(7, cmap='inferno')[::-1]
for i in [0,1,2]:
    pmax = stuff[i][0]
    dat = stuff[i][1]
    lab_str = stuff[i][2]
    #color = stuff[i][3]
    color = colors[2*i+1]

    if np.max(dat[0]) > maxp:
        maxp = np.max(dat[0])

    fitp = np.linspace(0, np.max(dat[0]), 100)
    fit = np.array(phi_ffun(fitp, pmax, 0))
    ax.scatter(dat[0], dat[1] / np.pi, edgecolors=color, facecolors='none', alpha=0.5)
    ax.plot(fitp, fit / np.pi, '-', color=color, lw=3, label=lab_str)

ax.set_xlim(-0.05*maxp, 1.05*maxp)
#ax.set_xlabel('Pressure, $p$ [mbar]')
ax.set_xlabel('$p$ [mbar]')
ax.set_ylabel('$\phi_{\mathrm{eq}}$ [$\pi$ rad]')
ax.legend(fontsize=10)
plt.tight_layout()
# fig.suptitle(title_str, fontsize=16)
# fig.subplots_adjust(top=0.91)

fig_path = base_plot_path + '.png'
fig_path2 = base_plot_path + '.pdf'
fig_path3 = base_plot_path + '.svg'
bu.make_all_pardirs(fig_path)
fig.savefig(fig_path)
fig.savefig(fig_path2)
fig.savefig(fig_path3)
plt.show()

