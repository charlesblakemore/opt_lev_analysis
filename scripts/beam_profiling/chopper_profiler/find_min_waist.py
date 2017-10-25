import os, fnmatch

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import scipy.optimize as opti
import scipy.special as special
import scipy.interpolate as interp

import peakdetect as pdet

import bead_util as bu
import configuration as config

fontsize=16
plot_title = ''

lamb = 1.064e-6

def w(z, z0, w0, C):
    zR = np.pi * w0**2 / lamb
    return w0 * np.sqrt( 1 + ((z-75e-2)/zR)**2 )


x = np.array([35.5, 48.2, 61.1, 84.2, 104.1, 124.8, 144.4, 165.8, 183.1, 203.9])

wx = np.array([0.838, 0.818, 0.817, 0.814, 0.832, 0.836, 0.852, 0.869, 0.889, 0.905])
wy = np.array([0.818, 0.814, 0.812, 0.815, 0.817, 0.830, 0.840, 0.859, 0.864, 0.888])

#wx = np.array([0.61, 0.52, 0.47, 0.37, 0.39, 0.53, 0.71, 0.914, 1.075, 1.27])
#wy = np.array([0.60, 0.53, 0.46, 0.36, 0.385, 0.532, 0.71, 0.93, 1.085, 1.30])

x *= 1.0e-2
wx *= 1.0e-3
wy *= 1.0e-3

#plt.plot(x, wx)
#plt.plot(x, w(x, 84, 0.4))

x_popt, x_pcov = opti.curve_fit(w, x, wx, p0=[50e-2, 0.4e-3, 0], maxfev=10000)
y_popt, y_pcov = opti.curve_fit(w, x, wy, p0=[50e-2, 0.4e-3, 0], maxfev=10000)


xinterp = interp.interp1d(x, wx, kind="cubic")


print "w_0x [mm] and position [cm]: ", x_popt[1]*1e3, x_popt[0]*1e2
print "w_0y [mm] and position [cm]: ", y_popt[1]*1e3, y_popt[0]*1e2


plt_pts = np.linspace(np.min(x), np.max(x), 200)

plt.plot(x, wx, 's')
plt.plot(plt_pts, xinterp(plt_pts))

min_result = opti.minimize(xinterp, 80e-2)
print min_result.x

waist_pos = 25e-2
plt_pts_2 = plt_pts - min_result.x + waist_pos
new_xinterp = interp.interp1d(plt_pts_2, xinterp(plt_pts))

plt.plot(plt_pts_2, new_xinterp(plt_pts_2))

ind = 0
for pos in x:
    print "pos %i: " % ind, 
    ind += 1
    try:
        print new_xinterp(pos) * 1e3
    except:
        print "Too far!"
        break

fig1, axarr1 = plt.subplots(2,1,sharex=True,sharey=True)
axarr1[0].plot(x * 1e2, wx * 1e3, label="X Waists")
axarr1[0].plot(plt_pts * 1e2, w(plt_pts, *x_popt) * 1e3, '--', color = 'r', \
               linewidth=1.5, label="Gaussian Diffraction")
axarr1[0].set_xlabel("Displacement [cm]", fontsize=fontsize)
axarr1[0].set_ylabel("Waist [mm]", fontsize=fontsize)
axarr1[0].legend(fontsize=fontsize-4, loc=0)
plt.setp(axarr1[0].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr1[0].get_yticklabels(), fontsize=fontsize, visible=True)


axarr1[1].plot(x * 1e2, wy * 1e3, label="Y Waists")
axarr1[1].plot(plt_pts * 1e2, w(plt_pts, *y_popt) * 1e3, '--', color = 'r', \
               linewidth=1.5, label="Gaussian Diffraction")
axarr1[1].set_xlabel("Displacement [cm]", fontsize=fontsize)
axarr1[1].set_ylabel("Waist [mm]", fontsize=fontsize)
axarr1[1].legend(fontsize=fontsize-4, loc=0)
plt.setp(axarr1[1].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr1[1].get_yticklabels(), fontsize=fontsize, visible=True)

plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

if plot_title:
    plt.suptitle(plot_title, fontsize=20)
    plt.subplots_adjust(top=0.9)


plt.show()
