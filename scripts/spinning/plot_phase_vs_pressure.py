import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from scipy.optimize import curve_fit
import matplotlib

base_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/pramp_data/" 
in_fs = ["50k_1vpp", "50k_2vpp", "50k_3vpp", "50k_4vpp", "50k_5vpp", "50k_6vpp", "50k_7vpp", "50k_8vpp"]

def get_phi(fname):
    phi = np.load(base_path + fname + "phi.npy")
    return phi/2

def get_pressure(fname):
    pressures = np.load(base_path + fname + "pressures.npy")
    return pressures

def pressure_model(pressures, break_ind = 0, p_ind = 0, plt_press = False):
    ffun = lambda x, y0, m0, m1: \
            pw_line(x, break_ind, 1e7, 1.1e7, y0, m0, m1, 0., 0.)

    n = np.shape(pressures)[0]
    inds = np.arange(n)
    popt, pcov = curve_fit(ffun, inds, pressures[:, p_ind])
    pfit = ffun(inds, *popt)

    if plt_press:
        p_dict = {0:"Pirani", -1:"Baratron"}
        plt.plot(pressures[:, p_ind], 'o', label = p_dict[p_ind])
        plt.plot(pfit, label = "piecewise linear fit")
        plt.xlabel("File [#]")
        plt.ylabel("Pressure [mbar]")
        plt.legend()
        plt.show()

    return pfit


def phi_ffun(p, k, phinot):
    return -1.*np.arcsin(np.clip(p/k, 0., 1.)) + phinot

phases = np.array(list(map(get_phi, in_fs)))
pressures = np.array(list(map(get_pressure, in_fs)))
p_fits = np.array(list(map(pressure_model, pressures)))

p_maxs = [0.011, 0.024, 0.036, 0.048, 0.062, 0.074, 0.085, 0.099]
popts = []
pcovs = []

#p0 = [0.02, 0.75]

for i, p in enumerate(p_fits):
    bfit = p<p_maxs[i]
    p0 = [p_maxs[i], 0.75]
    pphi, covphi = curve_fit(phi_ffun, p_fits[i][bfit], phases[i][bfit], p0 = p0)
    popts.append(pphi)
    pcovs.append(covphi)

colors = ["b", "g", "c", "m", "y", "k"]
linestyles = [":", "-.", "--", "-"]
plt_inds = [0, 2, 4]
labels = ["12.5kV/m", "37.5kV/m", "62.5kV/m"]
axi = 1
matplotlib.rcParams.update({'font.size':12})
f, axarr = plt.subplots(len(plt_inds), 1, dpi = 200, sharex = True)
for i, ax in enumerate(axarr):
    ind = plt_inds[i]
    bi = p_fits[ind]<0.065
    p_plot = np.linspace(0, popts[ind][0], 1000)
    ax.plot(p_fits[ind][bi], (phases[ind][bi]-popts[ind][-1])/np.pi, 'o')
    ax.text(popts[ind][0]-0.009, -0.05, labels[i])
    ax.axhline(y = -0.5, linestyle = '--', color = 'k', alpha = 0.5)
    ax.plot(p_plot, (phi_ffun(p_plot, *popts[ind])-popts[ind][-1])/np.pi, 'r')
    ax.set_ylim([-0.6, 0.6])
    if i==axi:
        ax.set_ylabel(r"$\phi_{eq}[\pi]$")

plt.xlabel("Pressure [mbar]")
#plt.ylabel(r"$\phi_{eq}$")
plt.legend()
plt.show()

def line(x, m, b):
    return m*x + b

Es = np.array([1, 2, 3, 4, 5, 6, 7, 8])*50./0.004
E_plot = np.linspace(0, 10*50/0.004, 1000)
popts = np.array(popts)
pcovs = np.array(pcovs)
scale = 1000
popt, pcov = curve_fit(line, Es, popts[:, 0])
f, ax = plt.subplots(dpi = 200)
ax.errorbar(Es/scale, popts[:, 0], np.sqrt(pcovs[:, 0, 0]), fmt = 'o')
ax.plot(E_plot/scale, line(E_plot, *popt), 'r')
ax.set_xlabel("E[kV/m]")
ax.set_ylabel(r"$P_{max}$[mbar]")
plt.show()




