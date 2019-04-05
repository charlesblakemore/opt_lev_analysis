import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from scipy.optimize import curve_fit
import matplotlib

in_f0 = "50k_pressure_ramp_div_4"
in_f1 = "50k_pressure_ramp_div_40"
in_f2 = "50k_pressure_ramp_5vpp"
in_f3 = "50k_pressure_ramp_2vpp"
in_f4 = "50k_pressure_ramp_1vpp"
base_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/golden/pramp_data/" 


phi0 = np.load(in_f0 + "phi.npy")
phi0[phi0>0.]-=2.*np.pi
pressures0 = np.load(in_f0 + "pressures.npy")

phi1 = np.load(in_f1 + "phi.npy")
pressures1 = np.load(in_f1 + "pressures.npy")


phi2 = np.load(base_path + in_f2 + "phi.npy")
#phi2[phi2>0.]-=2.*np.pi
pressures2 = np.load(base_path + in_f2  + "pressures.npy")


phi3 = np.load(base_path + in_f3 + "phi.npy")
#phi3[phi3>0.]-=2.*np.pi
pressures3 = np.load(base_path + in_f3  + "pressures.npy")


phi4 = np.load(base_path + in_f4 + "phi.npy")
#phi4[phi4>0.]-=2.*np.pi
pressures4 = np.load(base_path + in_f4  + "pressures.npy")

b_ind0 = 107
p_ind0 = 0
p_ind1 = -1

b_ind1 = 180


def pressure_model(pressures, break_ind, p_ind, plt_press = False):
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



pfit0 = pressure_model(pressures0, b_ind0, p_ind0)
pfit1 = pressure_model(pressures1, b_ind1, p_ind1)
pfit2 = pressure_model(pressures2, b_ind1, p_ind0, plt_press = False)
pfit3 = pressure_model(pressures3, b_ind1, p_ind0)
pfit4 = pressure_model(pressures4, b_ind1, p_ind0)



pmax0 = 0.108
pmax1 = 0.0093
pmax2 = 0.109
pmax3 = 0.0434
pmax4 = 0.0206

def phi_ffun(p, k, phinot):
    return -1.*np.arcsin(np.clip(p/k, 0., 1.)) + phinot

bfit0 = pfit0<pmax0
bfit1 = pfit1<pmax1
bfit2 = pfit2<pmax2
bfit3 = pfit3<pmax3
bfit4 = pfit4<pmax4


p0 = [0.1, 0.75]
phi0 /= 2.
phi1 /= 2.
phi2 /= 2.
phi3 /= 2.
phi4 /= 2.

E0 = (10./4.)*100./0.004
E1 = (10./40.)*100./0.004

pphi0, covphi0 = curve_fit(phi_ffun, pfit0[bfit0], phi0[bfit0], p0 = p0)
pphi1, covphi1 = curve_fit(phi_ffun, pfit1[bfit1], phi1[bfit1], p0 = p0)
pphi2, covphi2 = curve_fit(phi_ffun, pfit2[bfit2], phi2[bfit2], p0 = p0)
pphi3, covphi3 = curve_fit(phi_ffun, pfit3[bfit3], phi3[bfit3], p0 = p0)
pphi4, covphi4 = curve_fit(phi_ffun, pfit4[bfit4], phi4[bfit4], p0 = p0)

'''phi0-=pphi0[-1]
phi1-=pphi1[-1]
phi2-=pphi2[-1]
phi3-=pphi3[-1]
phi4-=pphi4[-1]'''


colors = ["b", "g", "c", "m", "y", "k"]
linestyles = [":", "-.", "--", "-"]

matplotlib.rcParams.update({'font.size':10})
f, ax = plt.subplots(dpi = 200)


p_plot = np.linspace(np.min(pfit3), pphi3[0], 1000)


ax.plot(pfit3[bfit3], (phi3[bfit3]-pphi3[-1])/np.pi, 'o', color = 'C0', label = "31.3kV/m drive")
ax.plot(pfit3[np.logical_not(bfit3)], (phi3[np.logical_not(bfit3)]-pphi3[-1])/np.pi, 'o', color = 'C0')
ax.plot(p_plot, (phi_ffun(p_plot, *pphi3)-pphi3[-1])/np.pi, color = "red", label = r"$p_{max} = $0.047 mbar")




ax.axhline(y = -0.5,linestyle = '--' , color = 'k', alpha = 0.5)
#ax.set_xscale("log")



a_str = r"$\phi (p) = arcsin(p/p_{max})$"
plt.text(1e-4, -0.2, a_str)
plt.xlabel("Pressure [mbar]")
plt.ylabel(r"Phase Lag [$\pi$rad]")
plt.legend(loc = 2)
plt.show()


def line(x, m, b):
    return m*x + b



Es = np.array([62.5e3, 25e3, 12.5e3])
p_maxs = np.array([pphi2[0], pphi3[0], pphi4[0]])
sig_pm = np.sqrt([covphi2[0][0], covphi3[0][0], covphi4[0][0]])

pline, lcov = curve_fit(line, Es, p_maxs, sigma = sig_pm)
E_plot = np.linspace(0, 1.2*Es[0], 1000)

f, ax = plt.subplots(dpi = 200)
ax.errorbar(Es/1000, p_maxs, sig_pm, fmt = 'o')
ax.plot(E_plot/1000, line(E_plot, *pline), 'r', label = r"1.87$\mu$bar/kV/m at 50kHz")
plt.legend()
plt.xlabel("Drive Field [kV/m]")
plt.ylabel("Maximum Pressure [mbar]")
plt.show()

