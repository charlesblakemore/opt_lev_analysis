import glob, re
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


tot_thick_list = np.logspace(-7, np.log10(10e-6), 80)
L_list = [0.2e-6, 0.5e-6, 1e-6, 2e-6]
G = 6.67e-11

out_mat = np.load("cas_mat.npy")

out_mat = np.array(out_mat)

col_list = ['b','g','c','r']

fig = plt.figure()
for i in [0,3]:
    cdat = out_mat[:,i]
    gpts = cdat > 0

    if(i == 1):
        fit_pts = np.logical_and(tot_thick_list > 1e-6, tot_thick_list < 6e-6)
        p = np.polyfit( np.log10(tot_thick_list[fit_pts]), np.log10(cdat[fit_pts]), 1)
        cdat[tot_thick_list > 3.7e-6] = 10**np.polyval(p, np.log10(tot_thick_list[tot_thick_list > 3.7e-6]))
    if(i == 2):
        fit_pts = np.logical_and(tot_thick_list > 3e-6, tot_thick_list < 7e-6)
        p = np.polyfit( np.log10(tot_thick_list[fit_pts]), np.log10(cdat[fit_pts]), 1)
        cdat[tot_thick_list > 5e-6] = 10**np.polyval(p, np.log10(tot_thick_list[tot_thick_list > 5e-6]))

    plt.loglog(tot_thick_list[gpts]*1e6, cdat[gpts], linewidth=1.5, label="$s = %.1f\ \mu\mathrm{m}$, Au/Si"%(L_list[i]*1e6), color=col_list[i])


out_mat = np.load("cas_mat_aucu.npy")
tot_thick_list = np.logspace(-7, np.log10(10e-6), 40)

out_mat = np.array(out_mat)

#fig = plt.figure()
for i in [0,3]:
    cdat = out_mat[:,i]
    gpts = cdat > 0

    ll = plt.loglog(tot_thick_list[gpts]*1e6, cdat[gpts], '--', linewidth=1.5, color=col_list[i], label="$s = %.1f\ \mu\mathrm{m}$, Au/Cu"%(L_list[i]*1e6))
    seq = [8, 2]
    ll[0].set_dashes(seq)


xx = plt.xlim()
plt.plot(xx, [5e-20, 5e-20],'k:', linewidth=1.5)
plt.plot(xx, [1.4e-23, 1.4e-23],'k--', linewidth=1.5)

## overplot 1/r^2
m1 = 4./3*np.pi*(2.5e-6)**3*2e3
m2 = (10e-6)**3 * 19.3e3
r = tot_thick_list + (2.5e-6+5e-6)  ##center of the cube
F = G*m1*m2/r**2
plt.plot(tot_thick_list*1e6, F, 'k', linewidth=1.5)

plt.ylim([1e-25, 1e-12])

plt.xlabel("Total separation from attractor, $s+t$ [$\mu$m]")
plt.ylabel("Differential Casimir force [N]")
plt.legend(prop={"size": 13})

fig.set_size_inches(5, 3.75)
plt.subplots_adjust(bottom=0.13, top=0.95, left=0.15, right=0.97)
plt.savefig("diff_casimir_comb.pdf")

plt.show()
