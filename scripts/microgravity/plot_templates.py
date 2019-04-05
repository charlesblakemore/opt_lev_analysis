import numpy as np
import matplotlib.pyplot as plt
from build_yukfuncs import *
import matplotlib

lam25umind = np.argmin((lambdas-2.5e-5)**2)

xpltarr = np.arange(lims[0][0], lims[0][1], 1e-7)
ypltarr = np.arange(-2.5E-4, 2.5E-4, 1e-7)
zpltarr = np.arange(lims[2][0], lims[2][1], 1e-7)

ones = np.ones_like(ypltarr)
pts = ptarr(1.5e-5*ones, ypltarr, 0.*ones)

matplotlib.rcParams.update({'font.size':14, 'font.weight':'bold'})
lw = 4

plt.plot(ypltarr*10**6, yukfuncs[2][lam25umind](pts)*1e24, color = 'C2', label = r'$F_z$', linewidth = lw)
plt.plot(ypltarr*10**6, yukfuncs[1][lam25umind](pts)*1e24, color = 'C1', label = r'$F_y$', linewidth = lw)
plt.plot(ypltarr*10**6, yukfuncs[0][lam25umind](pts)*1e24, color = 'C0', label = r'$F_x$', linewidth = lw)
plt.xlabel(r"displacement [$\mu$m]", fontweight = 'bold')
plt.ylabel(r"F[yN/$\alpha$]", fontweight = 'bold')
plt.legend()
plt.tight_layout()
plt.savefig("force_templates.png", dpi = 200)
plt.show()
