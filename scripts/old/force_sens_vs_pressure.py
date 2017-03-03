## plot the theoretical force sensitivity vs pressure, 
## see Li thesis, eqn 6.14

import numpy as np
import matplotlib.pyplot as plt

p = np.logspace(-10, 3, 1e3) ## in mbar
bead_rad = 2.3e-6 ## m
bead_dens = 2e3 ## kg/m^3
air_viscosity = 18.54e-6 ## Pa s
kb = 1.38e-23
T = 300.
gas_diam = 1.42e-10 ## m

bead_mass = 4./3*np.pi*bead_rad**3 * bead_dens

p_pa = p*100. ##pascal

l = 101325/p_pa * 68e-9 ## scaling mfp to 68 nm at 1 atm, from wikipedia
kn = l/bead_rad
ck = 0.31*kn/(0.785 + 1.152*kn + kn**2)

## li thesis expression
gamma0 = 6*np.pi*air_viscosity*bead_rad/bead_mass * 0.619/(0.619 + kn) * (1+ck)

## libbrect expression
gamma1 = 1./(7. * bead_rad/1e-6 * 10**-10/p * (365.24*24*3600.))

## force sensitivity/sqrt(Hz) from sqrt( 4*m*Gamma*kT)

sig_f1 = np.sqrt(4*bead_mass*gamma0*kb*T)
sig_f2 = np.sqrt(4*bead_mass*gamma1*kb*T)

fmeas = [1e-5, 4.7e-17]

gpts_hi = l < bead_rad
gpts_lo = l > bead_rad

p_bead_rad = p[np.argmin( np.abs( l - bead_rad) )]

fig = plt.figure()
plt.loglog( p[gpts_hi], sig_f1[gpts_hi], 'k', linewidth=1.5, label="Predicted")
plt.loglog( p[gpts_lo], sig_f2[gpts_lo], 'k', linewidth=1.5)
plt.loglog( fmeas[0], fmeas[1], 'ro', markeredgewidth=2, markersize=5, markerfacecolor='w', markeredgecolor='r', label="Measured")
yy=plt.ylim()
plt.plot([p_bead_rad, p_bead_rad], yy, 'k--', linewidth=1.5)
plt.xlabel("Pressure [mbar]")
plt.ylabel(r"Force sensitivity, $\sigma_F$ [N Hz$^{-1/2}$]")
plt.grid("on")
plt.legend(loc="upper left", numpoints=1)
fig.set_size_inches(8,6)
plt.savefig("f_v_p.pdf")
plt.show()
