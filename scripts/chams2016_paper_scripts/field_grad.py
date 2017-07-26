import numpy as np
import matplotlib.pyplot as plt

L = 0.5 ## mm
z = 0.1 ## mm
rho = 0.1 ## mm
a = 4. ## mm
V = 1.

n = np.arange(1000)

def calc_phi( z, rho ):

    bnp = 2*n*L + np.abs(z + L )
    bnm = 2*n*L - np.abs(z - L )

    xnp = 0.5*( np.sqrt( bnp**2 + (rho+a)**2) + np.sqrt( bnp**2 + (rho-a)**2 ) )
    xnm = 0.5*( np.sqrt( bnm**2 + (rho+a)**2) + np.sqrt( bnm**2 + (rho-a)**2 ) )

    phi = 2*V/np.pi * np.sum( np.arcsin( a/xnm ) - np.arcsin( a/xnp ) )

    return phi

yy = np.linspace(0, 4., 1e3)
phi_vec = np.zeros_like(yy)
for i,y in enumerate(yy):
    phi_vec[i] = calc_phi( y, rho )
    
dy = np.median(np.diff(yy))

evec = (np.diff(phi_vec)/dy) / (V/(2*L))


plt.figure()
plt.plot( yy, phi_vec )

plt.figure()
plt.plot( yy[:-1], evec )

plt.show()
