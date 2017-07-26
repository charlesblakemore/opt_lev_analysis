import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu


#def get_chameleon_force( xpoints_in ):
    

#def get_chameleon_force_chas(xpoints_in, y=0, yforce=False):

xpts = np.arange(0, 0.0002, 1e-6) #Distance between the cantilever and microsphere in meters.

#fpts1 = bu.get_chameleon_force(xpts)
fpts2 = bu.get_chameleon_force_chas(xpts)

#plt.semilogy(xpts, fpts1, 'o', label = 'bu.get_chameleon_force') 
plt.semilogy(xpts*1e6, fpts2, 'o', label = 'bu.get_chameleon_force_chas') 
plt.xlabel("Distance from bead [$\mu m$]")
plt.ylabel("Chameleon Force [$N$]")
plt.legend()
plt.show()
