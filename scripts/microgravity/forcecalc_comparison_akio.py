import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import grav_util as gu
from matplotlib import cm
from matplotlib.ticker import LinearLocator

gpath = '/home/charles/opt_lev_analysis/scripts/gravity_sim/'
filname = 'data/attractorv2_1um_manysep_fullthrow_force_curves.p'

fcurve_obj = gu.Grav_force_curve(gpath+filname)

fcurve_obj.make_splines()

# Akio computed force for 10um sep between bead center and end of gold finger.
# With a 4um protection layer of Si and a 2.37um bead, this corresponds
# to a face to face separation of 3.63um
face_to_face = 3.63e-6  

akiopath1 = gpath + 'akio_calc/X10umAlpha10^10Lambda1um.txt'
akiopath2 = gpath + 'akio_calc/X10umAlpha10^10Lambda5um.txt'
akiopath3 = gpath + 'akio_calc/X10umAlpha10^10Lambda10um.txt'

akiodat1 = np.loadtxt(akiopath1)
akiodat2 = np.loadtxt(akiopath2)
akiodat3 = np.loadtxt(akiopath3)

akiopos = akiodat1[:,1] - 212.5e-6
akioforce1 = akiodat1[:,3]
akioforce2 = akiodat2[:,3]
akioforce3 = akiodat3[:,3]


xarr = np.linspace(-250e-6, 250e-6, 501)
separr = np.linspace(3.0e-6,15.0e-6, 101)

chasforce1 = fcurve_obj.mod_grav_force(xarr, sep=face_to_face, alpha=1e10, yuklambda=1.0e-6, verbose=True)
chasforce2 = fcurve_obj.mod_grav_force(xarr, sep=face_to_face, alpha=1e10, yuklambda=5.0e-6, verbose=True)
chasforce3 = fcurve_obj.mod_grav_force(xarr, sep=face_to_face, alpha=1e10, yuklambda=10.0e-6, verbose=True)

#chasforce1 = chasforce1 + (np.mean(akioforce1) - np.mean(chasforce1[50:-50]))
#chasforce2 = chasforce2 + (np.mean(akioforce2) - np.mean(chasforce2[50:-50]))
#chasforce3 = chasforce3 + (np.mean(akioforce3) - np.mean(chasforce3[50:-50]))


plt.semilogy(akiopos*1e6, akioforce1, color='b', linewidth=1.5, label='$\lambda = 1 \mu m$')
plt.semilogy(xarr*1e6, chasforce1, '--', color='b', linewidth=2)

plt.semilogy(akiopos*1e6, akioforce2, color='g', linewidth=1.5, label='$\lambda = 5 \mu m$')
plt.semilogy(xarr*1e6, chasforce2, '--', color='g', linewidth=2)

plt.semilogy(akiopos*1e6, akioforce3, color='r', linewidth=1.5, label='$\lambda = 10 \mu m$')
plt.semilogy(xarr*1e6, chasforce3, '--', color='r', linewidth=2)

plt.xlabel('Distance Along Cantilever [um]')
plt.ylabel('Modified Gravitational Force [N]')
plt.grid()
plt.legend(numpoints=1, fontsize=10)
plt.title(r'Modified Gravity Force Curves for  $\alpha = 10^{10}$')

plt.show()
