import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import grav_util as gu
from matplotlib import cm
from matplotlib.ticker import LinearLocator


gpath = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/'
filname = 'attractorv2_1um_manysep_fullthrow_force_curves.p'

fcurve_obj = gu.Grav_force_curve(gpath+filname)

fcurve_obj.make_splines()

xarr = np.linspace(-250e-6, 250e-6, 501)
separr = np.linspace(3.0e-6,15.0e-6, 101)


for x in xarr:
    try:
        sepgrid = np.vstack( (sepgrid, separr) )
    except:
        sepgrid = separr
sepgrid = np.transpose(sepgrid)


for sep in separr:
    fcurve = fcurve_obj.mod_grav_force(xarr, sep=sep, alpha=1e5, yuklambda=10.0e-6, verbose=False)
    try:
        xgrid = np.vstack( (xgrid, xarr) )
        out = np.vstack( (out, fcurve) )
    except:
        xgrid = xarr
        out = fcurve
#xgrid = np.transpose(xgrid)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.plot_wireframe(sepgrid, xgrid, out)
surf = ax.plot_surface(sepgrid*1e6, xgrid*1e6, out, rstride=2, cstride=2,\
                       cmap=cm.coolwarm, linewidth=0, antialiased=False)

fig.colorbar(surf, aspect=20)

plt.show()
