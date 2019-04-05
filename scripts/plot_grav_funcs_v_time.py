import numpy as np
import bead_util as bu
import grav_util_2 as gu2
import matplotlib.pyplot as plt
import dill as pickle
import scipy.signal as ss

theory_data_dir = "/data/grav_sim_data/2um_spacing_data/"

outdic = gu2.build_mod_grav_funcs(theory_data_dir)

xdc = 20E-6 #x dc position in m
zdc = -10E-6 #z dc position in m
ydc = 0.


lam = 25E-6
lam_ind = np.argmin((np.array(outdic['lambdas']) - lam)**2)

def make_pts(xdc, ydc, zdc, outdic, te = 10., fdrive= 17., \
        Fs = 5000, drive = 40E-6):
    t = np.arange(0, te, 1./Fs)
    yliml = outdic['lims'][1][0]
    ylimu = outdic['lims'][1][1]
    xs = np.ones_like(t)*xdc
    ys = drive*np.sin(2.*np.pi*fdrive*t) + ydc
    zs  = np.ones_like(t)*zdc
    return np.transpose(np.vstack([xs, ys, zs])), t


pts, t = make_pts(xdc, ydc, zdc, outdic)
yfs = outdic['yukfuncs']
fx = yfs[0][lam_ind](pts)
fy = yfs[1][lam_ind](pts)
fz = yfs[2][lam_ind](pts)
f = np.transpose(np.array([fx, fy, fz]))
savedic = {"pts": pts, "f": f}
pickle.dump(savedic, open("attractor_forces_t.p", "wb"))

pts*=1E6

plt.plot(t, ss.detrend(f[:, 0]),'b', label = '$f_{x}$')
plt.plot(t, ss.detrend(f[:, 1]),'g', label = '$f_{y}$')
plt.plot(t, ss.detrend(f[:, 2]),'r', label = '$f_{z}$')


plt.xlim(0, 2./17.)

#plt.ticklabel_format(style = 'sci', axis= 'x', scilimits=(0, 0))
plt.legend()
plt.xlabel('time[s]')
plt.ylabel('force [N]')
plt.show()
