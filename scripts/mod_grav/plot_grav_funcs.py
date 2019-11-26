import numpy as np
import bead_util as bu
import grav_util_2 as gu2
import matplotlib.pyplot as plt

theory_data_dir = "/data/grav_sim_data/2um_spacing_data/"

outdic = gu2.build_mod_grav_funcs(theory_data_dir)

xdc = 20E-6 #x dc position in m
zdc = -10E-6 #z dc position in m
ydc = 0.

lam = 25E-6
lam_ind = np.argmin((np.array(outdic['lambdas']) - lam)**2)

def make_pts(xdc, ydc, zdc, outdic, N = 1000):
    yliml = outdic['lims'][1][0]
    ylimu = outdic['lims'][1][1]
    xs = np.ones(N)*xdc
    ys = np.linspace(yliml, ylimu, N)
    zs  = np.ones(N)*zdc
    return np.transpose(np.vstack([xs, ys, zs]))


pts = make_pts(xdc, ydc, zdc, outdic)
yfs = outdic['yukfuncs']
fx = yfs[0][lam_ind](pts)
fy = yfs[1][lam_ind](pts)
fz = yfs[2][lam_ind](pts)

plt.plot(pts[:, 1], fx, label = 'fx')
plt.plot(pts[:, 1], fy, label = 'fy')
plt.plot(pts[:, 1], fz, label = 'fz')
plt.legend()
plt.xlabel('attractor displacement [$\mu m$]')
plt.ylabel('force [N]')
plt.show()
