import numpy as np
import bead_util as bu
import grav_util_2 as gu2
import matplotlib.pyplot as plt
import dill as pickle


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
f = np.transpose(np.array([fx, fy, fz]))
savedic = {"pts": pts, "f": f}
pickle.dump(savedic, open("attractor_forces.p", "wb"))

pts*=1E6

plt.plot(pts[:, 1], f[:, 0],'b', label = '$f_{x}$')
plt.plot(pts[:, 1], f[:, 1],'g', label = '$f_{y}$')
plt.plot(pts[:, 1], f[:, 2],'r', label = '$f_{z}$')

li = np.argmin((pts[:, 1] + 40)**2)
ri = np.argmin((pts[:, 1] - 40)**2)


plt.plot(pts[li:ri, 1], f[li:ri, 0],'b', linewidth = 7)
plt.plot(pts[li:ri, 1], f[li:ri, 1],'g',linewidth = 7)
plt.plot(pts[li:ri, 1], f[li:ri, 2],'r', linewidth = 7)

plt.xlim(-225, 225)
#plt.ticklabel_format(style = 'sci', axis= 'x', scilimits=(0, 0))
plt.legend()
plt.xlabel('attractor displacement [$\mu m$]')
plt.ylabel('force [N]')
plt.show()
