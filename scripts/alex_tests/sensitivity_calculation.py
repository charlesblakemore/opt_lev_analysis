import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import os
import glob
import matplotlib.mlab as ml
import sys
sys.path.append("../microgravity")
from scipy.optimize import minimize_scalar as ms
import alex_limit_v2 as al2
from scipy.optimize import curve_fit
import matplotlib
from scipy.stats import sem
import build_yukfuncs as yf
import image_util as iu
import scipy.signal as ss
reload(al2)

decca_path = "/home/arider/limit_data/just_decca.csv"
pre_decca_path = "/home/arider/limit_data/pre_decca.csv"
s = 20. #separation in um
fn = 5.e-17 #force noise in N/rt(Hz)

ypts = np.arange(-40, 40, 0.1)
pos = np.array([ypts, np.ones_like(ypts)*s, \
        np.zeros_like(ypts)])

def make_template(pos_data, yukfuncs, cf = 1.E6):
    plt.plot(pos_data[0, :])
    plt.plot(pos_data[1, :])
    plt.plot(pos_data[2, :])
    plt.show()
    template_pts = np.copy(pos_data)
    template_pts/= cf
    template_pts = np.transpose(template_pts)
    fs = np.array([yukfuncs[0](template_pts), yukfuncs[1](template_pts), \
                   yukfuncs[2](template_pts)])
    fs = map(matplotlib.mlab.detrend_linear, fs)
    return fs



pre_decca = np.loadtxt(pre_decca_path, delimiter = ',', skiprows = 1)
decca = np.loadtxt(decca_path, delimiter = ',', skiprows = 1)
plt.loglog(pre_decca[:, 0], pre_decca[:, 1], label = "pre decca")
plt.loglog(decca[:, 0], decca[:, 1], label = "decca")
#plt.loglog(yf.lambdas, , label = "sensitivity")
plt.legend()
plt.grid()
plt.xlabel("$\lambda$ [m]")
plt.ylabel("|$\\alpha$|")
plt.show()

