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
s = 4.0 #separation in um
fn = 1.e-19 #force noise in N/rt(Hz)
tint = 1e5


ypts = np.arange(-40, 40, 0.1)
pos_data = np.array([np.ones_like(ypts)*s, ypts, \
        np.zeros_like(ypts)])

def make_template(pos_data, yukfuncs, cf = 1.E6, plot_pos = False):
    plt.show()
    template_pts = np.copy(pos_data)
    template_pts/= cf
    template_pts = np.transpose(template_pts)
    if plot_pos:
        plt.plot(template_pts[:, 0], label = "x")
        plt.plot(template_pts[:, 1], label = "y")
        plt.plot(template_pts[:, 2], label = "z")
        plt.legend()
        plt.show()
    fs = np.array([yukfuncs[0](template_pts), yukfuncs[1](template_pts), \
                   yukfuncs[2](template_pts)])
    fs = map(matplotlib.mlab.detrend_linear, fs)
    return fs

def sensitivity_at_lambdai(pos_data, yukfuncsi, tint, noise, \
                            signif = 2., plt_fmag = False):
    fs = make_template(pos_data, yukfuncsi)
    fmag = np.sqrt(fs[0]**2 + fs[1]**2 + fs[2]**2)
    amp = (np.max(fmag) - np.min(fmag))/2.
    if plt_fmag:
        plt.plot(pos_data[1], fmag, label = "force magnitude")
        plt.plot(pos_data[1], amp*np.ones_like(pos_data[1]), label = "amp")
        plt.legend()
        plt.show()
    sigma = noise/np.sqrt(tint)
    return signif*sigma/amp

sensitivityer1e3 = lambda i: sensitivity_at_lambdai(pos_data, yf.yukfuncs[:, i], 1e3, fn)
sensitivityer1e4 = lambda i: sensitivity_at_lambdai(pos_data, yf.yukfuncs[:, i], 1e4, fn)
sensitivityer1e5 = lambda i: sensitivity_at_lambdai(pos_data, yf.yukfuncs[:, i], 1e5, fn)
sensitivityer1e6 = lambda i: sensitivity_at_lambdai(pos_data, yf.yukfuncs[:, i], 1e6, fn)

alphase3 = map(sensitivityer1e3, range(len(yf.lambdas)))
alphase4 = map(sensitivityer1e4, range(len(yf.lambdas)))
alphase5 = map(sensitivityer1e5, range(len(yf.lambdas)))
alphase6 = map(sensitivityer1e6, range(len(yf.lambdas)))

pre_decca = np.loadtxt(pre_decca_path, delimiter = ',', skiprows = 1)
decca = np.loadtxt(decca_path, delimiter = ',', skiprows = 1)
plt.loglog(pre_decca[:, 0], pre_decca[:, 1], label = "pre decca")
plt.loglog(decca[:, 0], decca[:, 1], label = "decca")
plt.loglog(yf.lambdas, alphase3, label = "sensitivity 1E3 s")
plt.loglog(yf.lambdas, alphase4, label = "sensitivity 1E4 s")
plt.loglog(yf.lambdas, alphase5, label = "sensitivity 1E5 s")
plt.loglog(yf.lambdas, alphase6, label = "sensitivity 1e6 s")




plt.legend()
plt.grid()
plt.xlabel("$\lambda$ [m]")
plt.ylabel("|$\\alpha$|")
plt.show()

