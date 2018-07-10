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

calculate_sps = False
recalculate = True
calculate_limit = True

save_name = "binned_force_data.npy"
save_limit_data = "limit_data.npy"
dat_dir = "/data/20180625/bead1/nobead/grav_data/shield/X50-75um_Z15-25um_17Hz_elec-term"
increment = 1
plt_file = 10
plt_increment = 100
ah5 = lambda fname: fname + '.h5'
files = bu.find_all_fnames(dat_dir)
if calculate_sps:
    sps = np.array(map(iu.getNanoStage, map(ah5, files)))
    np.save("sps.npy", sps)
else:
    sps = np.load("sps.npy")
ba0 = sps[:, 0]>74.
ba1 = sps[:, 2]>24.
files = np.array(files)
files = files[ba0*ba1]
#files = files[:100]
lam25umind = np.argmin((yf.lambdas - 25E-6)**2)
n_file = len(files)
#files = map(ah5, files)
p0 = [20., 0., 25.]
force_data = [] #np.zeros((n_file, 3, 2, 100))
if recalculate:
    for i, f in enumerate(files):
        bu.progress_bar(i, n_file)
        df = bu.DataFile()
        df.load(f)
        df.diagonalize()
        #df.inject_fake_signal(yf.yukfuncs[:, lam25umind], p0, fake_alpha = 5E10, make_plot = False)
        df.calibrate_stage_position()
        df.get_force_v_pos()
        df.binned_data = np.array(df.binned_data)
        df.conv_facs = np.array(df.conv_facs)
        df.binned_data[:, 1, :] = \
                np.einsum('ij,i->ij', df.binned_data[:, 1, :], df.conv_facs)
        force_data.append(df.binned_data)
    
    force_data = np.array(force_data)
    np.save(save_name, force_data)

else:
    force_data = np.load(save_name)

def plot_contours(force_data, bin_width = 80./100.):
    s = np.shape(force_data)
    xx, yy = np.meshgrid(np.arange(s[0]), np.arange(s[-1]))
    Z = force_data[:, 0, 1, :]
    plt.contour(X, Y, Z)
    plt.show()


def make_template(mean_data, yukfuncs, p0=p0, cf = 1.E6, stage_travel = [80., 80., 80.]):

    pvec = np.zeros_like(mean_data[:, 0, :])
    pvec[0, :] = np.zeros_like(mean_data[1, 0, :]) + p0[0]
    pvec[1, :] = mean_data[1, 0, :] - stage_travel[1]/2. + p0[1]
    pvec[2, :] = np.zeros_like(mean_data[1, 0, :])
    pvec/= cf
    plt.show()
    pts = np.stack(pvec, axis = -1)
    fs = np.array([yukfuncs[0](pts), yukfuncs[1](pts), \
                   yukfuncs[2](pts)])
    fs = map(matplotlib.mlab.detrend_linear, fs)
    return fs

def fit_alpha(mean_data, sems, yukfuncs, p0 = p0, cf = 1.E6, \
              alpha_scale = 1E9, plt_best_fit = False, signif = 1.92):
    mean_data[0, 1, :] = ss.detrend(mean_data[0, 1, :])
    fs = make_template(mean_data, yukfuncs, p0 = p0, cf = cf)
    n = np.shape(mean_data)[-1]
    def rcs(alpha):
        return np.sum((mean_data[0, 1, :] - alpha*alpha_scale*fs[0])**2\
                /sems[0, 1, :]**2)/(n-1)
    res0 = ms(rcs)
    def delt_chi_sq(b):
        return (rcs(b) - res0.fun - signif)**2
    res1 = ms(delt_chi_sq)
    error = np.abs((res0.x - res1.x))
    


    if plt_best_fit:
        plt.plot(mean_data[1, 0, :], mean_data[0, 1, :], label = "mean data")
        plt.plot(mean_data[1, 0, :], mean_data[0, 1, :]+sems[0, 1, :]\
                , 'k-')
        plt.plot(mean_data[1, 0, :], mean_data[0, 1, :]-sems[0, 1, :]\
                , 'k-')
        plt.plot(mean_data[1, 0, :], res0.x*alpha_scale*fs[0],\
                 label = "best fit, alpha="+str(res0.x)+"Galpha")
        plt.xlabel("cantilever y displacement [m]")
        plt.ylabel("force [N]")
        plt.legend()
        plt.show()
    return res0.x, error 

def fit_alpha_individual_files(force_data, sems, yukfuncs, p0=p0, cf = 1.E6,\
        alpha_scale = 1E9, plt_best_fit = False, signif = 1.0,\
        plot_alphas = True, delta_t = 10.):
    sems_ind = sems*np.sqrt(len(force_data))
    n = len(force_data)
    aes = np.zeros((n, 2))
    for i, data in enumerate(force_data):
        a, e = fit_alpha(data, sems_ind, yukfuncs, p0 = p0, cf = cf,\
                alpha_scale = alpha_scale, plt_best_fit = False, \
                signif = signif)
        aes[i, 0] = a
        aes[i, 1] = e
    if plot_alphas:
        tarr = delta_t*np.arange(n)
        plt.errorbar(tarr, alpha_scale*aes[:, 0], alpha_scale*aes[:, 1],\
                fmt = '.')
        plt.xlabel("time [s]")
        plt.ylabel("|$\\alpha$|")
        plt.show()
    return aes
#force data inds : [file, direction, drive/response, bin #]
#force_data = force_data[ba0*ba1, :, :, :]
mean_data = np.mean(force_data, axis = 0)
sems = sem(force_data, axis = 0)

#yf25um = yf.yukfuncs[0, yf.lam25umind]
n_lambda = len(yf.lambdas)
lim_data = np.zeros((n_lambda, 2))
if calculate_limit:
    for i, lams in enumerate(yf.lambdas):
        amin, e = fit_alpha(mean_data, sems, yf.yukfuncs[:, i])
        lim_data[i, 0] = amin
        lim_data[i, 1] = e
    np.save(save_limit_data, lim_data)

else:
    lim_data = np.load(save_limit_data)

n_cut = -10 #number of lambdas to cut
pre_decca = np.loadtxt(pre_decca_path, delimiter = ',', skiprows = 1)
decca = np.loadtxt(decca_path, delimiter = ',', skiprows = 1)
plt.loglog(pre_decca[:, 0], pre_decca[:, 1], label = "pre decca")
plt.loglog(decca[:, 0], decca[:, 1], label = "decca")
plt.loglog(yf.lambdas[:n_cut], np.abs(lim_data[:n_cut, 0])*1E9, label = "background")
#plt.loglog(yf.lambdas[:n_cut], lim_data[:n_cut, 1]*1E9, label = "background fluctuations")
plt.legend()
plt.grid()
plt.xlabel("$\lambda$ [m]")
plt.ylabel("|$\\alpha$|")
plt.show()
for i in np.arange(0, 4, 1):
    plt.plot(force_data[i, 0, 0, :], ss.detrend(force_data[i, 0, 1, :]), label = "t=" + str(i*10) + 's')


plt.plot(mean_data[0, 0, :], ss.detrend(mean_data[0, 1, :]), label = "mean", linewidth = 5)
        
        
plt.plot(mean_data[0, 0, :], ss.detrend(mean_data[0, 1, :]) + sems[0, 1, :], \
        label = "+sigma")

plt.plot(mean_data[0, 0, :], ss.detrend(mean_data[0, 1, :]) - sems[0, 1, :], \
        label = "-sigma")
plt.xlabel("attractor displacement[um]")
plt.ylabel("Force[N]")
plt.legend()
plt.show()

