import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import Tkinter
import tkFileDialog
import os
from scipy.optimize import curve_fit
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize 

dirs = map(str, np.arange(251, 263, 1))
#dirs = map(str, [247, 248, 249, 250])
cal_dir = str(246)
cal = 5.0e-14

ddict = bu.load_dir_file( "/home/arider/opt_lev/scripts/cant_force/dir_file.txt" )

load_from_file = False
calibrate = False

def proc_dir(d):
    #loads the average force vs position from the files given a key referencing that file.
    dv = ddict[d]
    dir_obj = cu.Data_dir(dv[0], [0, 0, dv[-2]])
    if load_from_file:
        dir_obj.avg_force_v_p()
        dir_obj.save_dir()
    else:
        dir_obj.load_from_file()
    
    return dir_obj


def throw_out_nan(vec):
    #Gets rid of points where either the x, the force, or the error is a nan. Assumes vector is of the form [xs, fs, es] returned by the avg_force_v_p method os a Data_dit object.
    vec = np.row_stack(vec)
    bool = -np.isnan(vec[2, :])
    return vec[:, bool]

def ave_force_offset(v0, v1):
    #Measures the average force offset between two average force vs positions with overlapping x ranges.
    vt0 = throw_out_nan(np.array(v0))
    vt1 = throw_out_nan(np.array(v1))
    x0inx1 = lambda x: x in vt1[0]
    b0 = np.array(map(x0inx1, vt0[0])) #boolian array True where v0[0] overlaps with v1[0].
    if np.sum(b0)<1:
        print "Warning: no overlap"
    x1inx0 = lambda x: x in vt0[0]
    b1 = np.array(map(x1inx0, vt1[0])) #b1 with v0<->v1.
    v0o = vt0[:, b0] 
    v1o = vt1[:, b1]
    sumsq = lambda oset: np.sum((v1o[1] - v0o[1] + oset)**2/(v1o[2]**2 + v0o[2]**2))
    oset0 = [0.]
    res = minimize(sumsq, tol = 1e-10, bounds = (-0.0005, 0.0005))
    return res
    

def cham_force_beta(xarr, beta, f0, units = 1e-6, beta_fac = 1.):
    #multiplies the chameleon force by beta for fitting out beta. Scale beta by fac to avoid issues with numerical precision in fit.
    xarr = np.array([xarr]).flatten()
    return beta*beta_fac*bu.get_chameleon_force(xarr*units).flatten() + f0

def es_force_v(xarr, v, f0, units = 1e-6):
    #multiplies the chameleon force by beta for fitting out beta. Scale beta by fac to avoid issues with numerical precision in fit.
    xarr = np.array([xarr]).flatten()
    return v*bu.get_es_force(xarr*units, is_fixed = True).flatten() + f0

def cham_es(xarr, beta, v, f0):
    #Combines chameleon force and es force
    return cham_force_beta(xarr, beta, f0) + es_force_v(xarr, v, 0.)

def plotter(plt_dict, fcal = cal):
    #Plots the average force vs position for each directory.
    arr = np.array(plt_dict.keys())
    for st in arr[arr != 'osets']:
        plt_vec = plt_dict[st]
        plt.errorbar(plt_vec[0], plt_vec[1]*fcal, plt_vec[2]*fcal, fmt = 'o', markersize = 5, label = [st][0] + 'V')
    plt.xlabel('Distance from microsphere')
    plt.ylabel('Attractive force [N]')

def fit_vec_cal(fit_vec, fcal):
    out_vec = fit_vec
    out_vec[1:, :] = out_vec[1:, :]*fcal
    return out_vec

def cham_fitter(fit_dict, fcal = cal):
    #fits the average force vs position for each directory to a chameleon force to estimate beta.
    arr = np.array(fit_dict.keys())
    for st in arr[arr != 'osets']:
        fit_vec = fit_vec_cal(fit_dict[st], fcal)
        p0 = [1., 0.]
        popt, pcov = curve_fit(cham_force_beta, fit_vec[0], fit_vec[1], p0 = p0, sigma = fit_vec[2])
        fitobj = cu.Fit(popt, pcov, cham_force_beta)
        f, axarr = plt.subplots(2, sharex = True)
        fitobj.plt_fit(fit_vec[0], fit_vec[1], axarr[0], xlabel = "Distance from microsphere [$\mu m$]", ylabel = "Attractive force [$nN$]", errors = fit_vec[2])
        fitobj.plt_residuals(fit_vec[0], fit_vec[1], axarr[1], xlabel = "Distance from microsphere [$\mu m$]", ylabel = "Residual force [$nN$]", errors = fit_vec[2])
        plt.show()
        return fitobj

def es_fitter(fit_dict, fun = es_force_v,fcal = cal, ft_range = [20, 100]):
    #fits the average force vs position for each directory to a chameleon force to estimate beta.
    arr = np.array(fit_dict.keys())
    f, axarr = plt.subplots(2, sharex = True)
    fit_objs = []
    for st in arr[arr == '4.0']:
        fit_vec = fit_vec_cal(fit_dict[st], fcal)
        b = fit_vec[0, :]>ft_range[0] and fit_vec[0, :]<ft_range[1]
        p0 = [1e12, 0.]
        fit_vec[0], fit_vec[1], fit_vec[2] = cu.sbin(fit_vec[0], fit_vec[1], 10.)
        popt, pcov = curve_fit(fun, fit_vec[0], fit_vec[1], p0 = p0, sigma = fit_vec[2])
        fitobj = cu.Fit(popt, pcov, es_force_v)
        
        fitobj.plt_fit(fit_vec[0], fit_vec[1], axarr[0], xlabel = "Distance from microsphere [$\mu m$]", ylabel = "Attractive force [$N$]", errors = fit_vec[2])
        fitobj.plt_residuals(fit_vec[0], fit_vec[1], axarr[1], xlabel = "Distance from microsphere [$\mu m$]", ylabel = "Residual force [$nN$]", errors = fit_vec[2])
        #plt.show()
        fit_objs.append(fitobj)
    return fit_objs


def cham_es_fitter(fit_dict, fun = cham_es, fcal = cal):
    #fits the average force vs position for each directory to a chameleon force to estimate beta.
    arr = np.array(fit_dict.keys())
    f, axarr = plt.subplots(2, sharex = True)
    fit_objs = []
    for st in arr[arr == '0.0']:
        init_vec = fit_vec_cal(fit_dict[st], fcal)
        fit_vec = [[], [], []]
        fit_vec[0], fit_vec[1], fit_vec[2] = cu.sbin(init_vec[0], init_vec[1], 10.)
        fit_vec = np.array(fit_vec)
        p0 = [0., 0., 0.]
        popt, pcov = curve_fit(fun, fit_vec[0, :], fit_vec[1, :], p0 = p0, sigma = fit_vec[2, :])
        fitobj = cu.Fit(popt, pcov, fun)
        
        fitobj.plt_fit(fit_vec[0], fit_vec[1], axarr[0], xlabel = "Distance from microsphere [$\mu m$]", ylabel = "Attractive force [$nN$]", errors = fit_vec[2])
        fitobj.plt_residuals(fit_vec[0], fit_vec[1], axarr[1], xlabel = "Distance from microsphere [$\mu m$]", ylabel = "Residual force [$nN$]", errors = fit_vec[2])
        #plt.show()
        fit_objs.append(fitobj)
    return fit_objs

#def chi_sq_beta(fit_dict)


def ave_f_dict(dir_obj):
    #Extraces force vs position info from dir object.
    return dir_obj.ave_force_vs_pos



def plt_fobj(fobj, volt_to_plt = 0.05, axis = 1, cant_indx = 24, label = ''):
    #Function to make the average force vs position plot for a single file object.
    plt_vec = [fobj.binned_cant_data[axis], fobj.binned_pos_data[axis], fobj.binned_data_errors[axis]]
    if fobj.electrode_settings[cant_indx] == 0.05: 
        plt.errorbar(plt_vec[0], plt_vec[1]*cal, plt_vec[2]*cal, fmt = 'o', label = "file number {}".format(label))
    plt.xlabel("Distance from Microsphere [$\mu m$]")
    plt.ylabel("Attractive Force [N]")

def sort_fun(dobj):

    #Function to sort the driectory objects based on closed z separation.
    return dobj.sep[2]



ks = lambda di: di.keys()


#calculate and correct for offset.
def oset_correct(ave_f_dicts):
    volts = map(ks, ave_f_dicts) #get all of the voltage keys.
    all_volts = list(set(volts[0]).intersection(*volts[1:]))
    #sel_keys = lambda di: 
    out_dict = {}
    osets = []
    for k in all_volts:
        out_dict[k] =ave_f_dicts[0][k]
        total_oset = 0.
        for i, v in enumerate(ave_f_dicts[:-1]):
            oset = ave_force_offset(ave_f_dicts[i][k], ave_f_dicts[i+1][k])
            total_oset += oset.x
            osets.append(osets)
            outvec = np.array(ave_f_dicts[i +1][k])
            outvec[1, :] += total_oset
            out_dict[k] = np.append(out_dict[k], outvec, axis = 1)

    out_dict['osets'] = osets
    return out_dict

def step_cal(dir_objs, step_indx = 0):
    #Charge step calibratioon to get from voltage to force units.
    dir_objs[step_indx].step_cal(dir_objs[step_indx])
    print dir_objs[step_indx].charge_step_calibration.popt

if calibrate:
    cal_dir_obj = proc_dir(cal_dir)
    step_cal([cal_dir_obj])

dir_objs = map(proc_dir, dirs)
dir_objs = sorted(dir_objs, key = sort_fun, reverse = True)
ave_f_dicts = map(ave_f_dict, dir_objs)
out_dict = oset_correct(ave_f_dicts)
fit_objs = cham_es_fitter(out_dict)
print fit_objs[0].popt
#step_cal(dir_objs)
#print fitobj.popt
#for i, fobj in enumerate(dir_objs[-1].fobjs):
    #plt_fobj(fobj, label = str(i))
plt.legend()
#plt.title('Cantilever biased at 0.05V')
plt.show()

