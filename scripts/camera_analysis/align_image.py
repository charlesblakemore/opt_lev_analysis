import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import scipy
import glob
from scipy.optimize import curve_fit
import cant_util as cu    

data_dir1 = "/data/20170831/image_calibration2/align_profs"
data_dir2 = "/data/20170831/image_calibration2/align_profs"
out_dir = "/calibrations/image_alignments"
date = '20170831'

def get_stage_column(attribs, stage_cols = [17, 18, 19], attrib_inds = [3, 6, 9], ortho_columns = [18, 17, 19]):
    '''gets the first driven stage axis from data attribs'''
    stage_settings = attribs['stage_settings']
    driven = np.array(map(bool, stage_settings[attrib_inds]))
    return (np.array(stage_cols)[driven])[0], (np.array(ortho_columns)[driven])[0]

def gauss_beam(r, mu, w, A):
    '''gaussian beam function for fitting'''
    return A*np.exp(-2.*(r-mu)**2/w**2)

def line(x, m, b):
    '''line function for fitting'''
    return m*x + b

def line_intersection(popt0, popt1):
    '''the intersection of 2 lines where y=mx+b and popt = [m, b]'''
    x_int = (popt1[1]-popt0[1])/(popt0[0]-popt1[0])
    return x_int, line(x_int, *popt0)


def profile(fname, ends = 100, stage_cal = 8., data_column = 5, make_plot = False, p0 = [30, 30, .001], ortho_column = [18, 17, 19]):
    '''takes raw data makes profile and fits to gaussian to determine beam center. returns beam center and position on orthogonal beam axis'''
    dat, attribs, f = bu.getdata(fname)
    dat = dat[ends:-ends, :]
    stage_column, ortho_column = get_stage_column(attribs)
    dat[:,stage_column]*=stage_cal
    dat[:, ortho_column]*=stage_cal
    f.close()
    bp, yp, ep = cu.sbin_pn(dat[:, stage_column], dat[:, data_column], bin_size = .1, vel_mult = 1.)
    bn, yn, en = cu.sbin_pn(dat[:, stage_column], dat[:, data_column], bin_size = .1, vel_mult = -1.)
    profp = np.abs(np.gradient(yp, bp))
    profn = np.abs(np.gradient(yn, bn))
    poptp, pcovp = curve_fit(gauss_beam, bp[10:-10], profp[10:-10], p0 = p0)
    poptn, pcovn = curve_fit(gauss_beam, bn[10:-10], profn[10:-10], p0 = p0)
    if make_plot:
        plt.semilogy(bp, profp, 'o')
        plt.semilogy(bp, gauss_beam(bp, *poptp), 'r')
        plt.semilogy(bn, profn, 'o')
        plt.semilogy(bn, gauss_beam(bn, *poptn), 'k')
        plt.show()
    return np.mean([poptn[0], poptp[0]]), np.mean(dat[:, ortho_column])


def find_edge(xsweep_dir, ysweep_dir, over_plot = 10.):
    xfs = glob.glob(xsweep_dir + '/*.h5')
    yfs = glob.glob(ysweep_dir + '/*.h5')
    xdata = np.array(map(profile, xfs))
    ydata = np.array(map(profile, yfs))
    plt.plot(xdata[:, 0], xdata[:, 1], 'x')
    plt.plot(ydata[:, 1], ydata[:, 0], 'x')
    p0x = [xdata[-1, 0]-xdata[0, 0]/(xdata[-1, 1]-xdata[0, 1]), 0]
    p0y = [ydata[-1, 0]-ydata[0, 0]/(ydata[-1, 1]-ydata[0, 1]), 0]
    poptx, pcovx = curve_fit(line, xdata[:, 0], xdata[:, 1], p0 = p0x)
    popty, pcovy = curve_fit(line, ydata[:, 1], ydata[:, 0], p0 = p0y)
    xplt = np.linspace(np.min(xdata[:, 0])-over_plot, np.max(xdata[:, 0])+over_plot, 1000)
    yplt = np.linspace(np.min(ydata[:, 1])-over_plot, np.max(ydata[:, 1])+over_plot, 1000) 
    plt.plot(xplt, line(xplt, *poptx))
    plt.plot(yplt, line(yplt, *popty))
    xint, yint = line_intersection(poptx, popty)
    plt.plot([xint], [yint], 'o')   
    plt.show()
    return np.array([xint, yint])

def save_cal(p_arr, path, date):
    #Makes path if it does not exist and saves parr to path/stage_position.npy
    if not os.path.exists(path):
        os.makedirs(path)
    outfile = os.path.join(path, 'stage_position_' + date)
    np.save(outfile, p_arr)

p_arr = find_edge(data_dir1, data_dir2)
save_cal(p_arr, out_dir, date)


