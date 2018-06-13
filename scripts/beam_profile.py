import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import scipy
import glob
from scipy.optimize import curve_fit


data_dir1 = "/data/20180529/imaging_tests/p0/xprofile"

def spatial_bin(xvec, yvec, bin_size = .13):
    fac = 1./bin_size
    bins_vals = np.around(fac*xvec)
    bins_vals/=fac
    bins = np.unique(bins_vals)
    y_binned = np.zeros_like(bins)
    y_errors = np.zeros_like(bins)
    for i, b in enumerate(bins):
        idx = bins_vals == b
        y_binned[i] =  np.mean(yvec[idx])
        y_errors[i] = scipy.stats.sem(yvec[idx])
    return bins, y_binned, y_errors
    
        
def gauss(x, A, mu, sig):
    '''gaussian fitting function'''
    return A*np.exp(-1.*(x-mu)**2/(2.*sig**2))

def profile(fname, data_column = 1):
    df = bu.DataFile()
    df.load(fname, load_FPGA = False)
    df.load_other_data()
    df.calibrate_stage_position()
    if 'ysweep' in fname:
        stage_column = 1
        if 'left' in fname:
            sign = -1.0
        elif 'right' in fname:
            sign = 1.0
        else:
            sign = 1.0
    else:
        stage_column = 0
        sign = 1.0

    b, a = sig.butter(1, 0.5)
    int_filt = sig.filtfilt(b, a, df.other_data[data_column, :])
    proft = np.gradient(int_filt)
    stage_filt = sig.filtfilt(b, a, df.cant_data[stage_column, :])
    dir_sign = np.sign(np.gradient(stage_filt)) * sign
    xvec = df.cant_data[stage_column, :]
    yvec = (proft - proft * dir_sign) * 0.5 - (proft + proft * dir_sign) * 0.5
    b, y, e = spatial_bin(xvec, yvec)
    return b, y, e

class File_prof:
    "Class storing information from a single file"
    
    def __init__(self, b, y, e):
        self.bins = b
        self.dxs = np.append(np.diff(b), 0)#0 pad left trapizoid rule
        self.y = y
        self.errors = e
        self.mean = "mean not computed"
        self.sigmasq = "std dev not computed"
        self.date = "date not entered"
        
    def dist_mean(self, roi = [-10, 10]):
        #Finds the cnetroid of intensity distribution iteratively. First over all the data then the centroind over the region of interst 
        norm = np.sum(self.y*self.dxs)
        c1 = np.sum(self.dxs*self.y*self.bins)/norm
        lebin = np.argmin((self.bins - (c1+roi[0]))**2)
        rebin = np.argmin((self.bins - (c1+roi[1]))**2)
        norm2 = np.sum(self.y[lebin:rebin]*self.dxs[lebin:rebin])
        c2 = np.sum(self.dxs[lebin:rebin]*self.y[lebin:rebin]*\
                self.bins[lebin:rebin])/norm2
        self.mean = c2

    def sigsq(self):
        #finds second moment of intensity distribution.
        if type(self.mean) == str:
            self.dist_mean()
        derp1 = self.bins > ROI[0]
        derp2 = self.bins < ROI[1]
        ROIbool = np.array([a and b for a, b in zip(derp1, derp2)])
        norm = np.sum(self.y[ROIbool]*self.dxs[ROIbool])
        self.sigmasq = np.sum(self.bins[ROIbool]**2*self.y[ROIbool])/norm

    def sigsq2(self, p0 = [1., 0., 3.], make_plot = False, plt_region = [-10, 10]):
        '''finds second moment by fitting to gaussian'''
        if type(self.mean) == str:
            self.dist_mean()
        popt, pcov = curve_fit(gauss, self.bins, self.y, p0 = p0)
        if make_plot:
            pb = (self.bins<plt_region[1])*(self.bins>plt_region[0])
            plt.semilogy(self.bins[pb], self.y[pb], 'o')
            plt.semilogy(self.bins[pb], gauss(self.bins[pb], *popt), 'r')
            plt.show()
        self.sigmasq = popt[-1]**2
        
        
         

def proc_dir(dir):
    files = glob.glob(dir + '/*.h5')
    file_profs = []
    cents = []
    for fi in files:
        b, y, e = profile(fi)
        f = File_prof(b, y, e)
        f.date = dir[8:16]
        file_profs.append(f)
        f.dist_mean()
        cents.append(f.mean)
        
    return cents, file_profs
 
def plot_profs(fp_arr):
    #plots different profiles
    i = 1
    for fp in fp_arr:
        plt.plot(fp.bins, fp.y / np.max(fp.y), 'o')
        plt.ylim(10**(-5), 10)
    plt.xlabel("position [um]")
    plt.ylabel("margenalized irradiance ~[W/m]")
    plt.gca().set_yscale('linear')
    plt.show()

def find_beam_crossing(directory, make_plot = True):
    cents, fps = proc_dir(directory)
    cmean = np.mean(cents)
    error = scipy.stats.sem(cents)
    if make_plot:
        plot_profs(fps)
    return cmean, error



