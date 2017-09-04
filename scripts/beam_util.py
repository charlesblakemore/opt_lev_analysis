#################################################################################Helper functions and classes for analyzing beam profiles
###############################################################################

import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import scipy
import glob
from scipy.optimize import curve_fit
import bead_util as bu
from scipy import interpolate
    

def copy_attribs(attribs):
    '''copies attribs from hdf5 objects so they can be accessed after file is closed'''
    out_dict = {}
    for k in attribs.keys():
        out_dict[k] = attribs[k]
    return out_dict

def make_stage_dict(stage_settings):
    '''makes dictionary from stage settings for easier interpretation. organized into dictionary with format stage_dict['drive'] = [xdrive?, ydrive?, zdrive?], stage_dict['dc'] = [dcx, dcy, dcz], stage_dict['amplitude'] = [xamp, yamp, zamp], stage_dict['freq'] = [fx, fy, fz]'''
    stage_dict = {}
    stage_dict['drive'] = np.array([stage_settings[3], stage_settings[6], stage_settings[9]])
    stage_dict['dc'] = np.array([stage_settings[0], stage_settings[1], stage_settings[2]])
    stage_dict['amplitude'] = np.array([stage_settings[4], stage_settings[7], stage_settings[10]])
    stage_dict['freq'] = np.array([stage_settings[5], stage_settings[8], stage_settings[11]])
    return stage_dict

def gauss_wconst(x, A, x0, w0, C):
    return A * np.exp( -2 * (x-x0)**2 / (w0**2) ) + C
        
def gauss(x, A, x0, w0):
    return A * np.exp( -2 * (x-x0)**2 / (w0**2) )

def mask(arr, ind, n_harmonic):
    '''sets all values not at an integer multiple of ind to 0'''
    hars = np.arange(1, n_harmonic)*ind
    for i in range(len(arr)):
        if i in hars:
            arr[i] = arr[i]
        else:
            arr[i] = 0
    return arr

def generate_profile(intensity_dat, stage_dat, drive_freq, Fs, n_harmonic = 1000, vmult = -1.0, n_out = 5000):
    '''given intensity and stage data returns stage position vector and interpoating function of the derivative of the blocked light'''
    dc  = np.mean(stage_dat)
    freqs = np.fft.rfftfreq(len(intensity_dat), 1./Fs)
    drive_ind = np.argmin(np.abs(drive_freq-freqs))
    i_fft = np.fft.rfft(intensity_dat)
    i_dat = np.fft.irfft(mask(i_fft, drive_ind, n_harmonic))
    s_fft = np.fft.rfft(stage_dat)
    s_dat = np.fft.irfft(mask(s_fft, drive_ind, n_harmonic))
    if vmult == -1.:
        v_bool = np.gradient(s_dat)<0.
    elif vmult == 0.:
        v_bool = np.ones(len(s_dat))
    elif vmult == 1.:
        v_bool = np.gradient(s_dat)>0.
    #sort for interpolating 
    dat_arr = np.transpose(np.vstack((s_dat[v_bool], i_dat[v_bool])))
    dat_arr = np.transpose(dat_arr[dat_arr[:, 0].argsort()])
    #interpolate to resample
    f = interpolate.interp1d(dat_arr[0] + dc, np.abs(np.gradient(dat_arr[1])))
    s_out = np.linspace(np.min(dat_arr[0]), np.max(dat_arr[0]), n_out) + dc
    return s_out, f
     

class Profile:
    'class representing a beam profile  derived from a file'
       
    def __init__(self, fname, data_column = 5, ends = 100, stage_cal = 8., stage_cols = [17, 18, 19]):
        dat, i_attribs, f = bu.getdata(fname)
        self.attribs = copy_attribs(i_attribs)
        self.dat = dat[ends:-ends, :]
        self.stage_dict = make_stage_dict(self.attribs['stage_settings'])
        driven = self.stage_dict['drive']>0.
        self.drive_column = np.array(stage_cols)[driven][0] #be careful if multi drive
        self.fdrive = self.stage_dict['freq'][driven][0]
        dat[:,self.drive_column]*=stage_cal
        f.close()
        stage, f = generate_profile(dat[:, data_column], dat[:, self.drive_column], self.fdrive, self.attribs['Fsamp'])
        self.stage = stage
	self.prof_f = f        
        self.mean = "mean not computed"
        self.sigmasq = "std dev not computed"
        self.date = "date not entered"
        self.dxs = np.gradient(self.stage)
        
    def dist_mean(self, ROI = [5., 75.]):
        #Finds the cnetroid of intensity distribution. subtracts centroid from bins
        ROIb = (self.stage>ROI[0])*(self.stage<ROI[1])
        norm = np.sum(self.prof_f(self.stage[ROIb])*self.dxs[ROIb])
        self.mean = np.sum(self.dxs[ROIb]*self.prof_f(self.stage[ROIb])*self.stage[ROIb])/norm
        self.stage -= self.mean

    def sigsq(self, ROI = [0., 80.]):
        #finds second moment of intensity distribution.
        if type(self.mean) == str:
            self.dist_mean()
        derp1 = self.stage > ROI[0]
        derp2 = self.stage < ROI[1]
        ROIbool = np.array([a and b for a, b in zip(derp1, derp2)])
        norm = np.sum(self.prof_f(self.stage[ROIbool])*self.dxs[ROIbool])
        #norm = np.sum(self.y*self.dxs)
        self.sigmasq = np.sum(self.stage[ROIbool]**2*self.prof_f(self.stage[ROIbool]))/norm
         

def proc_dir(dir):
    files = glob.glob(dir + '/*.h5')
    #print files
    profs = []
    for fi in files:
        p = Profile(fi)
        p.sigsq()
        profs += [p]
    return profs

def mean_vs_dcp(profs):
    '''extracts mean and cantilever mean position from Profile object'''
    ms = []
    dc = []
    for p in profs:
        ms += [p.mean]
        dc += [p.stage_dict['dc']]
    return np.array(ms), np.array(dc)
 
def plot_profs(fp_arr):
    #plots average profile from different heights
    i = 1
    for fp in fp_arr:
        #plt.errorbar(fp.bins, fp.y, fp.errors, label = str(np.round(fp.cant_height)) + 'um')
        if multi_dir:
            lab = 'dir' + str(i)
        else:
            lab = str(np.round(fp.cant_height)) + 'um'
        i += 1
        if multi_dir:
            plt.plot(fp.bins, fp.y / np.max(fp.y), 'o', label = lab)
            plt.ylim(10**(-5), 10)
        else:
            plt.plot(fp.bins, fp.y, 'o', label = lab)
    plt.xlabel("position [um]")
    plt.ylabel("margenalized irradiance ~[W/m]")
    if log_profs:
        plt.gca().set_yscale('log')
    else:
        plt.gca().set_yscale('linear')
    plt.legend()
    plt.show()


def Szsq(z, s0, M, z0, lam = 1.064):
    #function giving propigation of W=2sig parameter. See Seegman
    W0 = 2.*s0
    Wzsq = W0**2 + M**4*(lam/(np.pi*W0))**2*(z-z0)**2
    return Wzsq/4.

def line(x, m, b):
    '''a line function for fitting.'''
    return m*x + b

def find_angle(pos, cent, make_plot = True, disp_digits = 3):
    '''fits centroid positiion vs cantilever step position to determine cantilever angle'''
    popt, pcov = curve_fit(line, pos, cent)
    var = pcov[0, 0] 
    Q = np.arctan(popt[0])
    sq = np.sqrt(var/(1. + popt[0]**2)**2)
    if make_plot:
        plot_x = np.linspace(np.min(pos), np.max(pos), 100)
        plt.plot(pos, cent, 'o')
        label_str =  "angle =" + str(Q)[:disp_digits + 3] +'$\pm$'+ str(sq)[:disp_digits + 2] + 'rad ' + 'os:'+ str(popt[1])[:disp_digits + 1]
        plt.plot(plot_x, line(plot_x, *popt), 'r', label =label_str)
        plt.xlabel('Step position[um]')
        plt.ylabel('centroid position [um]')
        plt.legend()
        plt.show()
    return Q  
    
    

#def compute_msquared(hs, sigmasqs):
    #fits beam profile data to extract M^2 value 


file_profs, hs, sigmasqs = proc_dir(data_dir1)

if multi_dir:
    fp2, hs2, sigsq2 = proc_dir(data_dir2)
    ind = np.argmin(np.abs(hs - height_to_plot))
    ind2 = np.argmin(np.abs(hs2 - height_to_plot))
    plot_profs([file_profs[ind]] + [fp2[ind2]])

else:
    plot_profs(file_profs)

if msq_fit:

    p0 = [5., 10., 40.]

    bfit = hs < 140.

    popt, pcov = curve_fit(Szsq, hs[bfit], sigmasqs[bfit], p0=p0, maxfev=10000)
    hplt = np.arange(np.min(hs), np.max(hs), 0.1)
    
    if multi_dir:
        bfit2 = hs2 < 140.
        popt2, pcov2 = curve_fit(Szsq, hs2[bfit2], sigsq2[bfit2], p0=p0, maxfev=10000)
        hplt2 = np.arange(np.min(hs2), np.max(hs2), 0.1)
        
        fig, axarr = plt.subplots(2, sharex=True, sharey=True)
        ax1 = axarr[0]
        ax2 = axarr[1]

    else:
        fig, axarr = plt.subplots(1, sharex=True)
        ax1 = axarr

        
    if multi_dir:    
        maxsig = np.max([np.max(sigmasqs), np.max(sigsq2)])
    else:
        maxsig = np.max(sigmasqs)
    
    ax1.plot(hs, sigmasqs, 'o')
    ax1.plot(hplt, Szsq(hplt, *popt), 'r',linewidth = 2,  label = "M^2=%0.3g"%popt[1]**2)
    ax1.set_title("Trap Focus at h = %g um, Waist w0 = %0.2g um"%(popt[-1],popt[0]))
    ax1.set_ylabel("second moment [um^2]")
    ax1.set_ylim(0, maxsig*1.2)
    ax1.legend(loc=0)

    if multi_dir:
        ax2.plot(hs2, sigsq2, 'o')
        ax2.plot(hplt2, Szsq(hplt2, *popt2), 'r',linewidth = 2,  label = "M^2=%0.3g"%popt2[1]**2)
        ax2.set_title("Trap Focus at h = %g um, Waist w0 = %0.2g um"%(popt2[-1],popt2[0]))
        ax2.set_ylabel("Second moment [um^2]")
        ax2.set_xlabel("Cantilever height [um]")
        ax2.set_ylim(0, maxsig*1.2)
        ax2.legend(loc=0)
    plt.show()


if gauss_fit:

    if multi_dir:
        f2, axarr2 = plt.subplots(2, sharex=True, sharey=True)
        ax1 = axarr2[0]
        ax2 = axarr2[1]
    else:
        f2, axarr2 = plt.subplots(1, sharex=True)
        ax1 = axarr2

    def gauss_wconst(x, A, x0, w0, C):
        return A * np.exp( -2 * (x-x0)**2 / (w0**2) ) + C
        
    def gauss(x, A, x0, w0):
        return A * np.exp( -2 * (x-x0)**2 / (w0**2) )
    
    if msq_fit:
        bestfit = np.argmin(np.abs(np.array(hs) - popt[-1]))
        if multi_dir:
            bestfit2 = np.argmin(np.abs(np.array(hs2) - popt2[-1]))
    else:
        bestfit = 0
        if multi_dir:
            bestfit2 = 0
    
    lab = hs[bestfit]
    if multi_dir:
        lab2 = hs2[bestfit2]
                        
    bestprof = file_profs[bestfit]
    if multi_dir:
        bestprof2 = fp2[bestfit2]

    #p02 = [10**(-3), 0, 10, 10**(-7)]
    #popt2, pcov2 = curve_fit(gauss_wconst, bestprof.bins, bestprof.y, p0=p02)

    p02 = [10**(-3), 0, 10]    
    popt3, pcov3 = curve_fit(gauss, bestprof.bins, bestprof.y, p0=p02)
    fitpts = np.arange(np.min(bestprof.bins), np.max(bestprof.bins), 0.1)
    
    if multi_dir:
        popt4, pcov4 = curve_fit(gauss, bestprof2.bins, bestprof2.y, p0=p02)
        fitpts2 = np.arange(np.min(bestprof2.bins), np.max(bestprof2.bins), 0.1)

        
    ax1.plot(bestprof.bins, bestprof.y, 'o')
    ax1.plot(fitpts, gauss(fitpts, *popt3), 'r', linewidth = 2, label='h = %0.2g' % lab)

    data_int = np.sum(bestprof.y) * (bestprof.bins[1] - bestprof.bins[0])
    gauss_int = np.sum(gauss(fitpts, *popt3)) * (fitpts[1] - fitpts[0])

    print 
    print "Non-Gaussian Part (1): ", (data_int - gauss_int) / data_int

    ax1.set_title('Gaussian Fit Waist = %0.2g um' % (np.abs(popt3[2])) )
    ax1.set_ylabel("Intensity Profile [arbitrary]")
    ax1.set_ylim(10**(-6), popt3[0] * 10)
    ax1.set_yscale('log')
    ax1.legend(loc=0)
    
    if multi_dir:
        ax2.plot(bestprof2.bins, bestprof2.y, 'o')
        ax2.plot(fitpts2, gauss(fitpts2, *popt4), 'r', linewidth = 2, label='h = %0.2g' % lab)
    
        data_int2 = np.sum(bestprof2.y) * (bestprof2.bins[1] - bestprof2.bins[0])
        gauss_int2 = np.sum(gauss(fitpts2, *popt4)) * (fitpts2[1] - fitpts2[0])
        print "Non-Gaussian Part (2): ", (data_int2 - gauss_int2) / data_int2

        ax2.set_title('Gaussian Fit Waist = %0.2g um' % (np.abs(popt4[2])) )
        ax2.set_xlabel("Cantilever Position [um]")
        ax2.set_ylabel("Intensity Profile [arbitrary]")
        ax2.set_ylim(10**(-6), popt4[0] * 10)
        ax2.set_yscale('log')
    
    plt.show()





