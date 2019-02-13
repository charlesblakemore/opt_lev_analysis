import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import scipy
import glob
from scipy.optimize import curve_fit


data_dir1 = r"C:\Data\20170704\profiling\ysweep5"
data_dir2 = r"C:\Data\20170704\profiling\zsweep5"
out_dir = r"C:\Data\20170704\profiling\output"
#data_dir2 = r"C:\Data\20160429\beam_profiles1"


data_dir1 = "/data/20181218/profiling/xsweep_vac"
data_dir1 = "/data/20190103/profiling/xsweep_adj_4"

data_dir1 = "/data/20190107/profiling/xsweep_init"

data_dir1 = "/data/20190108/profiling/xsweep_atm_adj_2"
data_dir2 = "/data/20190108/profiling/xsweep_atm"

multi_dir = True
height_to_plot = 40.

log_profs = True

ROI = [-80, 80] # um
#OFFSET = 2.*10**(-5)
OFFSET = 0

msq_fit = True
gauss_fit = True

#stage x = col 17, stage y = 18, stage z = 19
stage_column = 19
stage_column2 = 18

data_column = 4
data_column2 = 0  # For data circa 2016

cant_cal = 8. #um/volt


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
    
        
    



def profile(fname, data_column = 0):
    df = bu.DataFile()
    df.load(fname)
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
    #shape = np.shape(df.other_data)
    #for i in range(shape[0]):
    #    plt.plot(df.other_data[i, :], label = str(i))
    #plt.legend()
    #plt.show()

    h = np.mean(df.cant_data[2, :])
    h_round = bu.round_sig(h, sig=2)

    if h_round < 10.0:
        h_round = bu.round_sig(h_round, sig=1)

    int_filt = sig.filtfilt(b, a, df.other_data[data_column, :])
    proft = np.gradient(int_filt)

    #plt.plot(df.other_data[0])
    #plt.show()

    stage_filt = sig.filtfilt(b, a, df.cant_data[stage_column, :])
    dir_sign = np.sign(np.gradient(stage_filt)) * sign
    xvec = df.cant_data[stage_column, :]
    yvec = (proft - proft * dir_sign) * 0.5 - (proft + proft * dir_sign) * 0.5
    b, y, e = spatial_bin(xvec, yvec)
    return b, y, e, h_round



class File_prof:
    "Class storing information from a single file"
    
    def __init__(self, b, y, e, h):
        self.bins = b
        self.dxs = np.append(np.diff(b), 0)#0 pad left trapizoid rule
        self.y = y
        self.errors = e
        self.cant_height = h
        self.mean = "mean not computed"
        self.sigmasq = "std dev not computed"
        self.date = "date not entered"
        
    def dist_mean(self):
        #Finds the cnetroid of intensity distribution. subtracts centroid from bins
        norm = np.sum(self.y*self.dxs)
        self.mean = np.sum(self.dxs*self.y*self.bins)/norm
        self.bins -= self.mean

    def sigsq(self):
        #finds second moment of intensity distribution.
        if type(self.mean) == str:
            self.dist_mean()
        derp1 = self.bins > ROI[0]
        derp2 = self.bins < ROI[1]
        ROIbool = np.array([a and b for a, b in zip(derp1, derp2)])
        norm = np.sum(self.y[ROIbool]*self.dxs[ROIbool])
        #norm = np.sum(self.y*self.dxs)
        self.sigmasq = np.sum(self.bins[ROIbool]**2*self.y[ROIbool])/norm
        #self.sigmasq = np.sum(self.bins**2*self.y)/norm
         

def proc_dir(dir):
    files = glob.glob(dir + '\*.h5')
    files, lengths = bu.find_all_fnames(dir)
    #print files
    file_profs = []
    hs = []
    for fi in files:
        b, y, e, h = profile(fi)
        #print h
        if h not in hs:
            #if new height then create new profile object
            hs.append(h)
            f = File_prof(b, y, e, h)
            f.date = dir[8:16]
            file_profs.append(f)
        else:
            #if height repeated then append data to object for that height
            for fp in file_profs:
                if fp.cant_height == h:
                    fp.bins = np.append(fp.bins, b)
                    fp.y = np.append(fp.y, y)
                    fp.errors = np.append(fp.errors, e)
            
    #now rebin all profiles
    for fp in file_profs:
        b, y, e = spatial_bin(fp.bins, fp.y)
        fp.bins = b
        fp.y = y
        fp.errors = e
        fp.dxs = np.append(np.diff(fp.bins), 0)#0 pad left trapizoid rule

    sigmasqs = []
    hs = []

    for f in file_profs:
        f.sigsq()
        sigmasqs.append(f.sigmasq)
        hs.append(f.cant_height)
        
    return file_profs, np.array(hs), np.array(sigmasqs)
 
def plot_profs(fp_arr):
    #plots average profile from different heights
    i = 1
    colors = bu.get_color_map(len(fp_arr), cmap='jet')

    fp_arr_sort = sorted(fp_arr, key = lambda fp: fp.cant_height)

    for fp_ind, fp in enumerate(fp_arr_sort):
        color = colors[fp_ind]
        #plt.errorbar(fp.bins, fp.y, fp.errors, label = str(np.round(fp.cant_height)) + 'um')
        if multi_dir:
            lab = 'dir' + str(i)
        else:
            lab = str(np.round(fp.cant_height)) + 'um'
        i += 1
        if multi_dir:
            plt.plot(fp.bins, fp.y / np.max(fp.y), 'o', label = lab, color=color)
            plt.ylim(10**(-5), 10)
        else:
            plt.plot(fp.bins, fp.y, 'o', label = lab, color=color)
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





