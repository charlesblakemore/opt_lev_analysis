import numpy as np
import bead_util as bu

import matplotlib.pyplot as plt
import os, re
import scipy.signal as sig
import scipy
import glob

from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 14})


ysign = 0



# bu.configuration.col_labels["stage_pos"] = [17, 18, 19]
# data_dir1 = '/data/old_trap/20171026/profiling/xsweep_in2_5t_down1t_gf'
# data_dir2 = '/data/old_trap/20171026/profiling/ysweep_in2_5t_down1t'
# data_column = 5
# data_column2 = data_column
# ysign = 1.0


# data_dir1 = '/data/old_trap/20180514/profiling/xsweep_final'
# data_dir2 = '/data/old_trap/20180514/profiling/ysweep_final'
# data_column = 3
# data_column2 = data_column
# ysign = -1.0

# data_dir1 = '/data/old_trap/20190315/profiling/xsweep_adj13_atm'
# data_dir2 = '/data/old_trap/20190315/profiling/ysweep_right_adj13_atm'
# data_column = 1
# data_column2 = data_column


# data_dir1 = '/data/old_trap/20190902/profiling/post_trans/xsweep_vac_init'
# data_dir2 = '/data/old_trap/20190902/profiling/post_trans/ysweep_vac_init'

# data_dir1 = '/data/old_trap/20190902/profiling/xsweep_vac_out3_0'
# data_dir2 = '/data/old_trap/20190902/profiling/ysweep_vac_in2_0'


# data_dir1 = '/data/old_trap/20201202/profiling/xsweep_init'
# data_dir2 = '/data/old_trap/20201202/profiling/xsweep_init'


bu.configuration.col_labels["stage_pos"] = [1, 2, 3]
# data_dir1 = '/data/old_trap/20210624/profiling/xsweep_centered'
# data_dir1 = '/data/old_trap/20210624/profiling/ysweep_centered'
# data_dir1 = '/data/old_trap/20210624/profiling/ysweep_adj1'
# data_dir1 = '/data/old_trap/20210624/profiling/ysweep_adj4'
# data_dir1 = '/data/old_trap/20210624/profiling/xsweep_adj4'
# data_dir1 = '/data/old_trap/20210624/profiling/xsweep_adj8'
# data_dir1 = '/data/old_trap/20210624/profiling/ysweep_adj11'
# data_dir1 = '/data/old_trap/20210624/profiling/ysweep_adj12'
data_dir1 = '/data/old_trap/20220810/beam_profiling/xsweep_3Hz_init'

# data_dir1 = '/data/old_trap/20210628/profiling/ysweep_afterclean'
# data_dir2 = '/data/old_trap/20210628/profiling/ysweep_afterclean_nolights'
data_dir2 = '/data/old_trap/20210628/profiling/ysweep_adj9'
# data_dir1 = '/data/old_trap/20210628/profiling/xsweep_adj10'
# data_dir2 = '/data/old_trap/20210628/profiling/xsweep_adj11'
data_column = 0
data_column2 = data_column
ysign = 1.0


ideal_waist1 = 2.5   # um
ideal_waist2 = 2.5   # um


# debug_plot = True
debug_plot = False

# multi_dir = True
multi_dir = False

height_to_plot = 0.

log_profs = True

ROI = [-80.0, 80.0] # um
#OFFSET = 2.*10**(-5)
OFFSET = 0

msq_fit = True
gauss_fit = True




vec_inds = (100, -100)

nbin = 200


sigmasq_cutoff = 1e-3

baseline_fit = True
baseline_edge = 30.0






def Szsq(z, s0, M, z0, lam = 1.064):
    #function giving propigation of W=2sig parameter. See Seegman
    W0 = 2.*s0
    Wzsq = W0**2 + M**4 * (lam/(np.pi*W0))**2 * (z-z0)**2
    return Wzsq/4.

def gauss_wconst(x, A, x0, w0, C):
    return A * np.exp( -2 * (x-x0)**2 / (w0**2) ) + C
    
def gauss(x, A, x0, w0):
    return A * np.exp( -2 * (x-x0)**2 / (w0**2) )
    

    



def profile(fname, data_column = 0, plot=False, nbin=200):
    df = bu.DataFile()
    df.load(fname, skip_fpga=True)
    df.load_other_data()
    df.calibrate_stage_position()

    dt = 1.0 / df.fsamp

    if 'ysweep' in fname:
        stage_column = 1  
        if not ysign: 
            sign = 1.0
        else:
            sign = ysign
    else:
        stage_column = 0
        sign = 1.0

    b, a = sig.butter(1, 0.5)

    if plot:
        shape = np.shape(df.other_data)
        for i in range(shape[0]):
            plt.plot(df.other_data[i, :], label = str(i))
            plt.title('Data Columns')
        plt.legend()
        plt.tight_layout()

        plt.figure()
        for j in range(3):
            plt.plot(df.cant_data[j,:], label=str(j))
            plt.title('Attractor Coordinates')
        plt.legend()
        plt.tight_layout()

        plt.show()

        input()

    h = np.mean(df.cant_data[2, :])
    h_round = bu.round_sig(h, sig=3)

    if h_round < 10.0:
        h_round = bu.round_sig(h_round, sig=2)

    int_filt = sig.filtfilt(b, a, df.other_data[data_column])
    proft = np.gradient(int_filt)

    # proft = np.gradient(df.other_data[data_column])

    #plt.plot(df.other_data[0])
    #plt.show()

    stage_filt = sig.filtfilt(b, a, df.cant_data[stage_column, :])
    dir_sign = np.sign(np.gradient(stage_filt)) * sign

    dir_sign = np.sign(np.gradient(df.cant_data[stage_column])) * sign

    xvec = df.cant_data[stage_column, :]
    yvec = (proft - proft * dir_sign) * 0.5 - (proft + proft * dir_sign) * 0.5

    # sort_inds = np.argsort(xvec)

    # b, y, e = bu.spatial_bin(xvec, yvec, dt, nbin=300, nharmonics=300, add_mean=True)
    b, y, e = bu.rebin(xvec[vec_inds[0]:vec_inds[1]], \
                       yvec[vec_inds[0]:vec_inds[1]], \
                       nbin=nbin, plot=False, correlated_errs=True)

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

    def sigsq(self, cutoff=1e-9):
        #finds second moment of intensity distribution.
        if type(self.mean) == str:
            self.dist_mean()

        derp1 = self.bins > ROI[0]
        derp2 = self.bins < ROI[1]
        derp3 = self.y > cutoff * np.max(self.y)

        ROIbool = derp1 * derp2 * derp3

        num = np.sum(self.bins[ROIbool]**2 * self.y[ROIbool] * self.dxs[ROIbool])
        denom = np.sum(self.y[ROIbool]*self.dxs[ROIbool])
        #norm = np.sum(self.y*self.dxs)
        self.sigmasq = num / denom
        #self.sigmasq = np.sum(self.bins**2*self.y)/norm
         

def proc_dir(dir, data_column=0, plot=False):
    files, lengths = bu.find_all_fnames(dir)
    file_profs = []
    hs = []
    for fi in files:
        b, y, e, h = profile(fi, nbin=nbin, plot=plot, data_column=data_column)
        #print h
        if h not in hs:
            #if new height then create new profile object
            hs.append(h)
            f = File_prof(b, y, e, h)
            f.date = re.search(r"\d{8,}", dir)[0]
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

        sorter = np.argsort(fp.bins)
        fp.bins = fp.bins[sorter]
        fp.y = fp.y[sorter]
        fp.errors = fp.errors[sorter]
        fp.dxs = np.append(np.diff(fp.bins), 0)

        # if len(fp.bins) > nbin:
        #     b, y, e = bu.rebin(fp.bins, fp.y, nbin=nbin, correlated_errs=True)
        #     fp.bins = b
        #     fp.y = y
        #     fp.errors = e
        #     fp.dxs = np.append(np.diff(b), 0)#0 pad left trapizoid rule

    sigmasqs = []
    hs = []

    for f in file_profs:

        f.sigsq(cutoff=sigmasq_cutoff)
        sigmasqs.append(f.sigmasq)
        hs.append(f.cant_height)

        if baseline_fit:
            fit_inds = np.abs(f.bins) > baseline_edge
            baseline = np.mean( (f.y)[fit_inds] )
            f.y -= baseline
        
    return file_profs, np.array(hs), np.array(sigmasqs)


 
def plot_profs(fp_arr, title='', show=True):
    #plots average profile from different heights
    i = 1
    colors = bu.get_colormap(len(fp_arr), cmap='plasma')

    fp_arr_sort = sorted(fp_arr, key = lambda fp: fp.cant_height)

    plt.figure()
    for fp_ind, fp in enumerate(fp_arr_sort):
        color = colors[fp_ind]
        #plt.errorbar(fp.bins, fp.y, fp.errors, label = str(np.round(fp.cant_height)) + 'um')
        lab = str(np.round(fp.cant_height)) + 'um'
        if multi_dir:
            plt.plot(fp.bins, fp.y / np.max(fp.y), 'o', label = lab, color=color)
            plt.ylim(10**(-5), 10)
        else:
            plt.plot(fp.bins, fp.y, 'o', label = lab, color=color)
    plt.xlabel("Knife-edge Position [$\\mu$m]")
    plt.ylabel("Margenalized Irradiance [~W/m]")
    if log_profs:
        plt.gca().set_yscale('log')
    else:
        plt.gca().set_yscale('linear')
    plt.legend(loc='lower right', ncol=2)
    if title:
        plt.title(title)
    plt.tight_layout()

    heights = []
    means = []
    for fp_ind, fp in enumerate(fp_arr_sort):
        heights.append(fp.cant_height)
        means.append(fp.mean)

    plt.figure()
    plt.plot(heights, means, 'o', ms=10)
    plt.xlabel("Knife-edge Height [$\\mu$m]")
    plt.ylabel("Profile mean [$\\mu$m]")
    if title:
        plt.title(title)
    plt.tight_layout()


    if show:
        plt.show()


#def compute_msquared(hs, sigmasqs):
    #fits beam profile data to extract M^2 value 


file_profs, hs, sigmasqs = proc_dir(data_dir1, data_column=data_column, plot=debug_plot)





if multi_dir:
    fp2, hs2, sigsq2 = proc_dir(data_dir2, data_column=data_column2, plot=debug_plot)
    ind = np.argmin(np.abs(hs - height_to_plot))
    ind2 = np.argmin(np.abs(hs2 - height_to_plot))
    # plot_profs([file_profs[ind]] + [fp2[ind2]])
    plot_profs(file_profs, title='dir1', show=False)
    plot_profs(fp2, title='dir2')

else:
    plot_profs(file_profs)

if msq_fit:

    p0 = [5., 2.0, 40.]

    bfit = hs < 140.

    popt, pcov = curve_fit(Szsq, hs[bfit], sigmasqs[bfit], p0=p0, maxfev=10000)
    hplt = np.arange(np.min(hs), np.max(hs), 0.1)
    
    if multi_dir:
        bfit2 = hs2 < 140.
        popt2, pcov2 = curve_fit(Szsq, hs2[bfit2], sigsq2[bfit2], p0=p0, maxfev=10000)
        hplt2 = np.arange(np.min(hs2), np.max(hs2), 0.1)
        
        fig, axarr = plt.subplots(2, sharex=True, sharey=True, \
                                  figsize=(6,6))
        ax1 = axarr[0]
        ax2 = axarr[1]

    else:
        fig, axarr = plt.subplots(1, sharex=True)
        ax1 = axarr

        
    if multi_dir:    
        maxsig = np.max([np.max(sigmasqs), np.max(sigsq2)])
    else:
        maxsig = np.max(sigmasqs)
    
    popt_ideal = np.copy(popt)
    popt_ideal[0] = ideal_waist1 / 2
    popt_ideal[1] = 1.0
    # popt_ideal[2] = 40.0

    ax1.plot(hs, sigmasqs, 'o')
    ax1.plot(hplt, Szsq(hplt, *popt), 'r', linewidth=2, label="$M^2={:0.2g}$".format(popt[1]**2))
    ax1.plot(hplt, Szsq(hplt, *popt_ideal), 'r', linewidth=2, ls='--', alpha=0.6)
    ax1.set_title("Focus at $h = {:0.3g}~\\mu$m, 'Waist' $W_0 = {:0.2g}~\\mu$m"\
                        .format(popt[-1], 2*popt[0]), fontsize=14)
    # ax1.set_ylabel("$\\sigma_x^2$ [$\\mu{\\rm m}^2$]")
    ax1.set_ylabel("$\\sigma_{\\rm before}^2$ [$\\mu{\\rm m}^2$]")
    ax1.set_ylim(0, maxsig*1.2)
    ax1.legend(loc=0)

    if multi_dir:
    
        popt2_ideal = np.copy(popt2)
        popt2_ideal[0] = ideal_waist2 / 2
        popt2_ideal[1] = 1.0
        # popt2_ideal[2] = 40.0

        ax2.plot(hs2, sigsq2, 'o')
        ax2.plot(hplt2, Szsq(hplt2, *popt2), 'r', linewidth=2, label="$M^2={:0.2g}$".format(popt2[1]**2))
        ax2.plot(hplt2, Szsq(hplt2, *popt2_ideal), 'r', linewidth=2, ls='--', alpha=0.6)
        ax2.set_title("Focus at $h = {:0.3g}~\\mu$m, 'Waist' $W_0 = {:0.2g}~\\mu$m"\
                            .format(popt2[-1], 2*popt2[0]), fontsize=14)
        # ax1.set_ylabel("$\\sigma_y^2$ [$\\mu{\\rm m}^2$]")
        ax2.set_ylabel("$\\sigma_{\\rm after}^2$ [$\\mu{\\rm m}^2$]")
        ax2.set_xlabel("Knife-edge height along optical axis [$\\mu$m]")
        ax2.set_ylim(0, maxsig*1.2)
        ax2.legend(loc=0)

    else:
        ax1.set_xlabel("Knife-edge height along optical axis [$\\mu$m]")

    fig.tight_layout()



if gauss_fit:

    if multi_dir:
        f2, axarr2 = plt.subplots(2, sharex=True, sharey=True, 
                                  figsize=(6,6))
        ax1 = axarr2[0]
        ax2 = axarr2[1]
    else:
        f2, axarr2 = plt.subplots(1, sharex=True)
        ax1 = axarr2
    
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
    
    bestprof.y *= (1.0 / popt3[0])
    popt3[0] = 1.0

    if multi_dir:
        popt4, pcov4 = curve_fit(gauss, bestprof2.bins, bestprof2.y, p0=p02)
        fitpts2 = np.arange(np.min(bestprof2.bins), np.max(bestprof2.bins), 0.1)

        bestprof2.y *= (1.0 / popt4[0])
        popt4[0] = 1.0

    
    lab1 = '$w_0 = {:0.2g}~\\mu$m'.format(np.abs(popt3[2])) 

    ax1.plot(bestprof.bins, bestprof.y, 'o')
    ax1.plot(fitpts, gauss(fitpts, *popt3), 'r', linewidth = 2, label=lab1)

    data_int = np.sum(bestprof.y) * (bestprof.bins[1] - bestprof.bins[0])
    gauss_int = np.sum(gauss(fitpts, *popt3)) * (fitpts[1] - fitpts[0])

    print() 
    print("Non-Gaussian Part (1): ", (data_int - gauss_int) / data_int)

    # ax1.set_title('Gaussian Fit Waist = %0.2g $\\mu$m' % (np.abs(popt3[2])) )
    # ax1.set_ylabel("$d \\mathcal{P} / dx$ [arb.]")
    ax1.set_ylabel("$\\left( d \\mathcal{P} / dy \\right)_{\\rm before}$ [arb.]")
    # ax1.set_ylim(10**(-6), popt3[0] * 10)
    ax1.set_ylim(popt3[0] * 1e-4, popt3[0] * 1.5)
    ax1.set_yscale('log')
    ax1.legend(loc=0)
    
    if multi_dir:
        lab2 = '$w_0 = {:0.2g}~\\mu$m'.format(np.abs(popt4[2]))

        ax2.plot(bestprof2.bins, bestprof2.y, 'o')
        ax2.plot(fitpts2, gauss(fitpts2, *popt4), 'r', linewidth = 2, label=lab2)
    
        data_int2 = np.sum(bestprof2.y) * (bestprof2.bins[1] - bestprof2.bins[0])
        gauss_int2 = np.sum(gauss(fitpts2, *popt4)) * (fitpts2[1] - fitpts2[0])
        print("Non-Gaussian Part (2): ", (data_int2 - gauss_int2) / data_int2)

        # ax2.set_title('Gaussian Fit Waist = %0.2g $\\mu$m' % (np.abs(popt4[2])) )
        ax2.set_xlabel("Knife-edge position along beam [$\\mu$m]")

        # ax2.set_ylabel("$d \\mathcal{P} / dy$ [arb.]")
        ax2.set_ylabel("$\\left( d \\mathcal{P} / dy \\right)_{\\rm after}$ [arb.]")
        # ax2.set_ylim(10**(-6), popt4[0] * 10)
        ax2.set_ylim(popt4[0] * 1e-4, popt4[0] * 1.5)
        ax2.set_yscale('log')
        ax2.legend(loc=0)

    else:
        ax1.set_xlabel("Knife-edge position along beam [$\\mu$m]")
    
    f2.tight_layout()



if gauss_fit or msq_fit:

    plt.show()


