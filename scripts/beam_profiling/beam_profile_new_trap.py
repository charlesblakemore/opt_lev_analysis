import numpy as np
import bead_util as bu

import matplotlib.pyplot as plt
import os, re
import scipy.signal as signal

from scipy.special import erf
from scipy.optimize import curve_fit




data_dir = '/data/new_trap/20200320/Bead1/Shaking/'

xfiles, _ = bu.find_all_fnames(data_dir, ext='.h5', substr='_X_', \
                               skip_subdirectories=True)
yfiles, _ = bu.find_all_fnames(data_dir, ext='.h5', substr='_Y_', \
                               skip_subdirectories=True)

nbins = 300

plot_raw_data = False
log_profs = True

gauss_fit = True

use_quad_sum = True






def gauss_wconst(x, A, x0, w0, C):
    return A * np.exp( -2 * (x-x0)**2 / (w0**2) ) + C
    
def gauss(x, A, x0, w0):
    return A * np.exp( -2 * (x-x0)**2 / (w0**2) )






class Profile:
    '''Class storing information from a single file, with methods to add
       data from other files and compute average profiles.'''
    
    def __init__(self, fname, load=True, nbins=300):
        self.fname = fname
        self.mean = "mean not computed"
        self.sigmasq = "std dev not computed"

        self.date = re.search(r"\d{8,}", fname)[0]

        if load:
            profile_data = self.get_profile(self.fname, nbins=nbins)

            self.profile = profile_data['profile']
            self.integral = profile_data['integral']
            self.cant_height = profile_data['height']
            self.prof_dx = np.abs(self.profile[0][1] - self.profile[0][0])
            self.int_dx = np.abs(self.integral[0][1] - self.integral[0][0])


    def get_profile(self, fname, nbins=300, plot_raw_data=False):

        df = bu.DataFile()
        df.load_new(fname)

        df.calibrate_stage_position()

        dt = 1.0 / df.fsamp

        if '_Y_' in fname:
            stage_column = 1
            if 'left' in fname:
                sign = -1.0
            elif 'right' in fname:
                sign = 1.0
            else:
                sign = -1.0
        else:
            stage_column = 0
            sign = 1.0

        if plot_raw_data:
            plt.plot(np.sum(df.amp[:4], axis=0))

            plt.figure()
            for j in range(3):
                plt.plot(df.cant_data[j,:], label=str(j))
            plt.legend()

            plt.show()

        h = np.mean(df.cant_data[2, :])
        h_round = bu.round_sig(h, sig=2)

        if h_round < 10.0:
            h_round = bu.round_sig(h_round, sig=1)

        if use_quad_sum:
            sig = np.sum(df.amp[:4], axis=0)
        else:
            sig_hf = df.other_data
            sig_ds = signal.resample(sig_hf, len(df.cant_data[stage_column]), window=None)
            sig = -1.0 * sig_ds + np.max(sig_ds)

            # plt.plot(sig)
            # plt.show()

            # input()

        proft = np.gradient(sig)

        dir_sign = np.sign(np.gradient(df.cant_data[stage_column])) * sign

        xvec = df.cant_data[stage_column, :]
        yvec = (proft - proft * dir_sign) * 0.5 - (proft + proft * dir_sign) * 0.5

        b_int, y_int, e_int = bu.spatial_bin(xvec, sig, dt, nbins=nbins,\
                                             nharmonics=300, \
                                             add_mean=True, plot=False)

        b, y, e = bu.spatial_bin(xvec, yvec, dt, nbins=nbins, nharmonics=300, \
                                 add_mean=True, plot=False)

        self.profile = [b, y, e]
        self.integral = [b_int, y_int, e_int]
        self.cant_height = h_round
        self.prof_dx = np.abs(self.profile[0][1] - self.profile[0][0])
        self.int_dx = np.abs(self.integral[0][1] - self.integral[0][0])

        result = {}
        result['profile'] = self.profile
        result['integral'] = self.integral
        result['height'] = self.cant_height

        return result



    def add_profile(self, profile_obj):

        new_profx = np.append(self.profile[0], profile_obj.profile[0])
        new_profy = np.append(self.profile[1], profile_obj.profile[1])
        new_profe = np.append(self.profile[2], profile_obj.profile[2])

        new_intx = np.append(self.integral[0], profile_obj.integral[0])
        new_inty = np.append(self.integral[1], profile_obj.integral[1])
        new_inte = np.append(self.integral[2], profile_obj.integral[2])

        prof_sort = np.argsort(new_profx)
        int_sort = np.argsort(new_intx)

        self.profile = [new_profx[prof_sort], \
                        new_profy[prof_sort], \
                        new_profe[prof_sort] ]

        self.integral = [new_intx[int_sort], \
                         new_inty[int_sort], \
                         new_inte[int_sort] ]



    def rebin_profile(self, nbins=300, plot=False):

        x, y, e = bu.rebin(self.profile[0], self.profile[1], \
                           errs=self.profile[2], nbins=nbins, \
                           plot=plot)
        self.profile = [x, y, e]
        self.prof_dx = np.abs(x[1] - x[0])

        x2, y2, e2 = bu.rebin(self.integral[0], self.integral[1], \
                              errs=self.integral[2], nbins=nbins, \
                              plot=plot)
        self.integral = [x2, y2, e2]
        self.int_dx = np.abs(x2[1] - x2[0])


        
    def dist_mean(self):
        #Finds the cnetroid of intensity distribution. subtracts centroid from bins
        norm = np.sum(self.profile[1]*self.prof_dx)
        self.mean = np.sum(self.prof_dx * self.profile[1] * self.profile[0]) / norm
        # self.bins -= self.mean



    def sigsq(self, ROI=(-1000.0, 1000.0)):
        #finds second moment of intensity distribution.
        if type(self.mean) == str:
            self.dist_mean()
        derp1 = self.profile[0] > ROI[0]
        derp2 = self.profile[0] < ROI[1]
        ROIbool = np.array([a and b for a, b in zip(derp1, derp2)])
        norm = np.sum(self.profile[1][ROIbool] * self.prof_dx)
        #norm = np.sum(self.y*self.dxs)
        self.sigmasq = np.sum(self.profile[0][ROIbool]**2 \
                                * self.profile[1][ROIbool]) / norm
        #self.sigmasq = np.sum(self.bins**2*self.y)/norm
         


    def fit_integral(self, plot=False):

        xvec = self.integral[0]
        yvec = self.integral[1]
        errs = self.integral[2]

        if yvec[0] > yvec[-1]:
            def fit_func(x, a, b, c, d):
                # return a * erf( b * (x - c) ) + d
                return a * (1.0 - erf( b * (x - c) )) + d
        else:
            def fit_func(x, a, b, c, d):
                # return a * erf( b * (x - c) ) + d
                return a * erf( b * (x - c) ) + d

        a_guess = 0.5 * np.max(yvec)
        b_guess = 0.005 * np.abs(xvec[-1] - xvec[0])
        c_guess = xvec[np.argmin(np.abs(yvec - 0.5*a_guess))]
        d_guess = np.min(yvec)
        p0 = [a_guess, b_guess, c_guess, d_guess]

        popt, pcov = curve_fit(fit_func, xvec, yvec, \
                               sigma=errs, p0=p0, maxfev=10000)

        if plot:
            plt.errorbar(xvec, yvec, yerr=errs, ls='', marker='o', \
                         ms=6, zorder=1, label='data')
            plt.plot(xvec, fit_func(xvec, *p0), ls='--', lw=3, \
                     color='k', zorder=2, label='initial')
            plt.plot(xvec, fit_func(xvec, *popt), ls='--', lw=3, \
                     color='r', zorder=3, label='fit result')
            plt.yscale('log')
            plt.legend(fontsize=10)
            plt.title("Fit of ERF to Measured Intensity")

            plt.tight_layout()
            plt.show()

        print('Fit result: ')
        print(popt)

        self.integral_fit = [fit_func, popt, pcov]



def proc_dir(files, nbins=300, plot_raw_data=False):

    avg_profs = []
    hs = []
    for fi in files:
        prof = Profile(fi)
        h = prof.cant_height
        if h not in hs:
            ### if new height then create new profile object
            hs.append(h)
            avg_profs.append(prof)
        else:
            ### if height repeated then append data to object for 
            ### that height
            for fp in avg_profs:
                if fp.cant_height == h:
                    fp.add_profile(prof)
            
    #now rebin all profiles
    for fp in avg_profs:
        if len(fp.profile[0]) <= nbins:
            continue
        fp.rebin_profile(nbins=nbins, plot=False)
        fp.fit_integral(plot=True)

    sigmasqs = []
    hs = []

    for f in avg_profs:
        f.sigsq()
        sigmasqs.append(f.sigmasq)
        hs.append(f.cant_height)
        
    return avg_profs, np.array(hs), np.array(sigmasqs)





 
def plot_profs(fp_arr):
    #plots average profile from different heights
    i = 1
    colors = bu.get_color_map(len(fp_arr), cmap='plasma')

    fp_arr_sort = sorted(fp_arr, key = lambda fp: fp.cant_height)

    for fp_ind, fp in enumerate(fp_arr_sort):
        color = colors[fp_ind]
        #plt.errorbar(fp.bins, fp.y, fp.errors, label = str(np.round(fp.cant_height)) + 'um')
        # if multi_dir:
        #     lab = 'dir' + str(i)
        # else:
        lab = str(np.round(fp.cant_height)) + 'um'
        i += 1
        # if multi_dir:
        #     plt.plot(fp.bins, fp.y / np.max(fp.y), 'o', label = lab, color=color)
        #     plt.ylim(10**(-5), 10)
        # else:
        plt.plot(fp.profile[0], fp.profile[1], 'o', label=lab, color=color)
    plt.xlabel("Position [um]")
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
    





x_profs, x_hs, x_sigmasqs = proc_dir(xfiles, plot_raw_data=plot_raw_data)
y_profs, y_hs, y_sigmasqs = proc_dir(yfiles, plot_raw_data=plot_raw_data)

plot_profs(x_profs)
plot_profs(y_profs)







