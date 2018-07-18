import sys, time, itertools, copy

import dill as pickle

import numpy as np
import pandas as pd
import scipy

from scipy.optimize import curve_fit

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opti
import scipy.linalg as linalg

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config
import pandas as pd

sys.path.append('../microgravity')

import warnings
warnings.filterwarnings("ignore")


### Current constraints

limitdata_path = '/sensitivities/decca1_limits.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/sensitivities/decca2_limits.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'



### Stats

confidence_level = 0.95
chi2dist = stats.chi2(1)
# factor of 0.5 from Wilks's theorem: -2 log (Liklihood) ~ chi^2(1)
con_val = 0.5 * chi2dist.ppf(confidence_level)



# Various fitting functions
# pretty self explanatory so there are few comments
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def line(x, a, b):
    return a * x + b

def plane(x, z, a, b, c):
    return a * x + b * z + c

def const(x, a):
    return a

def flicker(x, a):
    return a * (1. / x)


def gauss(x, A, mu, sigma, scale = 1.):
    "standard gaussian pdf"
    return A/np.sqrt(2.*np.pi*(sigma*scale)**2)*np.exp(-1.*(x*scale-mu*scale)**2/(2*(sigma*scale)**2))




def solve_parabola(yvalue, popt):
    '''Computes the two xvalues corresponding to a given
       yvalue in the equation: y = ax^2 + bx + c.

       INPUTS:   yvalue, solution you're looking for
                 popt, list [a,b,c] paramters of parabola

       OUTPUTS:  soln, tuple with solution (x_small, x_large)
    '''
    a, b, c = popt
    soln1 = (-b + np.sqrt(b**2 - 4.0*a*(c-yvalue))) / (2.0*a)
    soln2 = (-b - np.sqrt(b**2 - 4.0*a*(c-yvalue))) / (2.0*a)
    if soln1 > soln2:
        soln = (soln2, soln1)
    else:
        soln = (soln1, soln2)

    return soln






def get_chi2_vs_param(data, data_err, ignore_ax, template, params):
    chi_sqs = np.zeros(len(params))
    for param_ind, param in enumerate(params):
        chi_sq = 0
        Ndof = 0
        Nax = data.shape[0]
        for ax in range(Nax):
            if ignore_ax[ax]:
                continue
            diff = data[ax] - (param * template[ax])
            chi_sq += np.sum( np.abs(diff)**2 / (data_err[ax]**2) )
            Ndof += len(diff)
        chi_sqs[param_ind] = chi_sq / (Ndof - 1)
    return chi_sqs






def get_chi2_vs_param_complex(data, data_err, ignore_ax, template, params):
    Nax = data.shape[0]
    chi_sqs = np.zeros(len(params))
    for param_ind, param in enumerate(params):
        chi_sq = 0
        Ndof = 0
        for ax in range(Nax):
            if ignore_ax[ax]:
                continue
            re_diff = data[ax].real - (param * template[ax].real)
            im_diff = data[ax].imag - (param * template[ax].imag)
            chi_sq += np.sum( np.abs(re_diff)**2 / (0.5 * (data_err[ax]**2)) ) \
                      + np.sum( np.abs(im_diff)**2 / (0.5 * (data_err[ax]**2)) )
            Ndof += len(re_diff) + len(im_diff)
        chi_sqs[param_ind] = chi_sq / (Ndof - 1)
    return chi_sqs






def fit_parabola_to_chi2(params, chi_sqs, plot=False):
    max_chi = np.max(chi_sqs)
    max_param = np.max(params)
    p0 = [max_chi / max_param**2, 0, 1]

    if plot:
        fig, axarr = plt.subplots(1,1)
        axarr.plot(params, chi_sqs)
        axarr.set_xlabel('$\\alpha$ Parameter')
        axarr.set_xlabel('Reduced $\chi^2$ Statistic')
        plt.show()

    try:
        popt, pcov = opti.curve_fit(parabola, params, chi_sqs, p0=p0, maxfev=100000)
    except:
        print "Couldn't fit"
        popt = [0,0,0]
        popt[2] = np.mean(chi_sqs)

    ### Get all the important information from the fit
    best_fit_param = -0.5 * popt[1] / popt[0]
    min_chi = parabola(best_fit_param, *popt)
    chi95 = min_chi + con_val
    soln = solve_parabola(chi95, popt)
    param95 = np.abs(soln[np.argmax(np.abs(soln))]) # select the larger (worse) solution
    fit_err = param95 - best_fit_param

    outdic = {'best_fit_param': best_fit_param, 'min_chi': min_chi, \
              'param95': param95, 'chi95': chi95}

    return outdic





def build_mod_grav_funcs(theory_data_dir):
    '''Loads data from the output of /data/grav_sim_data/process_data.py
       which processes the raw simulation output from the farmshare code

       INPUTS: theory_data_dir, path to the directory containing the data

       OUTPUTS: gfuncs, 3 element list with 3D interpolating functions
                        for regular gravity [fx, fy, fz]
                yukfuncs, 3 x Nlambda array with 3D interpolating function
                          for modified gravity with indexing: 
                          [[y0_fx, y1_fx, ...], [y0_fy, ...], [y0_fz, ...]]
                lambdas, np.array with all lambdas from the simulation
    '''

    ### Load modified gravity curves from simulation output
    Gdata = np.load(theory_data_dir + 'Gravdata.npy')
    yukdata = np.load(theory_data_dir + 'yukdata.npy')
    lambdas = np.load(theory_data_dir + 'lambdas.npy')
    xpos = np.load(theory_data_dir + 'xpos.npy')
    ypos = np.load(theory_data_dir + 'ypos.npy')
    zpos = np.load(theory_data_dir + 'zpos.npy')
    
    if lambdas[-1] > lambdas[0]:
        lambdas = lambdas[::-1]
        yukdata = np.flip(yukdata, 0)

    ### Find limits to avoid out of range erros in interpolation
    xlim = (np.min(xpos), np.max(xpos))
    ylim = (np.min(ypos), np.max(ypos))
    zlim = (np.min(zpos), np.max(zpos))

    ### Build interpolating functions for regular gravity
    gfuncs = [0,0,0]
    for resp in [0,1,2]:
        gfuncs[resp] = interp.RegularGridInterpolator((xpos, ypos, zpos), Gdata[:,:,:,resp])

    ### Build interpolating functions for yukawa-modified gravity
    yukfuncs = [[],[],[]]
    for resp in [0,1,2]:
        for lambind, yuklambda in enumerate(lambdas):
            lamb_func = interp.RegularGridInterpolator((xpos, ypos, zpos), yukdata[lambind,:,:,:,resp])
            yukfuncs[resp].append(lamb_func)
    lims = [xlim, ylim, zlim]

    outdic = {'gfuncs': gfuncs, 'yukfuncs': yukfuncs, 'lambdas': lambdas, 'lims': lims}
    return outdic

    #return gfuncs, yukfuncs, lambdas, lims



def plot_histogram_fit(data):
    '''plots histogram of data and shows fit to gaussian'''
    bins, xo = np.histogram(data)
    fbool = bins>30.
    bin_cents = (xo[1:] + xo[:-1])/2.
    p0 =[np.std(data)*max(bins)*np.sqrt(2.*np.pi), np.mean(data), np.std(data)]
    popt, pcov = curve_fit(gauss, bin_cents[fbool], bins[fbool], sigma = np.sqrt(bins[fbool]), p0 = p0)
    plt.hist(data)
    lab = "$\mu$ = " + str(popt[1]) + " +/- " + str(np.sqrt(pcov[1][1]))
    plt.plot(bin_cents, gauss(bin_cents, *popt),'r' , label = lab, linewidth = 2)
    plt.legend()
    plt.show()

    return

def extend_complex(arr):
    '''extends an array of complex numbers with shape (..., n) to a real array 
       with shape (..., 2n) by concatenating the real and imaginary parts along 
       the last axis.'''
    return np.concatenate((np.real(arr), np.imag(arr)), axis = -1)


def fit_templates(templates, data, weights, method = 'BFGS', x0 = 0):
    '''wrapper for scipy.optimize.minimize to fit a sum of templates to the data. returns the maximum liklihood template coefficients and the covariance matrix determined 
    from the inverse Hessian matrix returned by the fitting subroutine'''
    def NLL(arr):
        return (1./2.)*np.sum((np.einsum('i, ij->j', arr, templates) - data)**2/weights**2)
    x0 = x0*np.ones(len(templates))
    res = scipy.optimize.minimize(NLL, x0, method = method)
    return res.x, res.hess_inv, res.success  


def norm(dirbynarr):
    '''computes the rms norm alond the last axis of dirbynarr'''
    return np.sqrt(np.sum(np.abs(dirbynarr)**2, axis = -1))



class FileData:
    '''A class to store data from a single file, only
       what is relevant for higher level analysis.'''
    

    def __init__(self, fname, tfdate='', tophatf=2500, plot_tf=False):
        '''Load an hdf5 file into a bead_util.DataFile obj. Calibrate the stage position.
           Calibrate the microsphere response with th transfer function.'''

        df = bu.DataFile()
        try:
            df.load(fname)
            self.badfile = False
        except:
            self.badfile = True
            return
        
        df.calibrate_stage_position()
        df.diagonalize(date=tfdate, maxfreq=tophatf, plot=plot_tf)

        self.time = df.time
        self.fsamp = df.fsamp
        self.nsamp = df.nsamp
        self.phi_cm = df.phi_cm
        self.df = df

        self.data_closed = False
        

    def rebuild_drive(self, mean_pos=40.0):

        ## Transform ginds array to array of indices for drive freq
        temp_inds = np.array(range(int(self.nsamp*0.5) + 1))
        inds = temp_inds[self.ginds]

        freqs = np.fft.rfftfreq(self.nsamp, d=1.0/self.fsamp)

        full_drive_fft = [[], [], []]
        for resp in [0,1,2]:
            full_drive_fft[resp] = np.zeros(len(freqs), dtype=np.complex128)
            for ind, freq_ind in enumerate(inds):
                full_drive_fft[resp][freq_ind] = self.driveffts[resp][ind]
        drivevec = np.fft.irfft(full_drive_fft)[self.drive_ind]

        return drivevec + mean_pos



    def extract_data(self, ext_cant=(False,1), nharmonics=10, harms=[], width=0, \
                     npos=500, noiseband=5, plot_harm_extraction=False):
        '''Extracts the microsphere resposne at the drive frequency of the cantilever, and
           however many harmonics are requested. Uses a notch filter that is default set 
           one FFT bin width, although this can be increased'''

        if self.data_closed:
            return

        ## Get the notch filter
        cantfilt = self.df.get_boolean_cantfilt(ext_cant=ext_cant, nharmonics=nharmonics, harms=harms, width=width)
        self.ginds = cantfilt['ginds']
        self.fund_ind = cantfilt['fund_ind']
        self.drive_freq = cantfilt['drive_freq']
        self.drive_ind = cantfilt['drive_ind']


        ## Apply the notch filter
        fftdat = self.df.get_datffts_and_errs(self.ginds, self.drive_freq, noiseband=noiseband, \
                                              plot=plot_harm_extraction)
        self.datffts = fftdat['datffts']
        self.daterrs = fftdat['daterrs']
        self.diagdatffts = fftdat['diagdatffts']
        self.diagdaterrs = fftdat['diagdaterrs']

        self.driveffts = fftdat['driveffts']

        ### Get the binned data and calibrate the non-diagonalized part
        self.df.get_force_v_pos()
        binned = np.array(self.df.binned_data)
        for resp in [0,1,2]:
            binned[resp][1] = binned[resp][1] * self.df.conv_facs[resp]
        self.binned = binned


        ### Analyze the attractor drive and build the relevant position vectors
        ### for the bead
        mindrive = np.min(self.df.cant_data[self.drive_ind])
        maxdrive = np.max(self.df.cant_data[self.drive_ind])
        self.posvec = np.linspace(mindrive, maxdrive, npos)



    def load_position_and_bias(self, ax1='x', ax2='z'):

        if not self.data_closed:
            ax_keys = {'x': 0, 'y': 1, 'z': 2}

            self.cantbias = self.df.electrode_settings['dc_settings'][0]

            ax1pos = np.mean(self.df.cant_data[ax_keys[ax1]])
            self.ax1pos = round(ax1pos, 1)

            ax2pos = np.mean(self.df.cant_data[ax_keys[ax2]])
            self.ax2pos = round(ax2pos, 1)



    def generate_pts(self, p0, attractor_travel = 80.):
        '''generates pts arry for determining gravitational template.'''
        xpos = p0[0] + (attractor_travel - self.ax1pos)
        height = self.ax2pos - p0[2]
            
        ones = np.ones_like(self.posvec)
        pts = np.stack((xpos*ones, self.posvec, height*ones), axis=-1)

        drivevec = self.rebuild_drive()

        full_ones = np.ones_like(drivevec)
        full_pts = np.stack((xpos*full_ones, drivevec, height*full_ones), axis=-1)

        return full_pts
    
    def fit_alpha_xyz(self, tempsxyz, diag = False, \
                      columns = ["fit coefs", "sigmas", "fit success"]):
        '''fits x, y, and z force data sepretly. Returns a pandas Series object containing
           the results of the fit.'''
        if diag:
            dat = self.diagdatffts
            sigmas = self.diagdaterrs
        else:
            dat = self.datffts
            sigmas = self.daterrs
        
        #first normalize data and template so fitter does not shit itself.
        ndat = norm(dat)
        ns_temps = np.array(map(norm, tempsxyz))
        datned = np.einsum("i, ij->ij", 1./ndat, dat)
        sigmasned = np.einsum("i, ij->ij", 1./ndat, sigmas)
        tempsn = np.einsum("ij, ijk->ijk", 1./ns_temps, tempsxyz)
        #extend imaginary templates and data to a real vector twice the length 
        datnt = np.concatenate((np.real(datned), np.imag(datned)), axis = -1)
        tempst = np.concatenate((np.real(tempsn), np.imag(tempsn)), axis = -1)
        #error may laread have factor of sqrt(2). Need to check
        sigmast = np.concatenate((sigmasned/np.sqrt(2), sigmasned/np.sqrt(2)), axis = -1)
        #loop over directions and compute independent alpha for each direction
        df = pd.DataFrame()
        for i in range(len(dat)):
            x, hess_inv, success = \
                    fit_templates(tempst[:, i, :], datnt[i, :], sigmast[i, :])
            sigs_fit = np.sqrt(np.diag(hess_inv))
            #undo normalization
            x *= ndat[i]
            x = np.einsum("i, i", 1./ns_temps[:, i], x)
            sigs_fit *= ndat[i]
            sigs_fit = np.einsum("i, i", 1./ns_temps[:, i], sigs_fit)
            dfi = pd.DataFrame([[x, sigs_fit, success]], columns = [c +" " + str(i) for c in columns])
            df = pd.concat([df, dfi], axis = 1, sort = False)
        
        return df





    def close_datafile(self):
        '''Clear the old DataFile class by assigning and empty class to self.df
           and assuming the python garbage collector will take care of it.'''
        self.data_closed = True
        self.df = bu.DataFile()











class AggregateData:
    '''A class to store data from many files. Stores a FileDat object for each and
       has some methods that work on each object in a loop.'''

    
    def __init__(self, fnames, p0_bead=[16,0,20], tophatf=2500):
        
        self.fnames = fnames
        self.p0_bead = p0_bead
        self.file_data_objs = []

        Nnames = len(self.fnames)

        suff = 'Processing %i files' % Nnames
        for name_ind, name in enumerate(self.fnames):
            bu.progress_bar(name_ind, Nnames, suffix=suff)

            # Initialize FileData obj, extract the data, then close the big file
            try:
                new_obj = FileData(name, tophatf=tophatf)
                new_obj.extract_data()#harms=harms)
                new_obj.load_position_and_bias()

                new_obj.close_datafile()
            
                self.file_data_objs.append(new_obj)

            except:
                continue

        self.grav_loaded = False

        self.alpha_dict = ''
        self.agg_dict = ''


    def save(self, savepath):
        parts = savepath.split('.')
        if len(parts) > 2:
            print "Bad file name... too many periods/extensions"
            return
        else:
            savepath = parts[0] + '.agg'
            self.clear_grav_funcs()
            pickle.dump(self, open(savepath, 'wb'))
            self.reload_grav_funcs()

            
    def load_grav_funcs(self, theory_data_dir, verbose=True):
        self.theory_data_dir = theory_data_dir
        if verbose:
            print "Loading Gravity Data...",
        grav_dict = build_mod_grav_funcs(theory_data_dir)
        self.gfuncs = grav_dict['gfuncs']
        self.yukfuncs = grav_dict['yukfuncs']
        self.lambdas = grav_dict['lambdas']
        self.lims = grav_dict['lims']
        self.grav_loaded = True
        if verbose:
            print "Done!"


    def reload_grav_funcs(self):
        try:
            self.load_grav_funcs(self.theory_data_dir,verbose=False)
        except:
            print 'No theory_data_dir saved'


    def clear_grav_funcs(self):
        self.gfuncs = ''
        self.yukfuncs = ''
        self.lambdas = ''
        self.lims = ''
        self.grav_loaded = False


    def bin_rough_stage_positions(self, ax_disc=0.5):
        '''Loops over the preprocessed file_data_objs and organizes them by rough stage position,
           discretizing the rough stage position by a user-controlled parameter. Unfortunately,
           because the final object is a nested dictionary, it's somewhat cumbersome to put this
           into any subroutines.
        '''
        
        print 'Sorting data by rough stage position...',

        agg_dict = {}

        biasvec = []

        ax1vec = []
        Nax1 = {}
        ax2vec = []
        Nax2 = {}

        for file_data_obj in self.file_data_objs:
            bias = file_data_obj.cantbias
            ax1pos = file_data_obj.ax1pos
            ax2pos = file_data_obj.ax2pos

            if bias not in agg_dict.keys():
                agg_dict[bias] = {}
                biasvec.append(bias)
                Nax1[bias] = {}
                Nax2[bias] = {}
            

            #### Check for the first axis (usually X)
            ax1_is_new = False
            if len(ax1vec):
                ## Finds the closest position in the array (if the array already is populated)
                ## and then decides if the current position is "new", or should be averaged
                ## together with other positions
                close_ind = np.argmin( np.abs( np.array(ax1vec) - ax1pos ) )
                if np.abs(ax1vec[close_ind] - ax1pos) < ax_disc:
                    old_ax1key = ax1vec[close_ind]
                    oldN = Nax1[bias][old_ax1key]

                    ## Adjust the average position to include this new key
                    new_ax1key = (old_ax1key * oldN + ax1pos) / (oldN + 1.0)
                    new_ax1key = round(new_ax1key, 1)

                    ## If the discretized/rounded version of the average new key doesn't 
                    ## equal the average old key, then collect all of the data for both
                    ## keys under the new key
                    if old_ax1key != new_ax1key:
                        ax1vec[close_ind] = new_ax1key

                        agg_dict[bias][new_ax1key] = agg_dict[bias][old_ax1key]
                        Nax1[bias][new_ax1key] = oldN + 1.0

                        del Nax1[bias][old_ax1key]
                        del agg_dict[bias][old_ax1key]
                    #else:
                    #    Nax1[bias][old_ax1key] += 1
                    
                else:
                    ax1_is_new = True
            else:
                ax1_is_new = True

            ## If the new position is truly "new", added it to the rough stage position
            ## vectors and makes a new dictionary entry
            if ax1_is_new:
                agg_dict[bias][ax1pos] = {}
                Nax1[bias][ax1pos] = 1
                new_ax1key = ax1pos
                ax1vec.append(new_ax1key)
                ax1vec.sort()



            #### Check for the second axis (usually Z)
            ax2_is_new = False
            if len(ax2vec):
                ## Finds the closest position in the array (if the array already is populated)
                ## and then decides if the current position is "new", or should be averaged
                ## together with other positions
                close_ind = np.argmin( np.abs( np.array(ax2vec) - ax2pos ) )
                if np.abs(ax2vec[close_ind] - ax2pos) < ax_disc:
                    old_ax2key = ax2vec[close_ind]
                    oldN = Nax2[bias][old_ax2key]

                    ## Adjust the average position to include this new key
                    new_ax2key = (old_ax2key * oldN + ax2pos) / (oldN + 1.0)
                    new_ax2key = round(new_ax2key, 1)
               
                    if old_ax2key not in agg_dict[bias][new_ax1key].keys():
                        agg_dict[bias][new_ax1key][new_ax2key] = []

                    ## If the discretized/rounded version of the average new key doesn't 
                    ## equal the average old key, then collect all of the data for both
                    ## keys under the new key
                    if old_ax2key != new_ax2key:
                        ax2vec[close_ind] = new_ax2key

                        Nax2[bias][new_ax2key] = oldN + 1.0
                        del Nax2[bias][old_ax2key]

                        for ax1key in ax1vec:
                            ax2keys = agg_dict[bias][ax1key].keys()
                            if old_ax2key in ax2keys:
                                agg_dict[bias][ax1key][new_ax2key] = agg_dict[bias][ax1key][old_ax2key]
                                del agg_dict[bias][ax1key][old_ax2key]
                    #else:
                    #    Nax2[bias][old_ax2key] += 1
                    
                else:
                    ax2_is_new = True
            else:
                ax2_is_new = True

            ## If the new position is truly "new", addes it to the rough stage position
            ## vector and makes a new dictionary entry
            if ax2_is_new:
                agg_dict[bias][new_ax1key][ax2pos] = []
                Nax2[bias][ax2pos] = 1
                new_ax2key = ax2pos
                ax2vec.append(new_ax2key)
                ax2vec.sort()

            ## Add in the new data to our aggregate dictionary
            agg_dict[bias][new_ax1key][new_ax2key].append( file_data_obj )

        print 'Done!'

        ax1vec.sort()
        ax2vec.sort()
        biasvec.sort()

        self.biasvec = biasvec
        self.ax1vec = ax1vec
        self.ax2vec = ax2vec
        self.agg_dict = agg_dict






    def find_alpha_vs_time(self, br_temps = [], single_lambda = True, lambda_value = 25E-6):
        
        print "Computing alpha as a function of time..."

        if not self.grav_loaded:
            print "Must load theory data first..."
            return
        
        Nobj = len(self.file_data_objs)

        dft = pd.DataFrame()
        for objind, file_data_obj in enumerate(self.file_data_objs):
            bu.progress_bar(objind, Nobj, suffix='Fitting Alpha vs. Time')

            t = file_data_obj.time
            phi = file_data_obj.phi_cm

            ## Get sep and height from axis positions

            full_pts  = file_data_obj.generate_pts(self.p0_bead)

            ## Loop over lambdas and
            lambda_inds = np.arange(len(self.lambdas))
            if single_lambda:
                lind  = np.argmin((lambda_value - self.lambdas)**2)
                lambda_inds = [lambda_inds[lind]]
                n_lam = 1
            else:
                n_lam = len(self.lambdas)


            for i, lambind in enumerate(lambda_inds):
                yukfft = [[], [], []]
                for resp in [0,1,2]:
                    yukforcet = self.yukfuncs[resp][lambind](full_pts*1.0e-6)
                    yukfft[resp] = np.fft.rfft(yukforcet)[file_data_obj.ginds]
                yukfft = np.array(yukfft)
            
                dfl = file_data_obj.fit_alpha_xyz([yukfft] + br_temps)
                dfl["lambda"] = self.lambdas[lambind]
                index = [[objind], [lambind]]
                dfl.index = index
                dft = dft.append(dfl)

        return dft
                




    def find_mean_alpha_vs_position(self, ignoreXYZ=(0,0,0)):

        print 'Finding alpha vs. height/separation...'
        
        if not self.grav_loaded:
            print "FAILED: Must load thoery data first!"
            return

        alpha_dict = {}
        for bias in self.agg_dict.keys():
            alpha_dict[bias] = {}
            for ax1key in self.ax1vec:
                alpha_dict[bias][ax1key] = {}
                for ax2key in self.ax2vec:
                    alpha_dict[bias][ax1key][ax2key] = []

        i = 0
        DataFrameTot = pd.DataFrame
        totlen = len(self.agg_dict.keys()) * len(self.ax1vec) * len(self.ax2vec)
        for bias, ax1, ax2 in itertools.product(self.agg_dict.keys(), self.ax1vec, self.ax2vec):
            i += 1
            suff = '%i / %i position combinations' % (i, totlen)
            newline=False
            if i == totlen:
                newline=True

            objs = self.agg_dict[bias][ax1][ax2]

            nfiles = len(objs)
            filfac = 1.0 / float(nfiles)

            xpos = 0.0
            height = 0.0

            ### Initialize average arrays 
            drivevec_avg = np.zeros_like(objs[0].rebuild_drive())
            posvec_avg = np.zeros_like(objs[0].posvec)
            datfft_avg = np.zeros_like(objs[0].datffts)
            daterr_avg = np.zeros_like(objs[0].daterrs)
            binned_avg = np.zeros_like(objs[0].binned)
            old_ginds = []

            #average over integrateions at the same position
            for obj in objs:
                xpos += filfac * (self.p0_bead[0] + (80 - obj.ax1pos))
                height += filfac * (obj.ax2pos - self.p0_bead[2])

                if not len(old_ginds):
                    old_ginds = obj.ginds
                np.testing.assert_array_equal(obj.ginds, old_ginds, err_msg='notch filter changes between files...')
                old_ginds = obj.ginds

                drivevec = obj.rebuild_drive()
                drivevec_avg += filfac * drivevec

                posvec_avg += filfac * obj.posvec
                datfft_avg += filfac * obj.datffts
                daterr_avg += filfac * obj.daterrs
                binned_avg += filfac * obj.binned

            full_ones = np.ones_like(drivevec_avg)
            full_pts = np.stack((xpos*full_ones, drivevec_avg, height*full_ones), axis=-1)

            ones = np.ones_like(posvec_avg)
            pts = np.stack((xpos*ones, posvec_avg, height*ones), axis=-1)

            ## Include normal gravity in fit. But why???
            gfft = [[], [], []]
            for resp in [0,1,2]:
                if ignoreXYZ[resp]:
                    gfft[resp] = np.zeros(np.sum(old_ginds))
                    continue
                gforcet = self.gfuncs[resp](full_pts*1.0e-6)
                gfft[resp] = np.fft.rfft(gforcet)[old_ginds]
            gfft = np.array(gfft)

            best_fit_alphas = np.zeros(len(self.lambdas))
            best_fit_errs = np.zeros(len(self.lambdas))

            ## Loop over lambdas and 
            
            for lambind, yuklambda in enumerate(self.lambdas):
                bu.progress_bar(lambind, len(self.lambdas), suffix=suff, newline=newline)
                

                yukfft = [[], [], []]
                start_yuk2 = time.time()
                for resp in [0,1,2]:
                    if ignoreXYZ[resp]:
                        yukfft[resp] = np.zeros(np.sum(old_ginds))
                        continue
                    yukforce = self.yukfuncs[resp][lambind](pts*1.0e-6)
                    yukforce_func = interp.interp1d(posvec_avg, yukforce)

                    yukforcet = yukforce_func(drivevec_avg)
                    yukfft[resp] = np.fft.rfft(yukforcet)[old_ginds]
                yukfft2 = np.array(yukfft)
            
                newalpha = 2.0 * np.mean( np.abs(datfft_avg[0]) ) / np.mean( np.abs(yukfft[0]) ) * 1.5*10**(-1)
                testalphas = np.linspace(-1.0*newalpha, newalpha, 51)

                chi_sqs = get_chi2_vs_param_complex(datfft_avg, daterr_avg, ignoreXYZ, yukfft, testalphas)
                fit_result = fit_parabola_to_chi2(testalphas, chi_sqs)

                best_fit_alphas[lambind] = fit_result['best_fit_param']
                best_fit_errs[lambind] = fit_result['param95']

            #Create DataFrame to store the output of each fit 
            DataFrame_alphas = pd.DataFrame.from_records(\
                    [[best_fit_alphas, best_fit_errs]], columns = ["total_alpha", "total_alpha_error"], index = self.lambdas)
            DataFramei = pd.DataFrame.from_records(\
                    [[bias, ax1, ax2, DataFrame_alphas]], index = [i], columns = ["bias", "ax1", "ax2", "alphas"])
       
                
            if i != 1:
                DataFrameTot = pd.concat([DataFrameTot, DataFramei])
            elif i== 1:
                DataFrameTot = DataFramei
            else:
                print "wtf is going on"

            alpha_dict[bias][ax1][ax2] = [best_fit_alphas, best_fit_errs]

        print 'Done!'
        self.DataFrame = DataFrameTot    
        self.alpha_dict = alpha_dict


   # def alpha_dict_to_DataFrame(self, alpha_dict):
    #    '''generates a pandas data frame from the alpha_dict for further processing. 
     #      For the time being, assums that the data is rectangular'''

      #  vs = alpha_dict.keys()
      #  xs = alpha_dict[vs[0]].keys()
      #  zs = alpha_dict[xs[0]].keys()
      #  lams = self.lambdas



    def fit_alpha_vs_alldim(self, weight_planar=True):

        if not self.grav_loaded:
            try:
                self.reload_grav_funcs()
            except:
                print 'No grav funcs... Tried to reload but no filename'

        self.alpha_best_fit = []
        self.alpha_95cl = []

        ### Assume separations are encoded in ax1 and heights in ax2
        seps = 80 + self.p0_bead[0] - np.array(self.ax1vec)
        heights = self.p0_bead[2] - np.array(self.ax2vec) 
        
        ### Sort the heights and separations and build a grid
        sort1 = np.argsort(seps)
        sort2 = np.argsort(heights)
        seps_sort = seps[sort1]
        heights_sort = heights[sort2]
        heights_g, seps_g = np.meshgrid(heights_sort, seps_sort)

        for bias in self.biasvec:
            
            for lambind, yuklambda in enumerate(self.lambdas):

                dat = [[[] for i in range(len(heights))] for j in range(len(seps))]
                errs = [[[] for i in range(len(heights))] for j in range(len(seps))]

                for ax1ind, ax1pos in enumerate(self.ax1vec):

                    ### Loop over all files at each separation and collect
                    ### the value of alpha for the current value of yuklambda
                    for ax2ind, ax2pos in enumerate(self.ax2vec):

                        tempdat = self.alpha_dict[bias][ax1pos][ax2pos]

                        dat[ax1ind][ax2ind] = tempdat[0][lambind]
                        errs[ax1ind][ax2ind] = tempdat[1][lambind]

                dat = np.array(dat)
                errs = np.array(errs)

                ### Since the data dictionary was indexed by cantilever settings 
                ### rather than actual bead positions, we have to sort the data
                ### to match the sorted separations and heights
                dat = dat[sort1,:]
                dat = dat[:,sort2]

                errs = errs[sort1,:]
                errs = errs[:,sort2]

                scale_fac = np.mean(dat)

                dat_sc = dat * (1.0 / scale_fac)
                errs_sc = errs * (1.0 / scale_fac)
                if not weight_planar:
                    errs_sc = np.ones_like(dat_sc)

                ### Defined a funciton to minize via least-squared optimization
                def func(params, fdat=dat_sc, ferrs=errs_sc):
                    funcval = params[0] * heights_g + params[1] * seps_g + params[2]
                    return ((funcval - fdat) / ferrs).flatten()

                ### Optimize the previously defined function
                res = opti.leastsq(func, [0.2*np.mean(dat_sc), 0.2*np.mean(dat_sc), 0], \
                                   full_output=1, maxfev=10000)

                try:
                    x = res[0]
                    residue = linalg.inv(res[1])[2,2]
                except:
                    2+2
                
                ### Deplane the data and extract some statistics
                deplaned = dat - scale_fac * (x[0] * heights_g + x[1] * seps_g + x[2])
                deplaned_wconst = deplaned + scale_fac * x[2]

                deplaned_avg = np.mean(deplaned)
                deplaned_std = np.std(deplaned) / dat.size

                self.alpha_best_fit.append(np.abs(x[2]*scale_fac))
                self.alpha_95cl.append(deplaned_std)


    def plot_alpha_dict(self, bias=0.0, yuklambda=25.0e-6, plot_errs=True):

        ### Assume separations are encoded in ax1 and heights in ax2
        seps = 80 + self.p0_bead[0] - np.array(self.ax1vec)
        heights = self.p0_bead[2] - np.array(self.ax2vec) 
        
        ### Sort the heights and separations and build a grid
        sort1 = np.argsort(seps)
        sort2 = np.argsort(heights)
        seps_sort = seps[sort1]
        heights_sort = heights[sort2]
        heights_g, seps_g = np.meshgrid(heights_sort, seps_sort)

        cbias = self.biasvec[np.argmin(np.abs(bias-np.array(self.biasvec)))]
        yukind = np.argmin(np.abs(yuklambda-np.array(self.lambdas)))

        alpha_g = np.zeros((len(self.ax1vec), len(self.ax2vec)))
        err_g = np.zeros((len(self.ax1vec), len(self.ax2vec)))
        for (nax1, ax1), (nax2, ax2) in itertools.product(enumerate(self.ax1vec), enumerate(self.ax2vec)):
            alpha_g[nax1, nax2] = self.alpha_dict[cbias][ax1][ax2][0][yukind]
            err_g[nax1, nax2] = self.alpha_dict[cbias][ax1][ax2][1][yukind]

        alpha_g = alpha_g[sort1,:]
        alpha_g = alpha_g[:,sort2]

        err_g = err_g[sort1,:]
        err_g = err_g[:,sort2]

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(heights_g, seps_g, alpha_g, label='Best-fit alphas')
        ax.scatter(heights_g, seps_g, err_g, label='Errors for weighting in planar fit')
        ax.legend()
        ax.set_xlabel('Z-position [um]')
        ax.set_ylabel('X-separation [um]')
        ax.set_zlabel('Alpha [arb]')
        plt.show()

    



    def plot_sensitivity(self, plot_just_current=False):

        fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)
        
        if not plot_just_current:
            ax.loglog(self.lambdas, self.alpha_best_fit, linewidth=2, label='Size of Apparent Background')
            ax.loglog(self.lambdas, self.alpha_95cl, linewidth=2, label='95% CL at Noise Limit')

        ax.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
        ax.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
        ax.grid()

        #ax.set_xlim(lambda_plot_lims[0], lambda_plot_lims[1])
        #ax.set_ylim(alpha_plot_lims[0], alpha_plot_lims[1])

        ax.set_xlabel('$\lambda$ [m]')
        ax.set_ylabel('$\\alpha$')

        ax.legend(numpoints=1, fontsize=9)

        #ax.set_title(figtitle)
        plt.tight_layout()

        plt.show()



