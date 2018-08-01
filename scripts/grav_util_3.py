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


ax_dict = {0: 'X', 1: 'Y', 2: 'Z'}

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


def gauss(x, A, mu, sigma):
    "standard gaussian pdf"
    prefac = A / np.sqrt( 2.0 * np.pi * sigma**2 )
    exp = -1.0 * (x - mu)**2 / (2 * sigma**2)
    return prefac * np.exp(exp)



def null(A, eps=1.0e-15):
    '''Given a matrix A (which can be incredibly sparse), computes
       the set of vectors that span the null space of the matrix A

       INPUTS:   A,   Input matrix, assumed NxN dimensional. Can be a list of
                      lists, or a 2-dimensional ndarray
                 eps, error (relative to a vector with compoenents of 
                      order unity) accepted for an inner product to be
                      to be considered 0.

       OUTPUTS:  null_space, MxN dimensional ndarry with M basis vectors that 
                             span the null of A
    '''
    
    B = np.array(A)
    if np.sum(B) == 0:
        return B

    C = np.zeros_like(B)

    u, s, vh = scipy.linalg.svd(B)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return null_space




def make_basis_from_template_vec(template_vec):
    '''Uses SVD decomposition to make a set of N, linearly
       independent basis-vectors from a single N-dimensional
       input vector.

       INPUTS:   template_vec, iterable object with N-components
                               assumed complex-valued. 

       OUTPUTS (dictionary with entries keyed by strings):
            key: 'real_basis', ndarray with rows given by spanning
                               basis vectors for the real components
            key: 'imag_basis', ndarray with rows given by spanning
                               basis vectors for the imaginary comp.
    '''
    template_vec = np.array(template_vec)

    ### Extract real and imaginary to consider them independently
    reals = template_vec.real
    imags = template_vec.imag

    real_norm = np.linalg.norm(reals)
    imag_norm = np.linalg.norm(imags)

    zeros = np.zeros_like(reals)
    
    ### Build up the matrix of basis vectors with our single, known
    ### vector and the rest zeros (large null space)
    real_rows = [reals / real_norm]
    if imag_norm != 0: 
        # if template_vec is real, then imag_norm = 0. we don't
        # want any divide-by-zero errors do we
        imag_rows = [imags / imag_norm]
    else:
        imag_rows = [imags]

    ### Add the zero vectors
    for dim in range(len(reals)-1):
        real_rows.append(zeros)
        imag_rows.append(zeros)

    ### Compute a set of linearly independent vectors that span
    ### the null of the given matrices
    real_basis = null(real_rows) * real_norm
    imag_basis = null(imag_rows) * imag_norm
    
    ### Append the real and imaginary components of the given input
    ### vector in order to complete the spanning basis
    real_basis = np.append([reals], real_basis, axis=0)
    imag_basis = np.append([imags], imag_basis, axis=0)

    outdict = {'real_basis': real_basis, 'imag_basis': imag_basis}
    return outdict




def projection(u, v):
    '''Project the real vector v onto the real vector u.'''
    cross_term = np.inner(u, v)
    self_term = np.inner(u, u)

    return (cross_term / self_term) * np.array(u)




def gram_schmidt(basis, normalize=True, plot=False):
    '''Performs the gram_schmidt procedure to transform the NxN 
       dimensional input basis (assumed to span R^N) to an orthogonal
       basis, with an option to normalize

       INPUTS:   basis, NxN dimensional object with the first index
                        indexing the basis vectors
                 normalize, optional argument to normalize the output

       OUTPUTS:  orthogonal_basis, NxN ndarray
    '''
    N = basis.shape[0]
    orthogonal_basis = []
    for i in range(N):

        temp_vec = np.array(basis[i])

        ### Loop over all new basis vectors created so far and subtract 
        ### the projection of the current, old basis vector on those 
        ### new basis vectors to obtain the next orthogonal basis vector
        ###
        ### First orthogonal basis vector is just the first vector of our
        ### original basis (i.e. likely a template vector)

        for new_vec in orthogonal_basis:
            temp_vec -= projection( new_vec, np.array(basis[i]) )
        orthogonal_basis.append(temp_vec)

    ### Normalize if desired
    if normalize:
        orthonormal_basis = []
        for vec in orthogonal_basis:
            orthonormal_basis.append(vec / np.linalg.norm(vec))
    
    ### Plot basis (via imshow) and a matrix of inner products to 
    ### demonstrate orthogonality
    if plot:
        plt.subplot(131)
        plt.imshow(basis)
        plt.title('Original Basis')
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(orthogonal_basis)
        plt.title('Orthogonal Basis')
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(orthogonal_basis - basis)
        plt.title('Diff')
        plt.colorbar()

        if normalize:
            ident_basis = orthonormal_basis
        else:
            ident_basis = orthogonal_basis

        ident = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                ident[i,j] = np.inner(ident_basis[i], ident_basis[j])
        
        plt.figure()
        plt.title('Matrix of Inner Products')
        plt.imshow(ident)
        plt.colorbar()
        plt.show()

    outdict = {'orthogonal': np.array(orthogonal_basis)}
    if normalize:
        outdict['orthonormal'] = np.array(orthonormal_basis)

    return outdict




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
    '''Computes the chi^2 square for a particular combination of
       template, data, and errors. These arrays are assumed to have
       identical indexing and the 'independent variable' is ignored.

       INPUTS:   data,      D by N array of real-valued data points, with D
                            dimensions (e.g. D=3 => X, Y, Z) and N points
                 data_err,  standard deviation of observations in data,
                            with same shape as data
                 ignore_ax, D-length tuple or list of bool specifying whether
                            to ignore an axis when computing chi^2
                 template,  D by N array of real-valued template points
                 params,    M-length array of template amplitudes over 
                            which to compute the chi^2 score

       OUTPUTS (packed in a dictionary):  
                 chi_sqs,      M-length array of associated chi^2 scores
                 red_chi_sqs,  chi_sqs array reduced by: 1 / (Ndof - 1)
                 Ndof,         DOF in associated fit
    '''
    Nax = data.shape[0]
    chi_sqs = np.zeros(len(params))
    red_chi_sqs = np.zeros(len(params))
    old_Ndof = 0
    for param_ind, param in enumerate(params):
        chi_sq = 0
        Ndof = 0
        for ax in range(Nax):
            if ignore_ax[ax]:
                continue
            diff = data[ax] - (param * template[ax])
            chi_sq += np.sum( np.abs(diff)**2 / (data_err[ax]**2) )
            Ndof += len(diff)
        chi_sqs[param_ind] = chi_sq
        red_chi_sqs[param_ind] = chi_sq / (Ndof - 1)

        if old_Ndof != 0:
            assert old_Ndof == Ndof
        old_Ndof = Ndof

    outdic = {'chi_sqs': chi_sqs, 'red_chi_sqs': red_chi_sqs, 'Ndof': Ndof}
    return outdic






def get_chi2_vs_param_complex(data, data_err, ignore_ax, template, params):
    '''Computes the chi^2 square for a particular combination of
       template, data, and errors. These arrays are assumed to have
       identical indexing and the 'independent variable' is ignored.
       This version handles complex-valued data and treats the real 
       and imaginary components as independent DOFs

       INPUTS:   data,      D by N array of complex-valued data points, with D
                            dimensions (e.g. D=3 => X, Y, Z) and N points
                 data_err,  standard deviation of observations in data,
                            with same shape as data
                 ignore_ax, D-length tuple or list of bool specifying whether
                            to ignore an axis when computing chi^2
                 template,  D by N array of complex-valued template points
                 params,    M-length array of template amplitudes over 
                            which to compute the chi^2 score

       OUTPUTS (packed in a dictionary):  
                 chi_sqs,      M-length array of associated chi^2 scores
                 red_chi_sqs,  chi_sqs array reduced by: 1 / (Ndof - 1)
                 Ndof,         DOF in associated fit
    '''
    Nax = data.shape[0]
    chi_sqs = np.zeros(len(params))
    red_chi_sqs = np.zeros(len(params))
    old_Ndof = 0
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
        chi_sqs[param_ind] = chi_sq
        red_chi_sqs[param_ind] = chi_sq / (Ndof - 1)

        if old_Ndof != 0:
            assert old_Ndof == Ndof
        old_Ndof = Ndof

    outdic = {'chi_sqs': chi_sqs, 'red_chi_sqs': red_chi_sqs, 'Ndof': Ndof}
    return outdic






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








class FileData:
    '''A class to store data from a single file, only
       what is relevant for higher level analysis.'''
    

    def __init__(self, fname, tfdate='', tophatf=2500, plot_tf=False):
        '''Load an hdf5 file into a bead_util.DataFile obj. Calibrate the stage position.
           Calibrate the microsphere response with th transfer function.'''

        df = bu.DataFile()
        try:
            df.load(fname)
            self.fname = fname
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
        cantfilt = self.df.get_boolean_cantfilt(ext_cant=ext_cant, nharmonics=nharmonics, \
                                                harms=harms, width=width)
        self.ginds = cantfilt['ginds']
        self.fund_ind = cantfilt['fund_ind']
        self.drive_freq = cantfilt['drive_freq']
        self.drive_ind = cantfilt['drive_ind']


        ## Apply the notch filter
        fftdat = self.df.get_datffts_and_errs(self.ginds, self.drive_freq, noiseband=noiseband, \
                                              plot=plot_harm_extraction)
        self.datfft = fftdat['datffts']
        self.daterr = fftdat['daterrs']
        self.diagdatfft = fftdat['diagdatffts']
        self.diagdaterr = fftdat['diagdaterrs']

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
        else:
            print 'ERROR: load_position_and_bias()'
            print 'Whatchu doin boi?! The bu.DataFile() object is closed.'
            time.sleep(1)
            print '(press enter when you\'re ready to get a real Python error)'
            raw_input()



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




    def close_datafile(self):
        '''Clear the old DataFile class by assigning and empty class to self.df
           and assuming the python garbage collector will take care of it.'''
        self.data_closed = True
        self.df = bu.DataFile()


        
    def save(self):
        parts = self.fname.split('.')
        if len(parts) > 2:
            print "Bad file name... too many periods/extensions"
            return
        else:
            savepath = '/processed_data/fildat' + parts[0] + '.fildat'
            bu.make_all_pardirs(savepath)
            pickle.dump(self, open(savepath, 'wb'))



    def load(self):
        parts = self.fname.split('.')
        if len(parts) > 2:
            print "Bad file name... too many periods/extensions"
            return
        else:
            loadpath = '/processed_data/fildat' + parts[0] + '.fildat'
            
            #try:
            old_class = pickle.load( open(loadpath, 'rb') )
            
            ### Load all of the class attributes
            self.__dict__.update(old_class.__dict__)

            # except:
            #     print "Couldn't find previously saved fildat"







class AggregateData:
    '''A class to store data from many files. Stores a FileDat object for each and
       has some methods that work on each object in a loop.'''

    
    def __init__(self, fnames, p0_bead=[16,0,20], tophatf=2500, harms=[], \
                 reload_dat=False):
        
        self.fnames = fnames
        self.p0_bead = p0_bead
        self.file_data_objs = []

        Nnames = len(self.fnames)

        suff = 'Processing %i files' % Nnames
        for name_ind, name in enumerate(self.fnames):
            bu.progress_bar(name_ind, Nnames, suffix=suff)

            # Initialize FileData obj, extract the data, then close the big file
            new_obj = FileData(name, tophatf=tophatf)
            if new_obj.badfile:
                continue

            if not reload_dat:
                new_obj.load()
            else:
                new_obj.extract_data(harms=harms)
                new_obj.load_position_and_bias()

                new_obj.close_datafile()

            new_obj.save()

            self.file_data_objs.append(new_obj)

        self.grav_loaded = False

        self.alpha_dict = ''
        self.agg_dict = ''
        self.avg_dict = ''

        self.ginds = ''


    def save(self, savepath):
        parts = savepath.split('.')
        if len(parts) > 2:
            print "Bad file name... too many periods/extensions"
            return
        else:
            if parts[1] != 'agg':
                print 'Changing file extension on save: %s -> .agg' % parts[1]
                savepath = parts[0] + '.agg'
            self.clear_grav_funcs()
            pickle.dump(self, open(savepath, 'wb'))
            self.reload_grav_funcs()


    def load(self, loadpath):
        parts = loadpath.split('.')
        if len(parts) > 2:
            print "Bad file name... too many periods/extensions"
            return
        else:
            print 'Loading aggregate data... ',
            sys.stdout.flush()
            if parts[1] != 'agg':
                print 'Changing file extension to match autosave: %s -> .agg' % parts[1]
                loadpath = parts[0] + '.agg'
            old_class = pickle.load( open(loadpath, 'rb') )
            self.__dict__.update(old_class.__dict__)
            print 'Done!'

            
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



    def make_templates(self, posvec, drivevec, ax1pos, ax2pos, ginds, \
                       single_lambda=False, single_lambind=0):
        
        xpos = self.p0_bead[0] + (80 - ax1pos)
        height = ax2pos - self.p0_bead[2]

        ones = np.ones_like(posvec)
        pts = np.stack((xpos*ones, posvec, height*ones), axis=-1)

        ## Include normal gravity in fit. But why???
        gfft = [[], [], []]
        for resp in [0,1,2]:
                    
            gforce = self.gfuncs[resp](pts*1.0e-6)
            gforce_func = interp.interp1d(posvec, gforce)

            gforcet = gforce_func(drivevec)
            gfft[resp] = np.fft.rfft(gforcet)[ginds]

        gfft = np.array(gfft)

        yuks = []
        for lambind, yuklambda in enumerate(self.lambdas):

            if single_lambda and (lambind != single_lambind):
                continue

            yukfft = [[], [], []]
            for resp in [0,1,2]:
                yukforce = self.yukfuncs[resp][lambind](pts*1.0e-6)
                yukforce_func = interp.interp1d(posvec, yukforce)

                yukforcet = yukforce_func(drivevec)
                yukfft[resp] = np.fft.rfft(yukforcet)[ginds]
            yukfft = np.array(yukfft)

            yuks.append(yukfft)

        outdict = {'gfft': gfft, 'yukffts': yuks}
        return outdict



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
            if type(self.ginds) == str:
                self.ginds = file_data_obj.ginds

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



    def average_resp_by_coordinate(self):
        '''Once data has been binned, average together the response and drive
           for every file at a given (height, sep)'''
    
        avg_dict = {}
        for bias in self.agg_dict.keys():
            avg_dict[bias] = {}
            for ax1key in self.ax1vec:
                avg_dict[bias][ax1key] = {}
                for ax2key in self.ax2vec:
                    avg_dict[bias][ax1key][ax2key] = {}

        suff = 'Averaging response at each position'
        i = 0
        totlen = len(self.agg_dict.keys()) * len(self.ax1vec) * len(self.ax2vec)
        for bias, ax1, ax2 in itertools.product(self.agg_dict.keys(), self.ax1vec, self.ax2vec):
            i += 1
            newline=False
            if i == totlen:
                newline=True
            bu.progress_bar(i, totlen, newline=newline, suffix=suff)

            ### Pull out fileData() objects at the same position
            objs = self.agg_dict[bias][ax1][ax2]

            nfiles = len(objs)
            filfac = 1.0 / float(nfiles)

            xpos = 0.0
            height = 0.0

            ### Initialize average arrays 
            drivevec_avg = np.zeros_like(objs[0].rebuild_drive())
            posvec_avg = np.zeros_like(objs[0].posvec)
            datfft_avg = np.zeros_like(objs[0].datfft)
            daterr_avg = np.zeros_like(objs[0].daterr)
            binned_avg = np.zeros_like(objs[0].binned)
            old_ginds = []

            ### Average over integrations at the same position
            for objind, obj in enumerate(objs):

                ### Assumes all data files haves same nsamp, fsamp
                if objind == 0:
                    self.ginds = obj.ginds
                    self.nsamp = obj.nsamp
                    self.fsamp = obj.fsamp

                xpos += filfac * (self.p0_bead[0] + (80 - obj.ax1pos))
                height += filfac * (obj.ax2pos - self.p0_bead[2])

                if not len(old_ginds):
                    old_ginds = obj.ginds
                np.testing.assert_array_equal(obj.ginds, old_ginds, \
                                              err_msg='notch filter changes between files...')
                old_ginds = obj.ginds

                drivevec = obj.rebuild_drive()
                drivevec_avg += filfac * drivevec

                posvec_avg += filfac * obj.posvec
                datfft_avg += filfac * obj.datfft
                daterr_avg += filfac * obj.daterr
                binned_avg += filfac * obj.binned

            ### Store the data for each position, including the full drivevec
            ### array, as there should only be ~100 different positions. It may be
            ### the case that we only need one drivevec for the whole thing, but 
            ### I have yet to demonstrate that
            avg_dict[bias][ax1][ax2]['drivevec'] = drivevec_avg
            avg_dict[bias][ax1][ax2]['posvec'] = posvec_avg
            avg_dict[bias][ax1][ax2]['datfft'] = datfft_avg
            avg_dict[bias][ax1][ax2]['daterr'] = daterr_avg
            avg_dict[bias][ax1][ax2]['binned'] = binned_avg

        self.avg_dict = avg_dict

        print




    def get_dataframes_averaged(self):
        '''Collects response into a DataFrame object for each coordinate,
           which is many averaged integrations.'''

        file_data_avg = pd.DataFrame(columns = ["ax1pos", "ax2pos", \
                                                "fft_data", "fft_errors"] )
        template_data = pd.DataFrame(columns = ["ax1pos", "ax2pos", "templates"])
        
        i = 0
        for ax1ind, ax1pos in enumerate(self.ax1vec):
            for ax2ind, ax2pos in enumerate(self.ax2vec):

                avg_dat = self.avg_dict[self.biasvec[0]][ax1pos][ax2pos]
                file_data_avg.loc[i] = [ax1pos, ax2pos, avg_dat['datfft'], avg_dat['daterr']]

                templates = self.make_templates(avg_dat['posvec'], avg_dat['drivevec'], \
                                                ax1pos, ax2pos, self.ginds)

                template_data.loc[i] = [ax1pos, ax2pos, templates]
                i += 1

        self.avg_dataframes = file_data_avg
        self.template_dataframes = template_data

    


    def get_dataframes_raw(self):
        """Churns over the data making a DataFrame storing file level data
        At the harmonic level, and a Data frame of unique templates."""

        if not self.grav_loaded:
            print "Must load theory data first..."
            return
        
        Nobj = len(self.file_data_objs)

        dft = pd.DataFrame(columns = ["ax1pos", "ax2pos", "time", "phi", "fft_data", "fft_errors"])
        df_templates = pd.DataFrame(columns = ["ax1pos", "ax2pos", "templates"])

        temp_indx = 0
        #first loop over files
        for objind, file_data_obj in enumerate(self.file_data_objs):

            #create data frame holding info from ith file data object
            dft.loc[objind] = [file_data_obj.ax1pos, file_data_obj.ax2pos, file_data_obj.time, \
                    file_data_obj.phi_cm, file_data_obj.datfft, file_data_obj.daterr]

        return pd.Series([dft, df_templates], index = ["data", "templates"])






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
                




    def find_alpha_xyz_from_templates(self, plot=False, plot_basis=False):

        print 'Finding alpha for each coordinate via an FFT template fitting algorithm...'
        
        if not self.grav_loaded:
            print "FAILED: Must load thoery data first!"
            try:
                self.reload_grav_funcs()
                print "UN-FAILED: Loaded dat therory dat!"
            except:
                return

        alpha_xyz_dict = {}
        for bias in self.agg_dict.keys():
            alpha_xyz_dict[bias] = {}
            for ax1key in self.ax1vec:
                alpha_xyz_dict[bias][ax1key] = {}
                for ax2key in self.ax2vec:
                    alpha_xyz_dict[bias][ax1key][ax2key] = []

        ### Progress bar shit
        i = 0
        totlen = len(self.agg_dict.keys()) * len(self.ax1vec) * len(self.ax2vec)
        for bias, ax1, ax2 in itertools.product(self.agg_dict.keys(), self.ax1vec, self.ax2vec):
            
            ### Progress bar shit
            i += 1
            suff = '%i / %i position combinations' % (i, totlen)
            newline=False
            if i == totlen:
                newline=True

            file_data_objs = self.agg_dict[bias][ax1][ax2]

            j = 0
            totlen_2 = len(file_data_objs) * len(self.lambdas)
            for objind, obj in enumerate(file_data_objs):
                
                alpha_xyz_dict[bias][ax1][ax2].append([])

                drivevec = obj.rebuild_drive()
                posvec = obj.posvec
                datfft = obj.datfft
                daterr = obj.daterr
                binned = obj.binned


                ## Loop over lambdas and do the template analysis for each value of lambda
                for lambind, yuklambda in enumerate(self.lambdas):
                    ### Progress bar shit
                    bu.progress_bar(j, totlen_2, suffix=suff, newline=newline)
                    j += 1
    
                    amps = [[], [], []]
                    errs = [[], [], []]
                    templates = self.make_templates(posvec, drivevec, ax1, ax2, self.ginds, \
                                                    single_lambda=True, single_lambind=lambind)

                    if plot and i == 0:
                        fig, axarr = plt.subplots(3,1,sharex=True,sharey=False)

                    for resp in [0,1,2]:

                        ### Get the modified gravity fft template, with alpha = 1
                        yukfft = templates['yukffts'][0][resp]

                        template_vec = np.concatenate((yukfft.real, yukfft.imag))

                        c_datfft = datfft[resp] #+ 1.0e15 * yukfft
                        data_vec = np.concatenate((c_datfft.real, c_datfft.imag))
                        err_vec = np.concatenate((daterr[resp].real, daterr[resp].imag))

                        ### Compute an 2*Nharmonic-dimensional basis for the real and 
                        ### imaginary components of our template signal yukfft, where
                        ### the template itself will be one of the orthogonal basis vectors
                        bases = make_basis_from_template_vec(template_vec)

                        ### The SVD decomposition above should produce an orthonormal basis
                        ### but this function is included to demonstrate that
                        ortho_basis = gram_schmidt(bases['real_basis'], plot=plot_basis)['orthogonal']

                        ### Loop over our orthogonal basis vectors and compute the inner 
                        ### product of the data and the basis vector
                        #c_amps = np.sqrt( np.einsum('ij,j->i', ortho_basis, data_vec) )
                        #c_errs = np.sqrt( np.einsum('ij,j->i', ortho_basis, err_vec) )
                        for k in range(len(ortho_basis)):
                            amps[resp].append( np.inner( ortho_basis[k], data_vec) )
                            errs[resp].append( np.inner( ortho_basis[k], err_vec ) )

                        ### Normalize the amplitudes to units of alpha (convert previous inner
                        ### products to positive-definite projections)
                        amps[resp] = amps[resp] / np.inner(template_vec, template_vec)
                        errs[resp] = np.abs(errs[resp]) / np.inner(template_vec, template_vec)
                        if plot and i == 0:
                            axarr[resp].errorbar(range(len(amps[resp])), \
                                                 amps[resp], errs[resp], fmt='o')

                    if plot and i == 0:
                        plt.show()

                    alpha_xyz_dict[bias][ax1][ax2][objind].append([amps, errs])

        print 'Done!'   
        self.alpha_xyz_dict = alpha_xyz_dict






    def find_alpha_xyz_from_templates_avg(self, plot=False, plot_basis=False):

        print
        print 'Finding alpha for each coordinate via an FFT'
        print 'template fitting algorithm'
        
        if not self.grav_loaded:
            print "FAILED: Must load thoery data first!"
            try:
                self.reload_grav_funcs()
                print "UN-FAILED: Loaded dat therory dat!"
            except:
                return

        if type(self.avg_dict) == str:
            self.average_resp_by_coordinate()

        alpha_xyz_dict = {}
        for bias in self.avg_dict.keys():
            alpha_xyz_dict[bias] = {}
            for ax1key in self.ax1vec:
                alpha_xyz_dict[bias][ax1key] = {}
                for ax2key in self.ax2vec:
                    alpha_xyz_dict[bias][ax1key][ax2key] = []

        i = 0
        totlen = len(self.avg_dict.keys()) * len(self.ax1vec) * len(self.ax2vec)
        for bias, ax1, ax2 in itertools.product(self.avg_dict.keys(), self.ax1vec, self.ax2vec):
            ### Progress bar shit
            i += 1
            suff = '%i / %i position combinations' % (i, totlen)
            newline=False
            if i == totlen:
                newline=True

            avg_dat = self.avg_dict[bias][ax1][ax2]

            drivevec_avg = avg_dat['drivevec']
            posvec_avg = avg_dat['posvec']
            datfft_avg = avg_dat['datfft']
            daterr_avg = avg_dat['daterr']
            binned_avg = avg_dat['binned']


            ## Loop over lambdas and do the template analysis for each value of lambda
            for lambind, yuklambda in enumerate(self.lambdas):
                bu.progress_bar(lambind, len(self.lambdas), suffix=suff, newline=newline)

                amps = [[], [], []]
                errs = [[], [], []]
                templates = self.make_templates(posvec_avg, drivevec_avg, ax1, ax2, self.ginds, \
                                                single_lambda=True, single_lambind=lambind)

                if plot and i == 0:
                    fig, axarr = plt.subplots(3,1,sharex=True,sharey=False)

                for resp in [0,1,2]:

                    ### Get the modified gravity fft template, with alpha = 1
                    yukfft = templates['yukffts'][0][resp]
                
                    template_vec = np.concatenate((yukfft.real, yukfft.imag))

                    c_datfft = datfft_avg[resp] #+ 1.0e15 * yukfft
                    data_vec = np.concatenate((c_datfft.real, c_datfft.imag))
                    err_vec = np.concatenate((daterr_avg[resp].real, daterr_avg[resp].imag))

                    ### Compute an 2*Nharmonic-dimensional basis for the real and 
                    ### imaginary components of our template signal yukfft, where
                    ### the template itself will be one of the orthogonal basis vectors
                    bases = make_basis_from_template_vec(template_vec)

                    ### The SVD decomposition above should produce an orthonormal basis
                    ### but this function is included to demonstrate that
                    ortho_basis = gram_schmidt(bases['real_basis'], plot=plot_basis)['orthogonal']

                    ### Loop over our orthogonal basis vectors and compute the inner 
                    ### product of the data and the basis vector
                    #c_amps = np.sqrt( np.einsum('ij,j->i', ortho_basis, data_vec) )
                    #c_errs = np.sqrt( np.einsum('ij,j->i', ortho_basis, err_vec) )
                    for j in range(len(ortho_basis)):
                        amps[resp].append( np.inner( ortho_basis[j], data_vec) )
                        errs[resp].append( np.inner( ortho_basis[j], err_vec ) )

                    ### Normalize the amplitudes to units of alpha (convert previous inner
                    ### products to positive-definite projections)
                    amps[resp] = amps[resp] / np.inner(template_vec, template_vec)
                    errs[resp] = np.abs(errs[resp]) / np.inner(template_vec, template_vec)
                    if plot and i == 0:
                        axarr[resp].errorbar(range(len(amps[resp])), \
                                             amps[resp], errs[resp], fmt='o')

                if plot and i == 0:
                    plt.show()

                alpha_xyz_dict[bias][ax1][ax2].append([amps, errs])

        print 'Done!'   
        self.alpha_xyz_dict_avg = alpha_xyz_dict





    def find_mean_alpha_vs_position_avg(self, ignoreXYZ=(0,0,0)):

        print 'Finding alpha vs. height/separation...'
        
        if not self.grav_loaded:
            print "FAILED: Must load thoery data first!"
            return

        if type(self.avg_dict) == str:
            self.average_resp_by_coordinate()

        alpha_dict = {}
        for bias in self.avg_dict.keys():
            alpha_dict[bias] = {}
            for ax1key in self.ax1vec:
                alpha_dict[bias][ax1key] = {}
                for ax2key in self.ax2vec:
                    alpha_dict[bias][ax1key][ax2key] = []

        i = 0
        totlen = len(self.avg_dict.keys()) * len(self.ax1vec) * len(self.ax2vec)
        for bias, ax1, ax2 in itertools.product(self.avg_dict.keys(), self.ax1vec, self.ax2vec):
            ### Progress bar shit
            i += 1
            suff = '%i / %i position combinations' % (i, totlen)
            newline=False
            if i == totlen:
                newline=True

            avg_dat = self.avg_dict[bias][ax1][ax2]

            drivevec_avg = avg_dat['drivevec_avg']
            posvec_avg = avg_dat['posvec_avg']
            datfft_avg = avg_dat['datfft']
            daterr_avg = avg_dat['daterr']
            binned_avg = avg_dat['binned']

            best_fit_alphas = np.zeros(len(self.lambdas))
            best_fit_errs = np.zeros(len(self.lambdas))

            ## Loop over lambdas and             
            for lambind, yuklambda in enumerate(self.lambdas):
                bu.progress_bar(lambind, len(self.lambdas), suffix=suff, newline=newline)

                newalpha = 2.0 * np.mean( np.abs(datfft_avg[0]) ) / \
                           np.mean( np.abs(yukfft[0]) ) * 1.5*10**(-1)
                testalphas = np.linspace(-1.0*newalpha, newalpha, 51)

                chi_sq_dat = get_chi2_vs_param_complex(datfft_avg, daterr_avg, ignoreXYZ, \
                                                       yukfft, testalphas)
                chi_sqs = chi_sq_dat['red_chi_sqs']
                fit_result = fit_parabola_to_chi2(testalphas, chi_sqs)

                best_fit_alphas[lambind] = fit_result['best_fit_param']
                best_fit_errs[lambind] = fit_result['param95']

            alpha_dict[bias][ax1][ax2] = [best_fit_alphas, best_fit_errs]

        print 'Done!'   
        self.alpha_dict = alpha_dict











    def fit_alpha_xyz_vs_alldim(self, weight_planar=True):

        if not self.grav_loaded:
            try:
                self.reload_grav_funcs()
            except:
                print 'No grav funcs... Tried to reload but no filename'

        alpha_xyz_best_fit = [[[[] for k in range(2 * np.sum(self.ginds))] \
                                    for resp in [0,1,2]] \
                                   for yuklambda in self.lambdas]

        ### Assume separations are encoded in ax1 and heights in ax2
        seps = 80 + self.p0_bead[0] - np.array(self.ax1vec)
        heights = self.p0_bead[2] - np.array(self.ax2vec) 
        
        ### Sort the heights and separations and build a grid
        sort1 = np.argsort(seps)
        sort2 = np.argsort(heights)
        seps_sort = seps[sort1]
        heights_sort = heights[sort2]
        heights_g, seps_g = np.meshgrid(heights_sort, seps_sort)

        ### Progress bar shit
        ind = 0
        totlen = len(self.lambdas) * (2 * np.sum(self.ginds)) * 3

        for bias in self.biasvec:
            ### Doesn't actually handle different biases correctly, although the
            ### data is structured such that if different biases are present
            ### they will be in distinct datasets

            for lambind, yuklambda in enumerate(self.lambdas):

                for resp in [0,1,2]:

                    dat = []
                    errs = []
                    for k in range(2 * np.sum(self.ginds)):
                        dat.append([[[] for i in range(len(heights))] for j in range(len(seps))])
                        errs.append([[[] for i in range(len(heights))] for j in range(len(seps))])

                    for ax1ind, ax1pos in enumerate(self.ax1vec):

                        ### Loop over all files at each separation and collect
                        ### the value of alpha for the current value of yuklambda
                        for ax2ind, ax2pos in enumerate(self.ax2vec):

                            tempdat = self.alpha_xyz_dict[bias][ax1pos][ax2pos]

                            for fileind, filedat in enumerate(tempdat):
    
                                for k in range(2 * np.sum(self.ginds)):
                                    dat[k][ax1ind][ax2ind].append(tempdat[fileind][lambind][0][resp][k])
                                    errs[k][ax1ind][ax2ind].append(tempdat[fileind][lambind][1][resp][k])

                    dat = np.array(dat)
                    errs = np.array(errs)

                    ### Since the data dictionary was indexed by cantilever settings 
                    ### rather than actual bead positions, we have to sort the data
                    ### to match the sorted separations and heights
                    for k in range(2 * np.sum(self.ginds)):
                        dat[k] = dat[k][sort1,:]
                        dat[k] = dat[k][:,sort2]
                        errs[k] = errs[k][sort1,:]
                        errs[k] = errs[k][:,sort2]
    
                    #if lambind == 0:
                    #    fig = plt.figure()
                    #    ax = fig.gca(projection='3d')
                    #    for k in range(2 * np.sum(self.ginds)):
                    #        ax.scatter(heights_g, seps_g, dat[k], label='Best-fit alphas')
                    #        #ax.scatter(heights_g, seps_g, errs[k], \
                    #        #           label='Errors for weighting in planar fit')
                    #    ax.legend()
                    #    ax.set_xlabel('Z-position [um]')
                    #    ax.set_ylabel('X-separation [um]')
                    #    ax.set_zlabel('Alpha %s [arb]' % ax_dict[resp])
                    #    if resp == 2:
                    #        plt.show()
    
                    #scale_fac = np.mean(dat.flatten())
                    scale_fac = 10**9

                    #dat_sc = dat * (1.0 / scale_fac)
                    #errs_sc = errs * (1.0 / scale_fac)
                    #if not weight_planar:
                    #    errs_sc = np.ones_like(dat_sc)

                    for k in range(2 * np.sum(self.ginds)):
    
                        bu.progress_bar(ind, totlen, suffix='Fitting planes... ')
                        ind += 1
                        #print i,
    
    
                        if k == 0:
                            test_grids = []
                            for thing in range(len(dat[k][0][0])):
                                test_grids.append( np.zeros_like(seps_g) )
                            fig = plt.figure()
                            ax = fig.gca(projection='3d')
    
    
                        pts = []
                        for sepind, sep in enumerate(seps_sort):
                            for heightind, height in enumerate(heights_sort):
    
                                for filind in range(len(dat[k][sepind][heightind])):
    
                                    if k == 0:
                                        if filind >= len(test_grids) - 1:
                                            continue
                                        test_grids[filind][heightind][sepind] = \
                                                            dat[k][sepind][heightind][filind]
    
                                    #if dat[k][sepind][heightind][filind] > 5.0e10:
                                    #    continue
    
                                    pts.append([ sep, height, \
                                                 dat[k][sepind][heightind][filind], \
                                                 errs[k][sepind][heightind][filind] ])
                        pts = np.array(pts)
                        
                        pts[:,2] *= (1.0 / scale_fac)
                        pts[:,3] *= pts[:,3] * (1.0 / scale_fac)
    
                        ### Defined a function to minimize via least-squared optimization
                        def func(params):
                            funcval = params[0] * pts[:,0] + params[1] * pts[:,1] + params[2]
                            return np.sum( (pts[:,2] - funcval) / pts[:,3] )

                        print np.mean(pts[:,2])
                        print ind
    
                        ### Optimize the previously defined function
                        res = opti.minimize(func, [0.2*np.mean(pts[:,2]), 0.2*np.mean(pts[:,2]), 0])

                        x = res.x
    
                        if k == 0:
                            for thing in range(len(dat[k][sepind][heightind])):
                                ax.scatter(heights_g, seps_g, test_grid[filind])
                            fit = (heights_g * x[1] + seps_g * x[0] + x[2])
                            ax.plot_surface(heights_g, seps_g, fit, alpha=0.3)
                            plt.show()
    
                        #print res.success

                        ### Deplane the data and extract some statistics
                        #deplaned = dat[k] - scale_fac * (x[0] * heights_g + x[1] * seps_g + x[2])
                        #deplaned_wconst = deplaned + scale_fac * x[2]

                        #deplaned_avg = np.mean(deplaned)
                        #deplaned_std = np.std(deplaned) / dat.size

                        alpha_xyz_best_fit[lambind][resp][k] = np.abs(x[2] * scale_fac)
    
        
        self.alpha_xyz_best_fit = np.array(alpha_xyz_best_fit)
        raw_input()








    def fit_alpha_xyz_vs_alldim_avg(self, weight_planar=True):

        if not self.grav_loaded:
            try:
                self.reload_grav_funcs()
            except:
                print 'No grav funcs... Tried to reload but no filename'

        self.alpha_xyz_best_fit = [[], [], []]
        self.alpha_xyz_95cl = [[], [], []]

        self.alpha_xyz_best_fit = [[[[] for k in range(2 * np.sum(self.ginds))] \
                                    for resp in [0,1,2]] \
                                   for yuklambda in self.lambdas]

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
            ### Doesn't actually handle different biases correctly, although the
            ### data is structured such that if different biases are present
            ### they will be in distinct datasets

            for lambind, yuklambda in enumerate(self.lambdas):

                xyz_dat = [[], [], []]

                for resp in [0,1,2]:

                    dat = []
                    errs = []
                    for k in range(2 * np.sum(self.ginds)):
                        dat.append([[[] for i in range(len(heights))] for j in range(len(seps))])
                        errs.append([[[] for i in range(len(heights))] for j in range(len(seps))])

                    for ax1ind, ax1pos in enumerate(self.ax1vec):

                        ### Loop over all files at each separation and collect
                        ### the value of alpha for the current value of yuklambda
                        for ax2ind, ax2pos in enumerate(self.ax2vec):

                            tempdat = self.alpha_xyz_dict_avg[bias][ax1pos][ax2pos]

                            for k in range(2 * np.sum(self.ginds)):
                                dat[k][ax1ind][ax2ind] = tempdat[lambind][0][resp][k]
                                errs[k][ax1ind][ax2ind] = tempdat[lambind][1][resp][k]

                    dat = np.array(dat)
                    errs = np.array(errs)

                    ### Since the data dictionary was indexed by cantilever settings 
                    ### rather than actual bead positions, we have to sort the data
                    ### to match the sorted separations and heights
                    for k in range(2 * np.sum(self.ginds)):
                        dat[k] = dat[k][sort1,:]
                        dat[k] = dat[k][:,sort2]
                        errs[k] = errs[k][sort1,:]
                        errs[k] = errs[k][:,sort2]
    
                    if lambind == 0:
                        fig = plt.figure()
                        ax = fig.gca(projection='3d')
                        for k in range(2 * np.sum(self.ginds)):
                            ax.scatter(heights_g, seps_g, dat[k], label='Best-fit alphas')
                            #ax.scatter(heights_g, seps_g, errs[k], \
                            #           label='Errors for weighting in planar fit')
                        ax.legend()
                        ax.set_xlabel('Z-position [um]')
                        ax.set_ylabel('X-separation [um]')
                        ax.set_zlabel('Alpha %s [arb]' % ax_dict[resp])
                        if resp == 2:
                            plt.show()
    
                    scale_fac = np.mean(dat)

                    dat_sc = dat * (1.0 / scale_fac)
                    errs_sc = errs * (1.0 / scale_fac)
                    if not weight_planar:
                        errs_sc = np.ones_like(dat_sc)

                    for k in range(2 * np.sum(self.ginds)):
                        ### Defined a function to minimize via least-squared optimization
                        def func(params, fdat=dat_sc[k], ferrs=errs_sc[k]):
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
                        deplaned = dat[k] - scale_fac * (x[0] * heights_g + x[1] * seps_g + x[2])
                        deplaned_wconst = deplaned + scale_fac * x[2]

                        deplaned_avg = np.mean(deplaned)
                        deplaned_std = np.std(deplaned) / dat.size

                        self.alpha_xyz_best_fit[lambind][resp][k] = np.abs(x[2]*scale_fac)
                        #self.alpha_xyz_95cl[resp].append(deplaned_std)

            #plt.plot(lambdas, alphas)















    def fit_mean_alpha_vs_alldim(self, weight_planar=True):

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
            ### Doesn't actually handle different biases correctly, although the
            ### data is structured such that if different biases are present
            ### they will be in distinct datasets

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



