import os, sys, time, itertools, copy, re

import dill as pickle

import numpy as np
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

from tqdm import tqdm
from joblib import Parallel, delayed

from iminuit import Minuit, describe

import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 14})

### Current constraints

limitdata_path = '/data/old_trap_processed/sensitivities/decca1_limits.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/data/old_trap_processed/sensitivities/decca2_limits.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'



### Stats

confidence_level = 0.95
chi2dist = stats.chi2(1)
# factor of 0.5 from Wilks's theorem: -2 log (Liklihood) ~ chi^2(1)
con_val = 0.5 * chi2dist.ppf(confidence_level)


ax_dict = {0: 'X', 1: 'Y', 2: 'Z'}




def build_paths(dirname, opt_ext='', new_trap=False):
    date = re.search(r"\d{8,}", dirname)[0]
    parts = dirname.split('/')

    if new_trap:
        bead_label = re.search(r"Bead\d", dirname)[0]
    else:
        bead_label = re.search(r"bead\d", dirname)[0]

    nobead = ('no_bead' in parts) or ('nobead' in parts) or ('no-bead' in parts)
    if nobead:
        opt_ext += '_NO-BEAD'

    newstr = 'old'
    if new_trap:
        newstr = 'new'

    if not len(parts[-1]):
        name = parts[-2]
    else:
        name = parts[-1]

    agg_path = '/data/{:s}_trap_processed/aggdat/'.format(newstr) \
                    + date + '_' + name + opt_ext + '.agg'
    alpha_dict_path = '/data/{:s}_trap_processed/alpha_dicts/'.format(newstr) \
                        + date + '_' + name + opt_ext + '.dict'
    alpha_arr_path = '/data/{:s}_trap_processed/alpha_arrs/'.format(newstr) \
                        + date + '_' + name + opt_ext + '.arr'
    plot_dir = '/home/cblakemore/plots/{:s}/mod_grav/'.format(date)
    if len(opt_ext):
        plot_dir = os.path.join(plot_dir, opt_ext)
        if plot_dir[-1] != '/':
            plot_dir += '/'

    return {'agg_path': agg_path, \
            'alpha_dict_path': alpha_dict_path, \
            'alpha_arr_path': alpha_arr_path, \
            'date': date, \
            'name': name, \
            'plot_dir': plot_dir}




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

def cauchy(x, A, mu, gamma):
    '''Cauchy (Lorentz) distribution with standard location
     and scale parameters. An amplitude is also included for
     fitting arbitrary distributions of data.'''
    return A / (np.pi * gamma * (1 + (x - mu)**2 / gamma**2))


def r2_goodness_of_fit(func, xdata, ydata, params):
    ymean = np.mean(ydata)
    SStot = np.sum( (ydata - ymean)**2 )
    SSres = np.sum( (ydata - func(xdata, *params))**2 )
    return 1 - SSres / SStot



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
    except Exception:
        print("Couldn't fit")
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

    # print('Lims')
    # print(xlim)
    # print(ylim)
    # print(zlim)

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










class GravFuncs:

    def __init__(self, theory_data_dir, load=True, verbose=True):
        if load:
            self.load_grav_funcs(theory_data_dir, verbose=verbose)
        else:
            self.grav_loaded = False


    def load_grav_funcs(self, theory_data_dir, verbose=True):
        self.theory_data_dir = theory_data_dir
        if verbose:
            print("Loading Gravity Data...", end=' ')
            sys.stdout.flush()
        grav_dict = build_mod_grav_funcs(theory_data_dir)
        self.gfuncs = grav_dict['gfuncs']
        self.yukfuncs = grav_dict['yukfuncs']
        self.lambdas = grav_dict['lambdas']
        self.lims = grav_dict['lims']
        self.grav_loaded = True
        if verbose:
            print("Done!")


    def reload_grav_funcs(self):
        try:
            self.load_grav_funcs(self.theory_data_dir,verbose=False)
        except Exception:
            print('No theory_data_dir saved')


    def clear_grav_funcs(self):
        self.gfuncs = ''
        self.yukfuncs = ''
        self.lambdas = ''
        self.lims = ''
        self.grav_loaded = False



    def make_templates(self, cant_posvec, drivevec, ax0pos, ax1pos, ginds, \
                       p0_bead, fsamp, single_lambda=False, single_lambind=0, \
                       new_trap=False, plot=True, n_largest_harms=100):

        xpos = p0_bead[0] - ax0pos
        height = p0_bead[2] - ax1pos
        posvec = p0_bead[1] - cant_posvec

        drivevec = p0_bead[1] - drivevec

        nharm = np.min([len(ginds), n_largest_harms])

        ones = np.ones_like(posvec)
        pts = np.stack((xpos*ones, posvec, height*ones), axis=-1)

        nsamp = len(drivevec)
        freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)

        bin_sp = freqs[1] - freqs[0]
        normfac = np.sqrt(2.0 * bin_sp) * bu.fft_norm(nsamp, fsamp)

        ## Include normal gravity in fit. But why???
        gfft = np.zeros((3, nharm), dtype=np.complex128)
        gbool = np.zeros((3, len(ginds)), dtype=bool)
        for resp in [0,1,2]:
                    
            gforce = self.gfuncs[resp](pts*1.0e-6)
            gforce_func = interp.interp1d(posvec, gforce)

            gforcet = gforce_func(drivevec)
            curr_gfft = np.fft.rfft(gforcet)[ginds] * normfac
            curr_asd = np.abs(curr_gfft)

            thresh = (curr_asd[np.argsort(curr_asd)[::-1]])[nharm-1]
            bool_ginds = curr_asd >= thresh
            gbool[resp,:] = bool_ginds

            gfft[resp] += np.fft.rfft(gforcet)[ginds[bool_ginds]] * normfac

        yuks = np.zeros( (len(self.lambdas), 3, nharm), dtype=np.complex128)
        yukbool = np.zeros( (len(self.lambdas), 3, len(ginds)), dtype=bool)
        for lambind, yuklambda in enumerate(self.lambdas):

            if single_lambda and (lambind != single_lambind):
                continue

            if plot:
                fig1, ax1 = plt.subplots(1,1)
                fig2, ax2 = plt.subplots(1,1)
                fig3, ax3 = plt.subplots(1,1)
                fig4, ax4 = plt.subplots(1,1)

                ax1.set_title('Attractor Drive')
                ax1.plot((1.0 / fsamp) * np.arange(nsamp), drivevec)
                ax1.set_xlabel('Time [s]')
                ax1.set_ylabel('Attractor drive [um]')
                fig1.tight_layout()

            resp_dict = {0: 'X', 1: 'Y', 2: 'Z'}
            marker_dict = {0: 'o', 1: 'P', 2: 'X'}
            for resp in [0,1,2]:
                yukforce = self.yukfuncs[resp][lambind](pts*1.0e-6)
                yukforce_func = interp.interp1d(posvec, yukforce)

                yukforcet = yukforce_func(drivevec)
                curr_yukfft = np.fft.rfft(yukforcet)[ginds] * normfac
                curr_asd = np.abs(curr_yukfft)

                thresh = (curr_asd[np.argsort(curr_asd)[::-1]])[nharm-1]
                bool_ginds = curr_asd >= thresh
                yukbool[lambind,resp,:] = bool_ginds

                yuks[lambind,resp,:] += np.fft.rfft(yukforcet)[ginds[bool_ginds]] * normfac


                if plot:
                    ax2.plot((1.0 / fsamp) * np.arange(nsamp), yukforcet, label=resp_dict[resp])
                    ax3.plot(posvec, yukforce, label=resp_dict[resp])
                    ax4.loglog(freqs, np.abs(np.fft.rfft(yukforcet))*normfac, \
                                ls='', marker=marker_dict[resp], alpha=0.3, \
                                color='C{:d}'.format(resp), ms=5)
                    ax4.loglog(freqs[ginds[yukbool[lambind,resp,:]]], \
                               np.abs(yuks[lambind,resp,:]), \
                               ls='', marker=marker_dict[resp], label=resp_dict[resp], \
                               color='C{:d}'.format(resp), ms=10)

            if plot:
                title_str = 'Force for $\\alpha = 1$ and $\\lambda = {:0.1g}$ m'\
                                .format(yuklambda)
                ax2.set_title(title_str)
                ax2.set_xlabel('Time [s]')
                ax2.set_ylabel('Force [N]')
                ax2.legend(fontsize=10, ncol=3)
                fig2.tight_layout()

                ax3.set_title(title_str)
                ax3.set_xlabel('Position along Density Modulation [um]')
                ax3.set_ylabel('Force [N]')
                ax3.legend(fontsize=10, ncol=3)
                fig3.tight_layout()

                ax4.set_title(title_str)
                ax4.set_xlabel('Frequency [Hz]')
                ax4.set_ylabel('Force Spectral Density [N/$\\sqrt{\\rm{Hz}}$]')
                ax4.legend(fontsize=10, ncol=3)
                fig4.tight_layout()

                plt.show()

                input()

        yukamp = np.mean(np.abs(yuks[0]))
        #yukstr = '%0.3e' % yukamp
        #print xpos, height, yukstr,

        outdict = {'gfft': gfft, 'gbool': gbool, \
                   'yukffts': yuks, 'yukbool': yukbool, \
                   'debug': (xpos, height, yukamp)}
        return outdict









class FileData:
    '''A class to store data from a single file, only
       what is relevant for higher level analysis.'''
    

    def __init__(self, fname, diagonalize=True, tfdate='', tophatf=2500, \
                    plot_tf=False, step_cal_drive_freq=41.0, \
                    new_trap=False, empty=False, suppress_off_diag=False):
        '''Load an hdf5 file into a bead_util.DataFile obj. Calibrate the stage position.
           Calibrate the microsphere response with th transfer function.'''


        self.tfdate = tfdate
        self.plot_tf = plot_tf
        self.step_cal_drive_freq = step_cal_drive_freq
        self.tophatf = tophatf

        self.new_trap = new_trap
        self.suppress_off_diag = suppress_off_diag

        self.empty = empty
        if not self.empty:

            df = bu.DataFile()
            self.fname = fname
            try:
                if new_trap:
                    df.load_new(fname)
                else:
                    df.load(fname)
                self.badfile = False
            except Exception:
                self.badfile = True
                return

            df.calibrate_stage_position()
            if diagonalize:
                df.diagonalize(date=tfdate, maxfreq=tophatf, plot=plot_tf, \
                                step_cal_drive_freq=step_cal_drive_freq, \
                                suppress_off_diag=self.suppress_off_diag)

            # self.xy_tf_res_freqs = df.xy_tf_res_freqs
    
            self.time = df.time
            self.fsamp = df.fsamp
            self.nsamp = df.nsamp
            #self.phi_cm = df.phi_cm
            self.df = df

            self.data_closed = False



    def close_datafile(self):
        '''Clear the old DataFile class by assigning and empty class to self.df
           and assuming the python garbage collector will take care of it.'''
        self.data_closed = True
        self.df = bu.DataFile()



    def reload_datafile(self, diagonalize=True):
        '''Reload the raw HDF5 data back into the class for debugging purposes.'''
        self.data_closed = False

        df = bu.DataFile()
        if self.new_trap:
            df.load_new(self.fname)
        else:
            df.load(self.fname)

        df.calibrate_stage_position()

        if diagonalize:
            df.diagonalize(date=self.tfdate, maxfreq=self.tophatf, \
                            step_cal_drive_freq=self.step_cal_drive_freq, \
                            plot=self.plot_tf, interpolate=self.tf_interp, \
                            suppress_off_diag=self.suppress_off_diag)
        self.df = df


    def rebuild_drive(self):

        ## Transform ginds array to array of indices for drive freq
        #temp_inds = np.array(range(int(self.nsamp*0.5) + 1))
        #inds = temp_inds[self.ginds]
        inds = self.drive_ginds

        freqs = np.fft.rfftfreq(self.nsamp, d=1.0/self.fsamp)
        bin_sp = freqs[1] - freqs[0]
        normfac = np.sqrt(2.0 * bin_sp) * bu.fft_norm(self.nsamp, self.fsamp)

        full_drive_fft = np.zeros(len(freqs), dtype=np.complex128)
        for ind, freq_ind in enumerate(inds):
            full_drive_fft[freq_ind] = self.drivefft_all[ind] * (1.0 / normfac)

        drivevec = np.fft.irfft(full_drive_fft)

        npos = len(self.posvec)
        self.posvec = np.linspace(np.min(drivevec), np.max(drivevec), npos)

        return drivevec




    def extract_data(self, ext_cant=(False,1), nharmonics=10, harms=[], width=0, \
                     npos=500, noisebins=10, plot_harm_extraction=False, \
                     elec_drive=False, elec_ind=0, maxfreq=2500, noiselim=(10,100), \
                     find_xy_resonance=False):
        '''Extracts the microsphere resposne at the drive frequency of the cantilever, and
           however many harmonics are requested. Uses a notch filter that is default set 
           one FFT bin width, although this can be increased'''

        if self.data_closed:
            return

        ## Get the notch filter
        if elec_drive:
            drivefilt = self.df.get_boolean_elecfilt(nharmonics=nharmonics, harms=harms, \
                                                     width=width, elec_ind=elec_ind, \
                                                     maxfreq=maxfreq)
        else:
            drivefilt = self.df.get_boolean_cantfilt(ext_cant=ext_cant, nharmonics=nharmonics, \
                                                     harms=harms, width=width)
        #cantfilt_ginds = cantfilt['ginds']
        #self.ginds = np.arange(len(drivefilt_ginds)).astype(np.int)[drivefilt_ginds]

        self.drive_ginds = drivefilt['drive_ginds']
        self.ginds = drivefilt['ginds']
        self.fund_ind = drivefilt['fund_ind']
        self.drive_freq = drivefilt['drive_freq']
        self.drive_ind = drivefilt['drive_ind']

        ## Apply the notch filter
        fftdat = self.df.get_datffts_and_errs(self.ginds, self.drive_freq, self.drive_ginds, \
                                              noisebins=noisebins, \
                                              plot=plot_harm_extraction, \
                                              elec_drive=elec_drive, elec_ind=elec_ind, \
                                              noiselim=noiselim)
        self.err_ginds = fftdat['err_ginds']
        self.datfft = fftdat['datffts']
        self.daterr = fftdat['daterrs']
        self.diagdatfft = fftdat['diagdatffts']
        self.diagdaterr = fftdat['diagdaterrs']

        self.noisefft = fftdat['noiseffts']

        self.drivefft = fftdat['driveffts']
        self.drivefft_all = fftdat['driveffts_all']
        self.meandrive = fftdat['meandrive']

        ### Get the binned data and calibrate the non-diagonalized part
        self.df.get_force_v_pos(elec_drive=elec_drive, maxfreq=maxfreq, \
                                nharmonics=nharmonics, harms=harms)
        binned = np.array(self.df.binned_data)
        for resp in [0,1,2]:
            binned[resp][1] = binned[resp][1] * self.df.conv_facs[resp]
            binned[resp][2] = binned[resp][2] * self.df.conv_facs[resp]
        self.binned = binned

        #plt.plot(binned[0][0], binned[0][1])
        #plt.show()


        ### Analyze the attractor drive and build the relevant position vectors
        ### for the bead
        if elec_drive:
            mindrive = np.min(self.df.electrode_data[elec_ind])
            maxdrive = np.max(self.df.electrode_data[elec_ind])
        else:
            mindrive = np.min(self.df.cant_data[self.drive_ind])
            maxdrive = np.max(self.df.cant_data[self.drive_ind])

        self.posvec = np.linspace(mindrive, maxdrive, npos)

        if find_xy_resonance:
            # Assumes you were driving two tones on the electrode drive
            # that sit to either side of the resonance
            self.df.get_xy_resonance()



    def get_background_rms(self, harm_inds_to_use=[]):

        df = self.fsamp / self.nsamp
        fft_norm = bu.fft_norm(self.nsamp, self.fsamp) * np.sqrt(df)

        if not len(harm_inds_to_use):
            inds = list(range(len(self.datfft[0])))
            err_inds = list(range(len(self.daterr[0])))
            n_err = int(len(err_inds) / len(inds))
        else:
            inds = harm_inds_to_use

        xyz_background = np.sqrt( np.sum(self.datfft[:,inds]*self.datfft[:,inds].conj(), \
                                         axis=-1).real )
        xyz_noise = np.sqrt( np.sum(self.daterr[:,err_inds]*self.daterr[:,err_inds].conj(), \
                                    axis=-1).real * (1.0 / n_err) )
        diag_xyz_background = np.sqrt( np.sum(self.diagdatfft[:,inds]*self.diagdatfft[:,inds].conj(), \
                                              axis=-1).real * (1.0 / n_err) )
        diag_xyz_noise = np.sqrt( np.sum(self.diagdaterr[:,err_inds]*self.diagdaterr[:,err_inds].conj(), \
                                         axis=-1).real )

        self.background_rms = {'background'      :  xyz_background, \
                               'noise'           :  xyz_noise, \
                               'diag_background' :  diag_xyz_background, \
                               'diag_noise'      :  diag_xyz_noise}
            


    def load_position_and_bias(self, ax0='x', ax1='z', ax2='y', dim3=False):

        if dim3:
            ax0 = 'x'
            ax1 = 'y'
            ax2 = 'z'

        if not self.data_closed:
            ax_keys = {'x': 0, 'y': 1, 'z': 2}

            try:
                self.cantbias = self.df.electrode_settings['dc_settings'][0]
            except Exception:
                self.cantbias = 0.0

            ax0pos = np.mean(self.df.cant_data[ax_keys[ax0]])
            self.ax0pos = round(ax0pos, 1)

            ax1pos = np.mean(self.df.cant_data[ax_keys[ax1]])
            self.ax1pos = round(ax1pos, 1)

            ax2pos = np.mean(self.df.cant_data[ax_keys[ax2]])
            self.ax2pos = round(ax2pos, 1)
            
        else:
            print('ERROR: load_position_and_bias()')
            print('Whatchu doin boi?! The bu.DataFile() object is closed.')
            time.sleep(1)
            print('(press enter when you\'re ready to get a real Python error)')
            input()



    def generate_pts(self, p0):
        '''generates pts arry for determining gravitational template.'''

        xpos = p0[0] - self.ax0pos
        height = p0[2] - self.ax1pos
            
        ones = np.ones_like(self.posvec)
        pts = np.stack((xpos*ones, self.posvec, height*ones), axis=-1)

        drivevec = self.rebuild_drive()

        full_ones = np.ones_like(drivevec)
        full_pts = np.stack((xpos*full_ones, drivevec, height*full_ones), axis=-1)

        return full_pts



        
    def save(self, verbose=True):
        parts = self.fname.split('.')
        if len(parts) > 2:
            print("Bad file name... too many periods/extensions")
            return
        else:
            if verbose:
                print('Saving FileData object... ', end=' ')
                sys.stdout.flush()
            savepath = '/data/old_trap_processed/fildat' + parts[0] + '.fildat'
            bu.make_all_pardirs(savepath)
            pickle.dump(self, open(savepath, 'wb'))
            if verbose:
                print('Done!')
                print('Saved to: ', savepath)
                sys.stdout.flush()



    def load(self):
        parts = self.fname.split('.')
        if len(parts) > 2:
            print("Bad file name... too many periods/extensions")
            return
        else:
            loadpath = '/data/old_trap_processed/fildat' + parts[0] + '.fildat'
            
            #try:
            old_class = pickle.load( open(loadpath, 'rb') )
            
            ### Load all of the class attributes
            self.__dict__.update(old_class.__dict__)

            # except Exception:
            #     print "Couldn't find previously saved fildat"















class AggregateData:
    '''A class to store data from many files. Stores a FileDat object for each and
       has some methods that work on each object in a loop.'''

    
    def __init__(self, fnames, p0_bead=[0.0,0.0,0.0], tophatf=2500, harms=[], \
                 reload_dat=True, plot_harm_extraction=False, \
                 elec_drive=False, elec_ind=0, maxfreq=2500, noisebins=10,\
                 dim3=False, extract_resonant_freq=False, noiselim=(10.0,100.0), \
                 tfdate='', tf_interp=False, step_cal_drive_freq=41.0, \
                 new_trap=False, ncore=1, aux_data=[], suppress_off_diag=False):
        
        if new_trap:
            self.new_trap = True
        else:
            self.new_trap = False

        self.fnames = fnames
        self.p0_bead = p0_bead
        self.file_data_objs = []
        self.times = np.zeros(len(fnames))
        self.aux_data = aux_data

        # Nnames = len(self.fnames)

        # old_time = time.time()
        # per = 0
        # times = []

        # suff = 'Processing %i files' % Nnames
        # for name_ind, name in enumerate(self.fnames):

        #     ### Stuff for printing an ETA with the progress bar
        #     new_per = int( np.floor( 100.0 * float(name_ind) / float(Nnames) ) )
        #     if new_per != per:
        #         ctime = time.time()
        #         per_time = ctime - old_time
        #         times.append(per_time)
        #         old_time = ctime
        #         eta = np.mean(times) * (100.0 - new_per) * (1.0 / 60.0)
        #         per = new_per
        #         suff = 'Minutes remaining: %0.1f' % eta

        #     bu.progress_bar(name_ind, Nnames, suffix=suff)



        def process_file(name):

            # Initialize FileData obj, extract the data, then close the big file
            new_obj = FileData(name, tophatf=tophatf, tfdate=tfdate, new_trap=new_trap, \
                                step_cal_drive_freq=step_cal_drive_freq, \
                                suppress_off_diag=suppress_off_diag)

            if new_obj.badfile:
                print('FOUND BADDIE: ')
                print(name)
                #new_obj.load_position_and_bias(dim3=dim3)
                return

            no_drive = [0, 0, 0]
            for i in range(3):
                if np.std(new_obj.df.cant_data[i]) > 10.0:
                    no_drive[i] = 1
            if not np.sum(no_drive):
                print('Bad attractor data')
                return
                
            if not reload_dat and not new_obj.badfile:
                new_obj.load()
            else:
                new_obj.extract_data(harms=harms, noisebins=noisebins, \
                                     plot_harm_extraction=plot_harm_extraction, \
                                     elec_drive=elec_drive, elec_ind=elec_ind, \
                                     maxfreq=maxfreq, noiselim=noiselim)
                new_obj.load_position_and_bias(dim3=dim3)

                new_obj.close_datafile()

            new_obj.save(verbose=False)

            return new_obj
            #self.file_data_objs.append(new_obj)

        if len(self.fnames):
            file_data_objs = Parallel(n_jobs=ncore)(delayed(process_file)(name) \
                                                      for name in tqdm(self.fnames))
            self.file_data_objs = file_data_objs

            for ind, obj in enumerate(self.file_data_objs):
                self.times[ind] = obj.time

                obj.p0_bead = self.p0_bead

                try:
                    file_aux_data = \
                        self.aux_data.iloc[(self.aux_data["Time_Epoch"] - \
                                            obj.time).abs().idxmin()]
                    obj.aux_data = file_aux_data.to_dict()
                    obj.p0_bead[2] = obj.aux_data['height_cal']
                except Exception:
                    file_aux_data = []

            self.times0 = self.times - self.times[0]
        else:
            self.times0 = []

        self.alpha_dict = ''
        self.agg_dict = ''
        self.avg_dict = ''

        self.ginds = ''

        self.gfuncs_class = GravFuncs('', load=False)


    def save(self, savepath, verbose=True):
        parts = savepath.split('.')
        if len(parts) > 2:
            print("Bad file name... too many periods/extensions")
            return
        else:
            if verbose:
                print('Saving AggregateDate object... ', end=' ')
            if parts[1] != 'agg':
                if verbose:
                    print()
                    print('Changing file extension on save: %s -> .agg' % parts[1])
                savepath = parts[0] + '.agg'
            sys.stdout.flush()
            self.gfuncs_class.clear_grav_funcs()
            pickle.dump(self, open(savepath, 'wb'))
            self.gfuncs_class.reload_grav_funcs()
            if verbose:
                print('Done!')
                print('Saved to: ', savepath)
                sys.stdout.flush()


    def save_alpha_dict(self, savepath, verbose=True):
        if verbose:
            print('Saving alpha dict...', end=' ')
            sys.stdout.flush()
        savedict = {'dict': self.alpha_xyz_dict, \
                    'ax0vec': self.ax0vec, 'ax1vec': self.ax1vec}
        pickle.dump( savedict, open(savepath, 'wb') )
        if verbose:
            print('Done!')


    def save_alpha_arr(self, savepath, verbose=True):
        if verbose:
            print('Saving alpha arr...', end=' ')
            sys.stdout.flush()
        lambda_savepath = savepath[:-4] + '_lambdas.arr'
        np.save( open(savepath, 'wb'), self.alpha_xyz_best_fit)
        np.save( open(lambda_savepath, 'wb'), self.lambdas)
        if verbose:
            print('Done!')


    def load(self, loadpath):

        new_p0 = self.p0_bead

        parts = loadpath.split('.')
        if len(parts) > 2:
            print("Bad file name... too many periods/extensions")
            return
        else:
            print('Loading aggregate data... ', end=' ')
            sys.stdout.flush()
            if parts[1] != 'agg':
                print('Changing file extension to match autosave: %s -> .agg' % parts[1])
                loadpath = parts[0] + '.agg'
            old_class = pickle.load( open(loadpath, 'rb') )
            self.__dict__.update(old_class.__dict__)
            self.p0_bead = new_p0

            print('Done!')


    def load_alpha_dict(self, loadpath, verbose=True):
        if verbose:
            print('Loading alpha dict...', end=' ')
            sys.stdout.flush()
        loaddict = pickle.load( open(loadpath, 'rb') )
        self.alpha_xyz_dict = loaddict['dict']
        self.ax0vec = loaddict['ax0vec']
        self.ax1vec = loaddict['ax1vec']
        if verbose:
            print('Done!')


    def load_alpha_arr(self, loadpath, verbose=True):
        if verbose:
            print("Loading alpha arr...", end=' ')
            sys.stdout.flush()
        lambda_loadpath = loadpath[:-4] + '_lambdas.arr'
        self.alpha_xyz_best_fit = np.load( open(loadpath, 'rb') )
        self.lambdas = np.load( open(lambda_loadpath, 'rb') )
        if verbose:
            print("Done!")

            
    def load_grav_funcs(self, theory_data_dir, verbose=True):
        self.gfuncs_class = GravFuncs(theory_data_dir, verbose=verbose)




    def bin_rough_stage_positions(self, ax_disc=0.5, dim3=False):
        '''Loops over the preprocessed file_data_objs and organizes them by rough stage position,
           discretizing the rough stage position by a user-controlled parameter. Unfortunately,
           because the final object is a nested dictionary, it's somewhat cumbersome to put this
           into any subroutines.
        '''
        
        print('Sorting data by rough stage position...', end=' ')

        agg_dict = {}

        biasvec = []

        ax0vec = []
        Nax0 = {}
        ax1vec = []
        Nax1 = {}
        if dim3:
            ax2vec = []
            Nax2 = {}

        for file_data_obj in self.file_data_objs:
            if type(self.ginds) == str:
                self.ginds = file_data_obj.ginds

            bias = file_data_obj.cantbias
            ax0pos = file_data_obj.ax0pos
            ax1pos = file_data_obj.ax1pos
            ax2pos = file_data_obj.ax2pos

            if bias not in list(agg_dict.keys()):
                agg_dict[bias] = {}
                biasvec.append(bias)
                Nax0[bias] = {}
                Nax1[bias] = {}
                if dim3:
                    Nax2[bias] = {}
            

            #### Check for the first axis (usually X)
            ax0_is_new = False
            if len(ax0vec):
                ## Finds the closest position in the array (if the array already is populated)
                ## and then decides if the current position is "new", or should be averaged
                ## together with other positions
                close_ind = np.argmin( np.abs( np.array(ax0vec) - ax0pos ) )
                if np.abs(ax0vec[close_ind] - ax0pos) < ax_disc:
                    old_ax0key = ax0vec[close_ind]
                    oldN = Nax0[bias][old_ax0key]

                    ## Adjust the average position to include this new key
                    new_ax0key = (old_ax0key * oldN + ax0pos) / (oldN + 1.0)
                    new_ax0key = round(new_ax0key, 1)

                    ## If the discretized/rounded version of the average new key doesn't 
                    ## equal the average old key, then collect all of the data for both
                    ## keys under the new key
                    if old_ax0key != new_ax0key:
                        ax0vec[close_ind] = new_ax0key

                        agg_dict[bias][new_ax0key] = agg_dict[bias][old_ax0key]
                        Nax0[bias][new_ax0key] = oldN + 1.0

                        del Nax0[bias][old_ax0key]
                        del agg_dict[bias][old_ax0key]
                    else:
                        Nax0[bias][old_ax0key] += 1
                    
                else:
                    ax0_is_new = True
            else:
                ax0_is_new = True

            ## If the new position is truly "new", added it to the rough stage position
            ## vectors and makes a new dictionary entry
            if ax0_is_new:
                agg_dict[bias][ax0pos] = {}
                Nax0[bias][ax0pos] = 1
                new_ax0key = ax0pos
                ax0vec.append(new_ax0key)
                ax0vec.sort()



            #### Check for the second axis (usually Z)
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
               
                    if old_ax1key not in list(agg_dict[bias][new_ax0key].keys()):
                        if dim3:
                            agg_dict[bias][new_ax0key][new_ax1key] = {}
                        else:
                            agg_dict[bias][new_ax0key][new_ax1key] = []

                    ## If the discretized/rounded version of the average new key doesn't 
                    ## equal the average old key, then collect all of the data for both
                    ## keys under the new key
                    if old_ax1key != new_ax1key:
                        ax1vec[close_ind] = new_ax1key

                        Nax1[bias][new_ax1key] = oldN + 1.0
                        del Nax1[bias][old_ax1key]

                        for ax0key in ax0vec:
                            ax1keys = list(agg_dict[bias][ax0key].keys())
                            if old_ax1key in ax1keys:
                                agg_dict[bias][ax0key][new_ax1key] = agg_dict[bias][ax0key][old_ax1key]
                                del agg_dict[bias][ax0key][old_ax1key]
                    else:
                        Nax1[bias][old_ax1key] += 1
                    
                else:
                    ax1_is_new = True
            else:
                ax1_is_new = True

            ## If the new position is truly "new", addes it to the rough stage position
            ## vector and makes a new dictionary entry
            if ax1_is_new:
                if dim3:
                    agg_dict[bias][new_ax0key][ax1pos] = {}
                else:
                    agg_dict[bias][new_ax0key][ax1pos] = []
                Nax1[bias][ax1pos] = 1
                new_ax1key = ax1pos
                ax1vec.append(new_ax1key)
                ax1vec.sort()



            if dim3:
                #### Check for the third axis (usually Z)
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

                        if old_ax2key not in list(agg_dict[bias][new_ax0key][new_ax1key].keys()):
                            agg_dict[bias][new_ax0key][new_ax1key][new_ax2key] = []

                        ## If the discretized/rounded version of the average new key doesn't 
                        ## equal the average old key, then collect all of the data for both
                        ## keys under the new key
                        if old_ax2key != new_ax2key:
                            ax2vec[close_ind] = new_ax2key

                            Nax2[bias][new_ax2key] = oldN + 1.0
                            del Nax2[bias][old_ax2key]

                            for ax0key in ax0vec:
                                ax1keys = list(agg_dict[bias][ax0key].keys())
                                for ax1key in ax1keys:
                                    ax2keys = list(agg_dict[bias][ax0key][ax1key].keys())
                                    if old_ax2key in ax2keys:
                                        agg_dict[bias][ax0key][ax1key][new_ax2key] = \
                                                                agg_dict[bias][ax0key][ax1key][old_ax2key]
                                        del agg_dict[bias][ax0key][ax1key][old_ax2key]
                        else:
                            Nax2[bias][old_ax2key] += 1

                    else:
                        ax2_is_new = True
                else:
                    ax2_is_new = True

                ## If the new position is truly "new", adds it to the rough stage position
                ## vector and makes a new dictionary entry
                if ax2_is_new:
                    agg_dict[bias][new_ax0key][new_ax1key][ax2pos] = []
                    Nax2[bias][ax2pos] = 1
                    new_ax2key = ax2pos
                    ax2vec.append(new_ax2key)
                    ax2vec.sort()


            if dim3:
                agg_dict[bias][new_ax0key][new_ax1key][new_ax2key].append( file_data_obj )
            else:
                ## Add in the new data to our aggregate dictionary
                agg_dict[bias][new_ax0key][new_ax1key].append( file_data_obj )

        print('Done!')

        ax0vec.sort()
        ax1vec.sort()
        if dim3:
            ax2vec.sort()
        biasvec.sort()

        self.biasvec = biasvec
        self.ax0vec = ax0vec
        self.ax1vec = ax1vec
        if dim3:
            self.ax2vec = ax2vec
        self.agg_dict = agg_dict

        
        # print('Bin vecs')
        # print(ax0vec)
        # print(ax1vec)
        # print(ax2vec)

        # print()
        # print('key vecs')
        # print(np.sort(agg_dict[biasvec[0]].keys()))
        # print(np.sort(agg_dict[biasvec[0]][ax0vec[0]].keys()))
        # print(np.sort(agg_dict[biasvec[0]][ax0vec[0]][ax1vec[0]].keys()))

        # raw_input()
        


    def get_max_files(self, dim3=False, bad_axkeys=[[], [], []]):
        max_files = 10e9
        for bias in self.biasvec:

            for ax0 in self.ax0vec:
                if ax0 in bad_axkeys[0]:
                    continue

                for ax1 in self.ax1vec:
                    if ax1 in bad_axkeys[1]:
                        continue

                    if dim3:
                        for ax2 in self.ax2vec:
                            if ax2 in bad_axkeys[2]:
                                continue

                            numfiles = len(self.agg_dict[bias][ax0][ax1][ax2])
                            if numfiles < max_files:
                                max_files = numfiles

                    else:
                        numfiles = len(self.agg_dict[bias][ax0][ax1])
                        if numfiles < max_files:
                            max_files = numfiles

        self.max_files = max_files





    def handle_sparse_binning(self, dim3=False, verbose=False):
        '''Sometimes the rough stage binning above finds a bad file and entries 
           are missing from the nested dictionaries. This adds an empty file_data
           object at each missing entry, which can be ignored later
        '''
        ### Handle sparsity
        if verbose:
            print('ax0: ', self.ax0vec)
            print('ax1: ', self.ax1vec)
            print('ax2: ', self.ax2vec)
            print()
        print_bool = [False, False, False]

        bad_axkeys = [[], [], []]
        key_sets = []

        for bias in self.biasvec:
            ax0keys = list(self.agg_dict[bias].keys())
            ax0keys.sort()
            if not print_bool[0] and verbose:
                print('ax0k: ', ax0keys)
                print_bool[0] = True
            different = (len(ax0keys) != len(self.ax0vec))
            if different:
                if verbose:
                    #print different
                    print('Bad ax0keys: ', ax0keys)

            for ax0 in ax0keys:
                ax1keys = list(self.agg_dict[bias][ax0].keys())
                ax1keys.sort()
                if not print_bool[1] and verbose:
                    print('ax1k: ', ax1keys)
                    print_bool[1] = True
                different = (len(ax1keys) != len(self.ax1vec))
                if different:
                    if verbose:
                        #print diff
                        print('At AX0 = %0.2f' % ax0)
                        print('Bad ax1keys: ', ax1keys)
                    
                    if not dim3:
                        for key in self.ax1vec:
                            if key not in ax1keys:
                                bad_axkeys[1].append(key)
                                key_sets.append((ax0, key))

                
                if dim3:
                    for ax1 in ax1keys:
                        ax2keys = list(self.agg_dict[bias][ax0][ax1].keys())
                        ax2keys.sort()
                        if not print_bool[2] and verbose:
                            print('ax2k: ', ax2keys)
                            print_bool[2] = True
                        different = (len(ax2keys) != len(self.ax2vec))
                        if different:
                            if verbose:
                                #print diff
                                print('At AX0 = %0.2f and AX1 = %0.2f' % (ax0, ax1))
                                print('Bad ax2keys: ', ax2keys)
                            for key in self.ax2vec:
                                if key not in ax2keys:
                                    bad_axkeys[2].append(key)
                                    key_sets.append((ax0, ax1, key))


            self.get_max_files(dim3=dim3, bad_axkeys=bad_axkeys)

            for ks in key_sets:
                if not dim3:
                    self.agg_dict[bias][ks[0]][ks[1]] = []
                    for nfiles in range(self.max_files):
                        self.agg_dict[bias][ks[0]][ks[1]].append(FileData('', empty=True))
                else:
                    self.agg_dict[bias][ks[0]][ks[1]][ks[2]] = []
                    for nfiles in range(self.max_files):
                        self.agg_dict[bias][ks[0]][ks[1]][ks[2]].append(FileData('', empty=True))

                            




    def make_ax_arrs(self, dim3=False):
    
        if self.new_trap:
            attractor_travel = 500.0
        else:
            attractor_travel = 80.0

        if dim3:
            ### Assume separations are encoded in ax0 and heights in ax1
            seps = np.abs( self.p0_bead[0] - np.array(self.ax0vec) )
            ypos = self.p0_bead[1] - np.array(self.ax1vec)
            heights = self.p0_bead[2] - np.array(self.ax2vec) 
        else:
            seps = np.abs( self.p0_bead[0] - np.array(self.ax0vec) )
            heights = self.p0_bead[2] - np.array(self.ax1vec) 
        
        ### Sort the heights and separations and build a grid
        sort1 = np.argsort(seps)
        sort2 = np.argsort(heights)
        seps_sort = seps[sort1]
        heights_sort = heights[sort2]
        if dim3:
            sort3 = np.argsort(ypos)
            ypos_sort = ypos[sort3]

        outdic = {'seps': seps_sort, 'heights': heights_sort, \
                  'sep_sort': sort1, 'height_sort': sort2}
        if dim3:
            outdic['ypos'] = ypos_sort
            outdic['ypos_sort'] = sort3

        return outdic






    def average_resp_by_coordinate(self):
        '''Once data has been binned, average together the response and drive
           for every file at a given (height, sep)'''
    
        if self.new_trap:
            attractor_travel = 500.0
        else:
            attractor_travel = 80.0

        avg_dict = {}
        for bias in list(self.agg_dict.keys()):
            avg_dict[bias] = {}
            for ax0key in self.ax0vec:
                avg_dict[bias][ax0key] = {}
                for ax1key in self.ax1vec:
                    avg_dict[bias][ax0key][ax1key] = {}

        suff = 'Averaging response at each position'
        i = 0
        totlen = len(list(self.agg_dict.keys())) * len(self.ax0vec) * len(self.ax1vec)
        for bias, ax0, ax1 in itertools.product(list(self.agg_dict.keys()), self.ax0vec, self.ax1vec):
            i += 1
            newline=False
            if i == totlen:
                newline=True
            bu.progress_bar(i, totlen, newline=newline, suffix=suff)

            ### Pull out fileData() objects at the same position
            objs = self.agg_dict[bias][ax0][ax1]

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

                xpos += filfac * (self.p0_bead[0] + (attractor_travel - obj.ax0pos))
                height += filfac * (obj.ax1pos - self.p0_bead[2])

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
            avg_dict[bias][ax0][ax1]['drivevec'] = drivevec_avg
            avg_dict[bias][ax0][ax1]['posvec'] = posvec_avg - self.p0_bead[1]
            avg_dict[bias][ax0][ax1]['datfft'] = datfft_avg
            avg_dict[bias][ax0][ax1]['daterr'] = daterr_avg
            avg_dict[bias][ax0][ax1]['binned'] = binned_avg

        self.avg_dict = avg_dict

        print()









    def find_alpha_xyz_from_templates(self, plot=False, plot_basis=False, ncore=1, \
                                        alpha_scale=1e8, add_fake_data=False, \
                                        fake_alpha=1e13, plot_bad_alphas=False, \
                                        plot_templates=False, n_largest_harms=100):

        print('Finding alpha for each coordinate via an FFT template fitting algorithm...')
        
        if not self.gfuncs_class.grav_loaded:
            print("FAILED: Must load thoery data first!")
            try:
                self.gfuncs_class.reload_grav_funcs()
                print("UN-FAILED: Loaded dat therory dat!")
            except Exception:
                return

        alpha_xyz_dict = {}
        for bias in list(self.agg_dict.keys()):
            alpha_xyz_dict[bias] = {}
            for ax0key in self.ax0vec:
                alpha_xyz_dict[bias][ax0key] = {}
                for ax1key in self.ax1vec:
                    alpha_xyz_dict[bias][ax0key][ax1key] = []

        alpha_xyz_dict_2 = copy.deepcopy(alpha_xyz_dict)

        #derp_grid = np.zeros((len(self.ax0vec), len(self.ax1vec)))
        #derp_grid_2 = np.zeros_like(derp_grid)
        #derp_sep = np.zeros(len(self.ax0vec))
        #derp_height = np.zeros(len(self.ax1vec))

        ### Progress bar shit
        i = 0
        totlen = len(list(self.agg_dict.keys())) * len(self.ax0vec) * len(self.ax1vec)
        for bias, ax0, ax1 in itertools.product(list(self.agg_dict.keys()), self.ax0vec, self.ax1vec):

            #ax0ind = np.argmin(np.abs(np.array(self.ax0vec) - ax0))
            #ax1ind = np.argmin(np.abs(np.array(self.ax1vec) - ax1)

            file_data_objs = self.agg_dict[bias][ax0][ax1]
            nobjs = len(file_data_objs)

            ncomponents = 2 * np.min([len(self.ginds), n_largest_harms])
            nlambda = len(self.gfuncs_class.lambdas)

            p0_bead = self.p0_bead

            new_trap = self.new_trap

            gfunc_list = []
            for i in range(nobjs):
                gfunc_new = GravFuncs('', load=False)
                gfunc_new.__dict__.update(self.gfuncs_class.__dict__)
                gfunc_list.append(gfunc_new)

            arg_list = list(zip(file_data_objs, gfunc_list))

            # j = 0
            # totlen_2 = len(file_data_objs) * len(self.lambdas)
            # for objind, obj in enumerate(file_data_objs):
            def process_file_data(arg):  #obj):
                # file_start = time.time()
                obj, gfunc = arg

                p0_bead_new = obj.p0_bead

                # alpha_xyz_dict[bias][ax0][ax1].append([])

                # start = time.time()
                drivevec = obj.rebuild_drive()
                # stop = time.time()
                # print( 'Drive rebuild : {:0.4f}'.format(stop - start) )
                posvec = obj.posvec
                datfft = obj.datfft
                daterr = obj.daterr
                binned = obj.binned
                n_err = int(len(daterr[0]) / len(datfft[0]))

                out_arr = np.zeros((nlambda, n_err + 1, 3, ncomponents))
                out_arr_2 = np.zeros((nlambda, 2, 3))

                ## Loop over lambdas and do the template analysis for each value of lambda
                for lambind, yuklambda in enumerate(gfunc.lambdas):
    
                    out_subarr = np.zeros((n_err + 1, 3, ncomponents))
                    out_subarr_2 = np.zeros((2, 3))
                            
                    # start = time.time()
                    templates = gfunc.make_templates(posvec, drivevec, ax0, ax1, \
                                                        obj.ginds, p0_bead_new, obj.fsamp, \
                                                        single_lambda=True, \
                                                        single_lambind=lambind, \
                                                        new_trap=new_trap, \
                                                        plot=plot_templates, \
                                                        n_largest_harms=n_largest_harms)
                    # stop = time.time()
                    # print('Template time : {:0.4f}'.format(stop - start))

                    if plot and lambind == 0:
                        fig, axarr = plt.subplots(3,1,sharex=True,sharey=False,figsize=(10,8))

                    for resp in [0,1,2]:

                        ### Get the modified gravity fft template, with alpha = 1
                        yukfft = templates['yukffts'][lambind][resp]
                        yukbool = templates['yukbool'][lambind][resp]
                        erryukbool = yukbool.repeat(n_err, axis=0)

                        template_vec = np.concatenate((yukfft.real, yukfft.imag))
                        # print(template_vec)

                        c_datfft = datfft[resp][yukbool] #
                        if add_fake_data:
                            c_datfft = datfft[resp] + fake_alpha * yukfft
                        data_vec = np.concatenate((c_datfft.real, c_datfft.imag))

                        c_daterr = daterr[resp][erryukbool]
                        err_vec = np.concatenate((c_daterr.real, c_daterr.imag))

                        alphaguess = np.inner(data_vec, template_vec) / \
                                                np.inner(template_vec, template_vec)
                        alphaguess /= alpha_scale

                        # def alphacost(alpha, err_ind=0):
                        #     ndof = ncomponents - 1
                        #     num = (data_vec - alpha*alpha_scale*template_vec)**2
                        #     denom = (err_vec[err_ind::n_err])**2
                        #     return (1.0 / ndof) * np.sum(num / denom)

                        # vals = []
                        # for err_ind in range(n_err):
                        #     fitcost = lambda alpha: alphacost(alpha, err_ind=err_ind)

                        #     m = Minuit(fitcost,
                        #                alpha = alphaguess, # set start parameter
                        #                #fix_param = "True", # you can also fix it
                        #                # limit_param = (0.0, 10000.0),
                        #                errordef = 1,
                        #                print_level = 0, 
                        #                pedantic=False)
                        #     m.migrad(ncall=500000)
                        #     # m.draw_mnprofile('alpha')
                        #     # plt.show()
                        #     vals.append(m.values['alpha'])

                        # out_subarr_2[0,resp] = np.mean(vals)
                        # out_subarr_2[1,resp] = np.std(vals)
                        # out_subarr_2[:,resp] *= alpha_scale
                        # minos = m.minos()

                        ### Compute an 2*Nharmonic-dimensional basis for the real and 
                        ### imaginary components of our template signal yukfft, where
                        ### the template itself will be one of the orthogonal basis vectors
                        bases = make_basis_from_template_vec(template_vec)

                        ### The SVD decomposition above should produce an orthonormal basis
                        ### but this function is included to demonstrate that
                        # ortho_basis = gram_schmidt(bases['real_basis'], plot=plot_basis)['orthogonal']
                        ortho_basis = bases['real_basis']

                        # if resp == 2:
                        #     print(gfunc.lambdas[lambind])
                        #     print('Template : ', template_vec)
                        #     print('   Ortho : ', ortho_basis[0])
                        #     print('    Data : ', data_vec)
                        #     input()

                        ### Loop over our orthogonal basis vectors and compute the inner 
                        ### product of the data and the basis vector
                        for k in range(len(ortho_basis)):
                            out_subarr[0][resp][k] = np.inner( ortho_basis[k], data_vec)
                            for err_ind in range(n_err):
                                out_subarr[err_ind+1][resp][k] = \
                                        np.inner( ortho_basis[k], err_vec[err_ind::n_err])

                        ### Normalize the projection amplitudes to units of alpha
                        template_norm = np.inner(template_vec, template_vec)
                        # if resp == 2:
                        #     projection = out_subarr[0][resp][0] / template_norm
                        #     print('Projection : ', projection, np.log10(np.abs(projection)))
                        #     input()

                        out_subarr[:,resp,:] *= (1.0 / template_norm)
                        alphaz = out_subarr[0,2,0]
                        if plot_bad_alphas:
                            if resp == 2 and lambind == 0:
                                if np.abs(alphaz) > 10.0**10:
                                    obj.reload_datafile()
                                    print('bad file: {:s}'.format(obj.fname))
                                    fig, axarr = plt.subplots(3,1, sharex=True)
                                    for i in range(3):
                                        axarr[i].plot(obj.df.pos_data_3[i] - np.mean(obj.df.pos_data_3[i]))
                                    plt.show()


                        if plot and lambind == 0:
                            axarr[resp].errorbar(list(range(len(out_subarr[0][resp]))), \
                                                 out_subarr[0][resp], \
                                                 np.abs(np.mean(out_subarr[1:,resp,:], axis=0)), \
                                                 fmt='o')
                            axarr[resp].set_ylabel('Projection [$\\alpha$]')
                            if resp == 2:
                                axarr[resp].set_xlabel('Basis Vector Index')

                    if plot and lambind == 0:
                        plt.show()

                    out_arr[lambind] += out_subarr
                    out_arr_2[lambind] += out_subarr_2

                return (out_arr, out_arr_2)

            results = Parallel(n_jobs=ncore)(delayed(process_file_data)(arg) \
                                                    for arg in tqdm(arg_list))

            alpha_xyz_dict[bias][ax0][ax1] = np.array([i for i, j in results])
            alpha_xyz_dict_2[bias][ax0][ax1] = np.array([j for i, j in results])


        print('Done!')   
        self.alpha_xyz_dict = alpha_xyz_dict
        self.alpha_xyz_dict_2 = alpha_xyz_dict_2





    
    def plot_alpha_xyz_dict(self, resp=0, k=0, nobjs=1e9, lambind=0):
    
        if self.new_trap:
            attractor_travel = 500.0
        else:
            attractor_travel = 80.0

        self.get_max_files()
        ngrids = int( np.min([self.max_files, nobjs]) )

        ### Assume separations are encoded in ax0 and heights in ax1
        seps = attractor_travel + self.p0_bead[0] - np.array(self.ax0vec)
        heights = self.p0_bead[2] - np.array(self.ax1vec) 
        
        ### Sort the heights and separations and build a grid
        sort1 = np.argsort(seps)
        sort2 = np.argsort(heights)
        seps_sort = seps[sort1]
        heights_sort = heights[sort2]
        seps_g, heights_g = np.meshgrid(seps_sort, heights_sort, indexing='ij')

        bias = list(self.agg_dict.keys())[0]

        grids = []
        for objind in range(ngrids):
            grids.append([[[] for i in range(len(heights))] for j in range(len(seps))])

            for ax0ind, ax0pos in enumerate(self.ax0vec):
                for ax1ind, ax1pos in enumerate(self.ax1vec):

                    grids[objind][ax0ind][ax1ind] = \
                            self.alpha_xyz_dict[bias][ax0pos][ax1pos][objind][lambind][0][resp][k]
                
        grids = np.array(grids)

        for objind in range(ngrids):
            grids[objind][:,:] = grids[objind][sort1,:]
            grids[objind][:,:] = grids[objind][:,sort2]


        lambda_str_1 = ', lambda: %0.4e' % self.lambdas[lambind]
        title = 'resp: ' + str(resp) + lambda_str_1 + \
                ', basis index: ' + str(k)

        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        for i in range(ngrids):
            ax.scatter(seps_g, heights_g, grids[i], color='C0')
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel('X-separation [um]')
        ax.set_ylabel('Z-position [um]')
        ax.set_zlabel('Alpha %s [arb]' % ax_dict[resp])

        plt.show()    




    def plot_force_plane(self, resp=0, nobjs=1e9, ax_dict={0:'X', 1:'Y', 2:'Z'}, \
                         fig_ind=1, show=True):

        self.get_max_files()
        ngrids = int( np.min([self.max_files, nobjs]) )

        ### Assume separations are encoded in ax0 and heights in ax1
        seps = self.p0_bead[0] - np.array(self.ax0vec)
        heights = self.p0_bead[2] - np.array(self.ax1vec) 
        
        ### Sort the heights and separations and build a grid
        sort1 = np.argsort(seps)
        sort2 = np.argsort(heights)
        seps_sort = seps[sort1]
        heights_sort = heights[sort2]
        seps_g, heights_g = np.meshgrid(seps_sort, heights_sort, indexing='ij')

        bias = self.biasvec[0]

        grids = []
        noise_grids = []
        for objind in range(ngrids):
            grids.append([[[] for i in range(len(heights))] for j in range(len(seps))])
            noise_grids.append([[[] for i in range(len(heights))] for j in range(len(seps))])

            for ax0ind, ax0pos in enumerate(self.ax0vec):
                for ax1ind, ax1pos in enumerate(self.ax1vec):
                    
                    fd_obj = self.agg_dict[bias][ax0pos][ax1pos][objind]
                    fd_obj.get_background_rms()

                    grids[objind][ax0ind][ax1ind] = fd_obj.background_rms['background'][resp]
                    noise_grids[objind][ax0ind][ax1ind] = fd_obj.background_rms['noise'][resp]
                
                    #if grids[objind][ax0ind][ax1ind] > 2e-14:

        grids = np.array(grids)
        noise_grids = np.array(noise_grids)

        for objind in range(ngrids):
            grids[objind][:,:] = grids[objind][sort1,:]
            grids[objind][:,:] = grids[objind][:,sort2]
            noise_grids[objind][:,:] = noise_grids[objind][sort1,:]
            noise_grids[objind][:,:] = noise_grids[objind][:,sort2]

        fig = plt.figure(fig_ind)
        ax = fig.gca(projection='3d')
        for i in range(ngrids):
            ax.scatter(seps_g, heights_g, grids[i], color='C0')
            ax.scatter(seps_g, heights_g, noise_grids[i], color='C1')
        ax.legend()
        ax.set_xlabel('X-separation [um]')
        ax.set_ylabel('Z-position [um]')
        ax.set_zlabel('%s RMS [N]' % ax_dict[resp])
        # ax.xaxis._axinfo['label']['space_factor'] = 2.0
        # ax.xaxis.labelpad = 30
        # ax.yaxis._axinfo['label']['space_factor'] = 2.0
        # ax.yaxis.labelpad = 30
        # ax.zaxis._axinfo['label']['space_factor'] = 2.0
        # ax.zaxis.labelpad = 30
        fig.tight_layout()

        if show:
            plt.show()






    def get_vector_force_plane(self, plot_resp=(0,1), nobjs=1e9, ax_dict={0:'X', 1:'Y', 2:'Z'}, \
                               fig_ind=1, plot=True, show=True, sign=[1.0,1.0,1.0], \
                               dim3=False, keyscale=1.0e-14):
        

        self.get_max_files(dim3=dim3)
        ngrids = int( np.min([self.max_files, nobjs]) )

        pos_dict = self.make_ax_arrs(dim3=dim3)
        seps_sort = pos_dict['seps']
        heights_sort = pos_dict['heights']
        sort1 = pos_dict['sep_sort']
        sort2 = pos_dict['height_sort']
        if dim3:
            ypos_sort = pos_dict['ypos']
            sort3 = pos_dict['ypos_sort']
            seps_g, ypos_g, heights_g = np.meshgrid(seps_sort, heights_sort, ypos_sort, \
                                                    indexing='ij')
        else:
            seps_g, heights_g = np.meshgrid(seps_sort, heights_sort, indexing='ij')

        bias = self.biasvec[0]

        pos_grids = []
        err_grids = []

        noise_grids = []
        if dim3:
            drive_grid = np.zeros((len(seps_sort), len(ypos_sort), len(heights_sort)))
        else:
            drive_grid = np.zeros((len(seps_sort), len(heights_sort)))
        for ax in [0,1,2]:
            if dim3:
                pos_grids.append(np.zeros((len(seps_sort), len(ypos_sort), len(heights_sort))))
                err_grids.append(np.zeros((len(seps_sort), len(ypos_sort), len(heights_sort))))
                noise_grids.append(np.zeros((len(seps_sort), len(ypos_sort), len(heights_sort))))
            else:
                pos_grids.append(np.zeros((len(seps_sort), len(heights_sort))))
                err_grids.append(np.zeros((len(seps_sort), len(heights_sort))))
                noise_grids.append(np.zeros((len(seps_sort), len(heights_sort))))


        for objind in range(ngrids):

            for ax0ind, ax0pos in enumerate(self.ax0vec):
                for ax1ind, ax1pos in enumerate(self.ax1vec):
                    if dim3:
                        for ax2ind, ax2pos in enumerate(self.ax2vec):
                            fd_obj = self.agg_dict[bias][ax0pos][ax1pos][ax2pos][objind]
                            if fd_obj.empty:
                                continue

                            df = fd_obj.fsamp / fd_obj.nsamp
                            fft_to_amp = bu.fft_norm(fd_obj.nsamp, fd_obj.fsamp) * np.sqrt(2 * df)
                                
                            drive_grid[ax0ind,ax1ind,ax2ind] += \
                                            np.abs(fd_obj.drivefft[0]) * fft_to_amp

                            drive_phasor = fd_obj.drivefft[0]
                            drive_amp = np.abs(drive_phasor)
                            unit_drive_phasor = drive_phasor / drive_amp 

                            fd_obj.get_background_rms()
                            for resp in [0,1,2]:
                                pos_grids[resp][ax0ind,ax1ind,ax2ind] += \
                                            (fd_obj.diagdatfft[resp][0] / unit_drive_phasor).real * \
                                            fft_to_amp * sign[resp]

                                err_grids[resp][ax0ind,ax1ind,ax2ind] += \
                                            np.abs(fd_obj.diagdaterr[resp][0]) * \
                                            fft_to_amp * sign[resp]

                                noise_grids[resp][ax0ind,ax1ind,ax2ind] += \
                                            (fd_obj.noisefft[resp] / unit_drive_phasor).real * \
                                            fft_to_amp * sign[resp]
                    else:
                        fd_obj = self.agg_dict[bias][ax0pos][ax1pos][objind]
                        if fd_obj.empty:
                            continue

                        df = fd_obj.fsamp / fd_obj.nsamp
                        fft_to_amp = bu.fft_norm(fd_obj.nsamp, fd_obj.fsamp) * np.sqrt(2 * df)
                        
                        drive_grid[ax0ind,ax1ind] += \
                                        np.abs(fd_obj.drivefft[0]) * fft_to_amp

                        drive_phasor = fd_obj.drivefft[0]
                        drive_amp = np.abs(drive_phasor)
                        unit_drive_phasor = drive_phasor / drive_amp 

                        fd_obj.get_background_rms()
                        for resp in [0,1,2]:
                            pos_grids[resp][ax0ind,ax1ind] += \
                                        (fd_obj.diagdatfft[resp][0] / unit_drive_phasor).real * \
                                        fft_to_amp * sign[resp]
                            err_grids[resp][ax0ind,ax1ind] += \
                                        np.abs(fd_obj.diagdaterr[resp][0]) * \
                                        fft_to_amp * sign[resp]

                            noise_grids[resp][ax0ind,ax1ind] += \
                                        (fd_obj.noisefft[resp] / unit_drive_phasor).real * \
                                        fft_to_amp * sign[resp]

        if dim3:
            drive_grid[:,:,:] = drive_grid[sort1,:,:] 
            drive_grid[:,:,:] = drive_grid[:,sort2,:] 
            drive_grid[:,:,:] = drive_grid[:,:,sort3] 

        else:
            drive_grid[:,:] = drive_grid[sort1,:]
            drive_grid[:,:] = drive_grid[:,sort2]

        for resp in [0,1,2]:
            pos_grids[resp] *= (1.0 / ngrids)
            if dim3:
                pos_grids[resp][:,:,:] = pos_grids[resp][sort1,:,:] 
                pos_grids[resp][:,:,:] = pos_grids[resp][:,sort2,:] 
                pos_grids[resp][:,:,:] = pos_grids[resp][:,:,sort3] 

                err_grids[resp][:,:,:] = err_grids[resp][sort1,:,:] 
                err_grids[resp][:,:,:] = err_grids[resp][:,sort2,:] 
                err_grids[resp][:,:,:] = err_grids[resp][:,:,sort3] 

                noise_grids[resp][:,:,:] = noise_grids[resp][sort1,:,:] 
                noise_grids[resp][:,:,:] = noise_grids[resp][:,sort2,:] 
                noise_grids[resp][:,:,:] = noise_grids[resp][:,:,sort3]

            else:
                pos_grids[resp][:,:] = pos_grids[resp][sort1,:]
                pos_grids[resp][:,:] = pos_grids[resp][:,sort2]
                
                err_grids[resp][:,:] = err_grids[resp][sort1,:]
                err_grids[resp][:,:] = err_grids[resp][:,sort2]
                
                noise_grids[resp][:,:] = noise_grids[resp][sort1,:]
                noise_grids[resp][:,:] = noise_grids[resp][:,sort2]


        scale_pow = int(np.log10(keyscale))
        scale = keyscale * 4
        
        if plot and not dim3:

            fig = plt.figure(fig_ind)
            ax = fig.add_subplot(111)
        
            qdat = ax.quiver(seps_g, heights_g, pos_grids[plot_resp[0]], pos_grids[plot_resp[1]], \
                             color='k', pivot='mid', label='Force', scale=scale)
            qerr = ax.quiver(seps_g, heights_g, err_grids[plot_resp[0]], err_grids[plot_resp[1]], \
                             color='r', pivot='mid', label='Error', scale=scale)
            ax.set_xlabel('Separation [um]')
            ax.set_ylabel('Height [um]')

        if plot and dim3:
        
            for new_ind in range(len(sort1)):
                fig = plt.figure(fig_ind+new_ind)
                ax = fig.add_subplot(111)

                qdat = ax.quiver(seps_g[:,new_ind,:], heights_g[:,new_ind,:], \
                             pos_grids[plot_resp[0]][:,new_ind,:], \
                             pos_grids[plot_resp[1]][:,new_ind,:], \
                             color='k', pivot='mid', label='Force', scale=scale)

                qerr = ax.quiver(seps_g[:,new_ind,:], heights_g[:,new_ind,:], \
                             err_grids[plot_resp[0]][:,new_ind,:], \
                             err_grids[plot_resp[1]][:,new_ind,:], \
                             color='r', pivot='mid', label='Error', scale=scale)
                    
                ax.set_xlabel('Separation [um]')
                ax.set_ylabel('Height [um]')

                plt.title('Ypos %0.2f' % ypos_sort[new_ind])

                if show:
                    plt.show()
                else:
                    fig.clear()
                    plt.close(fig)

        if plot and show:
            ax.quiverkey(qdat, X=0.3, Y=1.05, U=keyscale, \
                         label='$10^{%i}~$N Force' % scale_pow, labelpos='N')
            ax.quiverkey(qerr, X=0.7, Y=1.05, U=keyscale, \
                         label='$10^{%i}~$N Force' % scale_pow, labelpos='N')


        outdict = {0: pos_grids[0], 1: pos_grids[1], 2: pos_grids[2], \
                   'xerr': err_grids[0], 'yerr': err_grids[1], 'zerr': err_grids[2], \
                   'drive': drive_grid, 'xnoise': noise_grids[0], \
                   'ynoise': noise_grids[1], 'znoise': noise_grids[2]}

        if plot and show:
            plt.show()
        if plot and not show:
            outdict['fig'] = fig
            outdict['ax'] = ax

        return outdict




    def fit_alpha_xyz_onepos_simple(self, resp=[0,1,2], confidence_level=0.95, \
                                    verbose=False, last_file=-1, plot=False, \
                                    show=True, plot_color='C0', plot_label='', \
                                    plot_alpha=1.0, sigma_to_profile=3.0):

        if not self.gfuncs_class.grav_loaded:
            try:
                self.gfuncs_class.reload_grav_funcs()
            except Exception:
                print('No grav funcs... Tried to reload but no filename')

        self.alpha_best_fit = []
        self.alpha_95cl = []

        self.alpha_best_fit_null = []
        self.alpha_95cl_null = []

        self.signal_chisq_prof = []
        self.sideband_chisq_prof = []

        ### Assume separations are encoded in ax0 and heights in ax1
        ax0 = self.ax0vec[0]
        ax1 = self.ax1vec[0]
        sep = self.p0_bead[0] - ax0
        height = self.p0_bead[2] - ax1

        if verbose:
            print('Computing limit for: sep = {:0.1f} um, height = {:0.1f}'\
                    .format(sep, height) )

        profiles = []
        for biasind, bias in enumerate(self.biasvec):
            ### Doesn't actually handle different biases correctly, although the
            ### data is structured such that if different biases are present
            ### they will be in distinct datasets
            profiles.append([])

            alpha_arr = self.alpha_xyz_dict[bias][ax0][ax1]
            alpha_arr_2 = self.alpha_xyz_dict_2[bias][ax0][ax1]

            n_sideband = alpha_arr.shape[2] - 1

            for lambind, yuklambda in enumerate(self.gfuncs_class.lambdas):

                ### Take the signal vector projection (last index 0) for the 
                ### desired response axis at the position defined
                for axind, ax in enumerate(resp):
                    new_dat = alpha_arr[:,lambind,0,ax,0]
                    new_dat2 = alpha_arr_2[:,lambind,0,ax]

                    new_sideband = alpha_arr[:,lambind,1,ax,0]
                    new_sideband2 = alpha_arr_2[:,lambind,1,ax]

                    new_sideband_long = np.empty((n_sideband*new_sideband.size,), \
                                                 dtype=new_sideband.dtype)
                    for i in range(n_sideband):
                        new_sideband_long[i::n_sideband] = alpha_arr[:,lambind,i+1,ax,0]

                    if axind == 0:
                        dat = new_dat
                        dat2 = new_dat2
                        sideband = new_sideband
                        sideband2 = new_sideband2
                        sideband_long = new_sideband_long
                    else:
                        dat = np.concatenate((dat, new_dat))
                        dat2 = np.concatenate((dat2, new_dat2))
                        sideband = np.concatenate((sideband, new_sideband))
                        sideband2 = np.concatenate((sideband2, new_sideband2))
                        sideband_long = np.concatenate((sideband_long, new_sideband_long))


                # ### Subselect to avoid crazy data
                # inds = np.abs(dat) < 1e-4 * np.abs(np.max(dat))
                # sideband_inds = np.abs(sideband_long) < 1e-4 * np.abs(np.max(sideband_long))

                # good_dat = dat[np.abs(dat) < 10.0 * np.std(dat[inds])]
                # good_dat_2 = dat2[np.abs(dat2) < 100 * np.std(dat2)]
                # good_sideband = sideband_long[np.abs(sideband_long) \
                #                         < 100 * np.std(sideband_long[sideband_inds])]

                inds = dat == dat
                sideband_inds = sideband_long == sideband_long

                good_dat = np.copy(dat)
                good_dat_2 = np.copy(dat2)
                good_sideband = np.copy(sideband_long)

                # if lambind == 0.0:
                #     # print(sideband2)
                #     # plt.hist(sideband2)

                #     plt.figure()
                #     datcdf = bu.ECDF(good_dat)
                #     datcdf2 = bu.ECDF(good_dat_2)
                #     # sidebandcdf = bu.ECDF(sideband)
                #     # sidebandcdf2 = bu.ECDF(sideband2)
                #     xarr = np.linspace(-1.0e9, 1.0e9, 100)
                #     plt.plot(xarr, datcdf(xarr), color='C0')
                #     # plt.plot(xarr, sidebandcdf(xarr), color='C0', ls='--')
                #     plt.plot(xarr, datcdf2(xarr), color='C1')
                #     # plt.plot(xarr, sidebandcdf2(xarr), color='C1', ls='--')

                #     plt.figure()
                #     # plt.errorbar(range(len(dat2)), dat2, yerr=sideband2)
                #     plt.plot(range(len(good_dat)), good_dat, zorder=99)
                #     # plt.plot(np.arange(len(good_sideband))*(1.0/n_sideband) - 0.5, good_sideband)

                #     plt.show()

                if last_file != -1:
                    good_dat = good_dat[:int(last_file)]
                    good_sideband = good_sideband[:int(last_file*n_sideband)]

                N = len(good_dat)
                M = len(good_sideband)

                alpha_scale = np.std(good_dat)

                fit_dat = good_dat * (1.0 / alpha_scale)
                fit_sideband = good_sideband * (1.0 / alpha_scale)

                def NLL_dat(mu_dat, sigma):
                    dat_nll = N * np.log(np.sqrt(2 * np.pi) * sigma) + \
                                (1.0 / (2.0 * sigma**2)) * np.sum( (fit_dat - mu_dat)**2 )
                    return dat_nll

                def NLL_sideband(mu_sideband, sigma):
                    sideband_nll = M * np.log(np.sqrt(2 * np.pi) * sigma) + \
                                (1.0 / (2.0 * sigma**2)) * np.sum( (fit_sideband - mu_sideband)**2 )
                    return sideband_nll

                def NLL(mu_dat, mu_sideband, sigma):
                    return NLL_dat(mu_dat, sigma) + NLL_sideband(mu_sideband, sigma)

                # print(N, end = ', ')
                sys.stdout.flush()
                sigma_guess = np.std(fit_dat)
                m_null = Minuit(NLL,
                                mu_dat = 0, # set start parameter
                                fix_mu_dat = 'True', # you can also fix it
                                #limit_mu_dat = (0.0, 10000.0),
                                mu_sideband = 0, # set start parameter
                                # fix_mu_sideband = 'True', 
                                #limit_mu_sideband = (0.0, 10000.0),
                                sigma = sigma_guess, # set start parameter
                                #fix_sigma = "True", 
                                limit_sigma = (1e-3 * sigma_guess, 1000.0 * sigma_guess),
                                errordef = 1,
                                print_level = 0, 
                                pedantic=False)
                m_null.migrad(ncall=500000)

                NLLR = lambda mu_dat, mu_sideband, sigma: \
                                2.0 * (m_null.fval - NLL(mu_dat, mu_sideband, sigma))

                m = Minuit(NLL,
                           mu_dat = 0, # set start parameter
                           #fix_mu_dat = "True", # you can also fix it
                           #limit_mu_dat = (0.0, 10000.0),
                           mu_sideband = 0, # set start parameter
                           #fix_mu_sideband = "True", 
                           #limit_mu_sideband = (0.0, 10000.0),
                           sigma = sigma_guess, # set start parameter
                           #fix_sigma = "True", 
                           limit_sigma = (1e-3 * sigma_guess, 1000.0 * sigma_guess),
                           errordef = 1,
                           print_level = 0, 
                           pedantic=False)
                m.migrad(ncall=500000)

                try:
                    minos_null = m_null.minos()
                    minos = m.minos()

                    alpha_best = minos['mu_dat']['min']
                    alpha_lower = minos['mu_dat']['lower']
                    alpha_upper = minos['mu_dat']['upper']
                    mu_dat_arr = np.linspace(alpha_best + sigma_to_profile*alpha_lower, \
                                             alpha_best + sigma_to_profile*alpha_upper, 31)

                    alphanull_best = minos_null['mu_sideband']['min']
                    alphanull_lower = minos_null['mu_sideband']['lower']
                    alphanull_upper = minos_null['mu_sideband']['upper']
                    mu_null_arr = np.linspace(alphanull_best + sigma_to_profile*alphanull_lower, \
                                             alphanull_best + sigma_to_profile*alphanull_upper, 31)

                    if verbose:
                        ### Rough estimate of goodness of fit
                        print('Chi-squared goodness of fit for...')
                        print('       in-band null hypothesis: {:0.2f}'\
                                      .format(2.0 * NLL_dat(0, minos['sigma']['min']) / N))

                        print('   out-of-band null hypothesis: {:0.2f}'\
                                      .format(2.0 * NLL_sideband(0, minos['sigma']['min']) / M))

                        print('                    full model: {:0.2f}'\
                                      .format(2.0 * NLL(alpha_best, minos['mu_sideband']['min'], \
                                                        minos['sigma']['min']) / (N + M)))

                        print()

                    chi_sq = np.zeros_like(mu_dat_arr)
                    for ind, test_mu in enumerate(mu_dat_arr):
                        m = Minuit(NLL,
                                   mu_dat = test_mu, # set start parameter
                                   fix_mu_dat = 'True', # you can also fix it
                                   #limit_mu_dat = (0.0, 10000.0),
                                   mu_sideband = 0, # set start parameter
                                   #fix_mu_sideband = "True", 
                                   #limit_mu_sideband = (0.0, 10000.0),
                                   sigma = sigma_guess, # set start parameter
                                   #fix_sigma = "True", 
                                   limit_sigma = (0.0, 1000.0 * sigma_guess),
                                   errordef = 1,
                                   print_level = 0, 
                                   pedantic=False)
                        m.migrad(ncall=500000)

                        chi_sq[ind] = m.fval


                    chi_sq_sideband = np.zeros_like(mu_null_arr)
                    for ind, test_mu in enumerate(mu_null_arr):
                        m_null = Minuit(NLL,
                                   mu_dat = 0, # set start parameter
                                   fix_mu_dat = 'True', # you can also fix it
                                   #limit_mu_dat = (0.0, 10000.0),
                                   mu_sideband = test_mu, # set start parameter
                                   fix_mu_sideband = 'True', 
                                   #limit_mu_sideband = (0.0, 10000.0),
                                   sigma = sigma_guess, # set start parameter
                                   #fix_sigma = "True", 
                                   limit_sigma = (0.0, 1000.0 * sigma_guess),
                                   errordef = 1,
                                   print_level = 0, 
                                   pedantic=False)
                        m_null.migrad(ncall=500000)

                        chi_sq_sideband[ind] = m_null.fval

                    test_alphas = mu_dat_arr * alpha_scale
                    test_alphas_sideband = mu_null_arr * alpha_scale

                    ### Subtract off the null hypothesis
                    chi_sq -= np.min(chi_sq)
                    chi_sq_sideband -= np.min(chi_sq_sideband)

                    if plot and lambind == 0.0:
                        # plt.plot(test_alphas, chi_sq, color=plot_color, label=plot_label,
                        #             alpha=plot_alpha)
                        plt.plot(test_alphas_sideband, chi_sq_sideband, color=plot_color,
                                    alpha=plot_alpha)
                        if show:
                            plt.xlabel('Alpha [Arb.]')
                            plt.ylabel('$\\Delta \\chi^2$ [Arb.]')
                            plt.ylim(0, sigma_to_profile**2 - 1)
                            plt.legend()
                            plt.tight_layout()
                            plt.show()

                    profiles[biasind].append([test_alphas, chi_sq, \
                                              test_alphas_sideband, chi_sq_sideband])

                    ### Fit the NLL to a parabola and extract the minimum and interval
                    ### corresponding to the requested confidence level
                    popt, pcov = opti.curve_fit(parabola, mu_dat_arr, chi_sq, \
                                                p0=[np.max(chi_sq)/np.max(mu_dat_arr)**2, 0, 0])
                    soln = solve_parabola(chi2dist.ppf(confidence_level), popt)

                    popt_null, pcov_null = opti.curve_fit(parabola, mu_null_arr, chi_sq_sideband, \
                                                p0=[np.max(chi_sq_sideband)/np.max(mu_null_arr)**2, 0, 0])
                    soln_null = solve_parabola(chi2dist.ppf(confidence_level), popt_null)

                    ### "Best fit" is like the sensitivity (should be consistent with 0
                    ### if we understand backgrounds)
                    sensitivity = (-1.0 * popt[1] / (2.0 * popt[0]))
                    sensitivity_null = (-1.0 * popt_null[1] / (2.0 * popt_null[0]))

                    ### Limit is derived from the larger of the ends of the confidence interval
                    limit = np.max(np.abs(np.array(soln) - sensitivity))
                    limit_null = np.max(np.abs(np.array(soln_null) - sensitivity_null))

                    self.alpha_best_fit.append(alpha_scale * sensitivity)
                    self.alpha_95cl.append(alpha_scale * limit)

                    self.alpha_best_fit_null.append(alpha_scale * sensitivity_null)
                    self.alpha_95cl_null.append(alpha_scale * limit_null)

                except Exception:
                    try:
                        self.alpha_best_fit.append(self.alpha_best_fit[-1])
                        self.alpha_95cl.append(self.alpha_95cl[-1])
                    except Exception:
                        self.alpha_best_fit.append(alpha_scale)
                        self.alpha_95cl.append(alpha_scale)

        return profiles





    def fit_alpha_xyz_onepos(self, resp=0):

        if self.new_trap:
            attractor_travel = 500.0
        else:
            attractor_travel = 80.0

        if not self.gfuncs_class.grav_loaded:
            try:
                self.gfuncs_class.reload_grav_funcs()
            except Exception:
                print('No grav funcs... Tried to reload but no filename')

        # self.alpha_best_fit = []
        # self.alpha_95cl = []

        ### Assume separations are encoded in ax0 and heights in ax1
        ax0 = self.ax0vec[0]
        ax1 = self.ax1vec[0]
        sep = self.p0_bead[0] - ax0
        height = self.p0_bead[2] - ax1


        print('Computing limit for: sep = {:0.1f} um, height = {:0.1f}'\
                .format(sep, height) )

        for bias in self.biasvec:
            ### Doesn't actually handle different biases correctly, although the
            ### data is structured such that if different biases are present
            ### they will be in distinct datasets

            alpha_arr = self.alpha_xyz_dict[bias][ax0][ax1]

            fig, axarr = plt.subplots(2,1, sharex=True, figsize=(10,8))
            axarr[0].hist(alpha_arr[:,50,0,resp,0], bins=50)
            axarr[1].hist(alpha_arr[:,0,1:,resp,0].flatten(), bins=50)
            fig.tight_layout()
            plt.show()

            # for lambind, yuklambda in enumerate(self.gfuncs_class.lambdas):

            #     ### Take the signal vector projection (last index 0) for the 
            #     ### desired response axis at the position defined
            #     # dat = self.alpha_xyz_dict[bias][ax0][ax1][:][lambind][0][resp][0]
            #     dat = alpha_arr[:,lambind,0,resp,0]
            #     # errs = self.alpha_xyz_dict[bias][ax0][ax1][:][lambind][1][resp][0]
            #     errs = alpha_arr[:,lambind,1,resp,0]

            #     # dat_mean = np.average(dat, weights=errs)
            #     dat_mean = np.mean(dat)
            #     dat_std = np.std(dat)

            #     self.alpha_best_fit.append(val1)
            #     self.alpha_95cl.append(val2)








    def fit_alpha_xyz_vs_alldim(self, weight_planar=True, plot=False, save_hists=False, \
                                hist_info={'date': 0, 'prefix': ''}):

        if not self.gfuncs_class.grav_loaded:
            try:
                self.gfuncs_class.reload_grav_funcs()
            except Exception:
                print('No grav funcs... Tried to reload but no filename')

        alpha_xyz_best_fit = [[[[] for k in range(2 * len(self.ginds))] \
                                    for resp in [0,1,2]] \
                                   for yuklambda in self.lambdas]

        ### Assume separations are encoded in ax0 and heights in ax1
        seps = self.p0_bead[0] - np.array(self.ax0vec)
        heights = self.p0_bead[2] - np.array(self.ax1vec) 
        
        ### Sort the heights and separations and build a grid
        sort1 = np.argsort(seps)
        sort2 = np.argsort(heights)
        seps_sort = seps[sort1]
        heights_sort = heights[sort2]
        seps_g, heights_g = np.meshgrid(seps_sort, heights_sort, indexing='ij')

        ### Progress bar shit
        ind = 0
        totlen = len(self.lambdas) * (2 * len(self.ginds)) * 3

        self.get_max_files()

        for bias in self.biasvec:
            ### Doesn't actually handle different biases correctly, although the
            ### data is structured such that if different biases are present
            ### they will be in distinct datasets

            for lambind, yuklambda in enumerate(self.lambdas):
                ### For each value of lambda, collate the projections of the data
                ### vector on the template + orthogonal basis, as functions of the 
                ### response axis and height/separation
                for resp in [0,1,2]:

                    grids = []
                    errs = []
                    for k in range(2 * len(self.ginds)):
                        grids.append([[[] for i in range(len(heights))] for j in range(len(seps))])
                        errs.append([[[] for i in range(len(heights))] for j in range(len(seps))])

                    for ax0ind, ax0pos in enumerate(self.ax0vec):

                        ### Loop over all files at each separation and collect
                        ### the value of alpha for the current value of yuklambda
                        for ax1ind, ax1pos in enumerate(self.ax1vec):

                            tempdat = self.alpha_xyz_dict[bias][ax0pos][ax1pos]

                            for fileind, filedat in enumerate(tempdat):
                                
                                #hypersphere_too_big = \
                                #     cut_by_hypersphere(tempdat[fileind][lambind][0][resp][:], \
                                #                        cut_std_dev=1)

                                if fileind + 1 > self.max_files:
                                    continue
                                #if hypersphere_too_big[fileind]:
                                #    continue
                                    
                                for k in range(2 * len(self.ginds)):

                                    grids[k][ax0ind][ax1ind].append(tempdat[fileind][lambind][0][resp][k])
                                    errs[k][ax0ind][ax1ind].append(tempdat[fileind][lambind][1][resp][k])

                    grids = np.array(grids)
                    errs = np.array(errs)

                    new_grids = np.moveaxis(grids, -1, 1)
                    new_errs = np.moveaxis(errs, -1, 1)


                    ### Since the data dictionary was indexed by cantilever settings 
                    ### rather than actual bead positions, we have to sort the data
                    ### to match the sorted separations and heights. It's probably
                    ### dumb to make entirely new arrays here, but I couldn't figure
                    ### out this stupid indexing/transposing bug when trying to sort
                    ### on these axes in-place
                    new_grids1 = grids[:,sort1,:,:]
                    new_grids2 = new_grids1[:,:,sort2,:]

                    new_errs1 = errs[:,sort1,:,:]
                    new_errs2 = new_errs1[:,:,sort2,:]

                    grids = new_grids2
                    errs = new_errs2

                    ### Loop over template vectors and fit each projection to a plane
                    for k in range(2 * len(self.ginds)):

                        ### Condition data to be order unity so that a generic least-squared
                        ### cost function and scipy optimization routine can be used without
                        ### specification of non-standard tolerances etc

                        scale_fac = np.std(grids[k]) / 100
                        grids_sc = grids[k] * (1.0 / scale_fac)
                        if weight_planar:
                            errs_sc = errs[k] * (1.0 / scale_fac)
                        else:
                            errs_sc = np.ones_like(grids_sc)

                        bu.progress_bar(ind, totlen, suffix='Fitting planes... ')
                        #print ind,
                        ind += 1
    
                        startplanar = time.time()

                        ### Defined a function to minimize via least-squared optimization
                        def cost_function(params, Ndof=True):
                            a, b, c = params
                            cost = 0.0
                            N = 0
                            func_vals = plane(seps_g, heights_g, a, b, c)
                            for num in range(self.max_files):
                                diff = np.abs(grids_sc[:,:,num] - func_vals)
                                var = (np.ones_like(grids_sc[:,:,num]))**2
                                #var = (errs_sc[:,:,num])**2
                                cost += np.sum( diff**2 / var )
                                N += diff.size
                            if Ndof:
                                cost *= (1.0 / float(N-1))
                            return cost

                        def err_cost_function(params, Ndof=True):
                            a, b, c = params
                            cost = 0.0
                            N = 0
                            func_vals = plane(seps_g, heights_g, a, b, c)
                            for num in range(self.max_files):
                                diff = np.abs(errs_sc[:,:,num] - func_vals)
                                var = (np.ones_like(errs_sc[:,:,num]))**2
                                cost += np.sum( diff**2 / var )
                                N += diff.size
                            if Ndof:
                                cost *= (1.0 / float(N-1))
                            return cost
    
                        ### Optimize the previously defined function
                        res = opti.minimize(cost_function, [0.0, 0.0, 1.0])
                        err_res = opti.minimize(err_cost_function, [0.0, 0.0, 1.0])

                        x = res.x
                        err_x = err_res.x

                        #print res
                        #print err_res

                        fit_plane = plane(seps_g, heights_g, x[0] * scale_fac, \
                                          x[1] * scale_fac, x[2] * scale_fac)
                        err_fit_plane = plane(seps_g, heights_g, err_x[0] * scale_fac, \
                                              err_x[1] * scale_fac, err_x[2] * scale_fac)

                        stopplanar = time.time()
                        #print "Planar fit: ", stopplanar - startplanar

                        ### Plot the set of grids and the fit result as a qualitative check
                        if plot and ind == 1:

                            fig = plt.figure()
                            ax = fig.gca(projection='3d')
                            for num in range(self.max_files):
                                ax.scatter(seps_g, heights_g, grids[k,:,:,num], color='C0')
                            ax.plot_surface(seps_g, heights_g, fit_plane, alpha=0.3, color='k')
                            ax.legend()
                            ax.set_xlabel('X-separation [um]')
                            ax.set_ylabel('Z-position [um]')
                            ax.set_zlabel('Alpha %s [arb]' % ax_dict[resp])

                            fig2 = plt.figure()
                            ax1 = fig2.gca(projection='3d')
                            for num in range(self.max_files):
                                ax1.scatter(seps_g, heights_g, errs[k,:,:,num], color='C1')
                            ax1.plot_surface(seps_g, heights_g, err_fit_plane, alpha=0.3, color='k')
                            ax1.legend()
                            ax1.set_xlabel('X-separation [um]')
                            ax1.set_ylabel('Z-position [um]')
                            ax1.set_zlabel('Alpha %s [arb]' % ax_dict[resp])

                            plt.show()


                        
                        #alphas = new_grids[k][:][:,:].flatten()
                        #alphas_sc = alphas * (1.0 / scale_fac)

                        alphas_sc = grids_sc.flatten()

                        n, bins = np.histogram(alphas_sc, bins=60, range=(-400,400))
                        bin_centers = bins[:-1] + 0.5*(bins[1] - bins[0])
                        inds = (bin_centers > -4e9) * (bin_centers < 4e9)

                        std_guess = np.std(alphas_sc)
                        hist_max = np.max(n)
                        p0 = [hist_max * np.sqrt(2 * np.pi) * std_guess, 0.0, std_guess]

                        try:
                            startg = time.time()
                            popt_g, pcov_g = opti.curve_fit(gauss, bin_centers[inds], \
                                                            n[inds], p0=p0, maxfev=10000)
                            stopg = time.time()
                            #print "Gauss fit: ", stopg-startg
                            
                            startc = time.time()
                            popt_c, pcov_c = opti.curve_fit(cauchy, bin_centers[inds], \
                                                            n[inds], p0=p0, maxfev=10000)
                            stopc = time.time()
                            #print "Cauchy fit: ", stopc-startc

                            gauss_r2 = r2_goodness_of_fit(gauss, bin_centers[inds], \
                                                          n[inds], popt_g)

                            cauchy_r2 = r2_goodness_of_fit(cauchy, bin_centers[inds], \
                                                           n[inds], popt_c)

                        except Exception:
                            lamb_str = '%0.4e' % self.lambdas[lambind]
                            print("COULDN'T FIT", resp, k, lamb_str)
                            popt_g = p0
                            popt_c = p0

                            gauss_r2 = 0.0
                            cauchy_r2 = 0.0

                        # Save histograms and grid plots
                        if save_hists:

                            lambda_str_1 = ', lambda: %0.4e' % self.lambdas[lambind]
                            title = 'resp: ' + str(resp) + lambda_str_1 + \
                                    ', basis index: ' + str(k) + ', N: ' + str(len(alphas_sc))

                            lambda_str_2 = '_lambda-%0.4e' % self.lambdas[lambind]
                            save_title = 'resp-' + str(resp) + lambda_str_2 + \
                                         '_basis-index-' + str(k) + '_N-' + str(len(alphas_sc)) + '.png'

                            date = hist_info['date']
                            prefix = hist_info['prefix']
                            if prefix[-1] != '_':
                                prefix += '_'

                            fig_path = '/home/cblakemore/plots/' + date + '/grids/' + prefix + save_title

                            fig2_path = '/home/cblakemore/plots/' + date + '/dists/' + prefix + save_title


                            startplot = time.time()
                            fig = plt.figure(1)
                            ax = fig.gca(projection='3d')
                            for num in range(self.max_files):
                                ax.scatter(seps_g, heights_g, grids[k,:,:,num], color='C0')
                            ax.plot_surface(seps_g, heights_g, fit_plane, alpha=0.3, color='k')
                            ax.legend()
                            ax.set_title(title)
                            ax.set_xlabel('X-separation [um]')
                            ax.set_ylabel('Z-position [um]')
                            ax.set_zlabel('Alpha %s [arb]' % ax_dict[resp])

                            try:
                                fig.savefig(fig_path)
                            except IOError:
                                bu.make_all_pardirs(fig_path)
                                fig.savefig(fig_path)
                            
                            plt.close(fig)


                            fig2 = plt.figure(2)
                            ax1 = fig2.add_subplot(111)
                            ax1.fill_between(bin_centers[inds], n[inds], \
                                             np.zeros_like(n[inds]), step='mid', \
                                             alpha=0.3, color='k')
                            ax1.set_ylim(0, 1.2*np.max(n))

                            plot_bins = np.linspace(np.min(bin_centers[inds]), \
                                                    np.max(bin_centers[inds]), \
                                                    500)

                            ax1.plot(plot_bins, gauss(plot_bins, *popt_g), \
                                     label='Gaussian', lw=2)
                            ax1.plot(plot_bins, cauchy(plot_bins, *popt_c), \
                                     label='Cauchy (Lorentz)', lw=2)
                            ax1.set_ylabel('Counts')
                            ax1.set_xlabel('Scaled Alpha [%0.3e $\\alpha$]' % scale_fac)
                            ax1.legend()

                            ax1.set_title(title)

                            gauss_val = popt_g[1] * scale_fac
                            gauss_val_2 = popt_g[2] * scale_fac / np.sqrt(len(alphas_sc))
                            gauss_text =   'Gauss fit: $\mu$ = %0.3e' % gauss_val
                            gauss_text_2 = '         $\sigma/rt(N)$ = %0.3e' % gauss_val_2
                            cauchy_text = 'Cauchy fit: $\mu$ = %0.3e' % (popt_c[1] * scale_fac)
                            ax1.annotate(gauss_text, [-400, np.max(n)*0.75], xycoords='data')
                            ax1.annotate(gauss_text_2, [-400, np.max(n)*0.7], xycoords='data')
                            ax1.annotate(cauchy_text, [-400, np.max(n)*0.60], xycoords='data')

                            try:
                                fig2.savefig(fig2_path)
                            except IOError:
                                bu.make_all_pardirs(fig2_path)
                                fig2.savefig(fig2_path)
                            plt.close(fig2)
                            
                            #plt.show()

                            stopplot = time.time()
                            #print "Plotting/Saving: ", stopplot - startplot



                        alpha_xyz_best_fit[lambind][resp][k] = [x[2] * scale_fac, \
                                                                err_x[2] * scale_fac, \
                                                                popt_g[1] * scale_fac, \
                                                                popt_c[1] * scale_fac, \
                                                                popt_g[2] * scale_fac, \
                                                                popt_c[2] * scale_fac, \
                                                                gauss_r2, cauchy_r2, \
                                                                len(alphas_sc)]
    
        
        self.alpha_xyz_best_fit = np.array(alpha_xyz_best_fit)












    def fit_mean_alpha_vs_alldim(self, weight_planar=True):

        if not self.gfuncs_class.grav_loaded:
            try:
                self.gfuncs_class.reload_grav_funcs()
            except Exception:
                print('No grav funcs... Tried to reload but no filename')

        self.alpha_best_fit = []
        self.alpha_95cl = []

        ### Assume separations are encoded in ax0 and heights in ax1
        seps = self.p0_bead[0] - np.array(self.ax0vec)
        heights = self.p0_bead[2] - np.array(self.ax1vec) 
        
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

                for ax0ind, ax0pos in enumerate(self.ax0vec):

                    ### Loop over all files at each separation and collect
                    ### the value of alpha for the current value of yuklambda
                    for ax1ind, ax1pos in enumerate(self.ax1vec):

                        tempdat = self.alpha_dict[bias][ax0pos][ax1pos]

                        dat[ax0ind][ax1ind] = tempdat[0][lambind]
                        errs[ax0ind][ax1ind] = tempdat[1][lambind]

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
                except Exception:
                    2+2
                
                ### Deplane the data and extract some statistics
                deplaned = dat - scale_fac * (x[0] * heights_g + x[1] * seps_g + x[2])
                deplaned_wconst = deplaned + scale_fac * x[2]

                deplaned_avg = np.mean(deplaned)
                deplaned_std = np.std(deplaned) / dat.size

                self.alpha_best_fit.append(np.abs(x[2]*scale_fac))
                self.alpha_95cl.append(deplaned_std)



    



    def plot_sensitivity(self, show=True, plot_just_current=False, plot_ob=True):

        if not self.gfuncs_class.grav_loaded:
            try:
                self.gfuncs_class.reload_grav_funcs()
            except Exception:
                print('No grav funcs... Tried to reload but no filename')

        fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)

        if not plot_just_current:
            ax.loglog(self.gfuncs_class.lambdas, self.alpha_best_fit, \
                        linewidth=2, label='Best fit $\\hat{\\alpha}_{\\rm ib}$', \
                        color='C0', ls='--')
            ax.loglog(self.gfuncs_class.lambdas, self.alpha_95cl, \
                      linewidth=2, label='95% CL on $\\hat{\\alpha}_{\\rm ib}$', color='C0')

            if plot_ob:
                ax.loglog(self.gfuncs_class.lambdas, self.alpha_best_fit_null, \
                          linewidth=2, color='C1', ls='--', \
                          label='Best fit $\\hat{\\alpha}_{\\rm ob}$ with $\\alpha_{\\rm ib}=0$',)
                ax.loglog(self.gfuncs_class.lambdas, self.alpha_95cl_null, \
                          linewidth=2, label='95% CL on $\\hat{\\alpha}_{\\rm ob}$', color='C1')


        ax.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
        ax.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
        ax.grid()

        #ax.set_xlim(lambda_plot_lims[0], lambda_plot_lims[1])
        #ax.set_ylim(alpha_plot_lims[0], alpha_plot_lims[1])

        ax.set_xlabel('$\\lambda$ [m]')
        ax.set_ylabel('$\\alpha$')

        ax.legend(numpoints=1, fontsize=9)

        #ax.set_title(figtitle)
        plt.tight_layout()

        if show:
            plt.show()
        else:
            return (fig, ax)



    # def save_sensitivity(self, path=''):


