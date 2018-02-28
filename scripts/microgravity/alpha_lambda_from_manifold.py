import sys, time

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opti

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config




#data_manifold_path = '/force_v_pos/20170903_diagforce_v_pos_dic.p'
data_manifold_path = '/force_v_pos/20171106_diagforce_v_pos_dic.p'

theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'

#savepath = '/sensitivities/20170903data_95cl_alpha_lambda.npy'
savepath = '/sensitivities/20171106data_95cl_alpha_lambda_farpoints.npy'
figtitle = 'Sensitivity: Patterned Attractor'

confidence_level = 0.95

#user_lims = [(65e-6, 80e-6), (-240e-6, 240e-6), (-5e-6, 5e-6)]
user_lims = [(5e-6, 20e-6), (-240e-6, 240e-6), (-5e-6, 5e-6)]
#user_lims = []

######### HARDCODED DUMB SHIT WHILE IMAGE ANALYSIS IS BEING TWEAKED

minsep = 5    # um
maxthrow = 75  # um


#########################################################

# Various fitting functions
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def line(x, a, b):
    return a * x + b

def const(x, a):
    return a

def flicker(x, a):
    return a * (1. / x)



def build_mod_grav_funcs(theory_data_dir):
    '''Loads data from the output of /data/grav_sim_data/process_data.py
       which processes the raw simulation output from the farmshare code

       INPUTS: data_dir, path to the directory containing the data

       OUTPUTS: gfuncs, 3 element list with 3D interpolating functions
                        for regular gravity [fx, fy, fz]
                yukfuncs, 3 x Nlambda array with 3D interpolating function
                          for modified gravity with indexing: 
                          [[y0_fx, y1_fx, ...], [y0_fy, ...], [y0_fz, ...]]
                lambdas, np.array with all lambdas from the simulation
                lims, 3 element with tuples for (min, max) of coordinate
                      limits in interpolation
    '''

    # Load modified gravity curves from simulation output
    Gdata = np.load(theory_data_dir + 'Gravdata.npy')
    yukdata = np.load(theory_data_dir + 'yukdata.npy')
    lambdas = np.load(theory_data_dir + 'lambdas.npy')
    xpos = np.load(theory_data_dir + 'xpos.npy')
    ypos = np.load(theory_data_dir + 'ypos.npy')
    zpos = np.load(theory_data_dir + 'zpos.npy')
    
    lambdas = lambdas[::-1]
    yukdata = np.flip(yukdata, 0)

    # Find limits to avoid out of range erros in interpolation
    xlim = (np.min(xpos), np.max(xpos))
    ylim = (np.min(ypos), np.max(ypos))
    zlim = (np.min(zpos), np.max(zpos))

    # Build interpolating functions for regular gravity
    g_fx_func = interp.RegularGridInterpolator((xpos, ypos, zpos), Gdata[:,:,:,0])
    g_fy_func = interp.RegularGridInterpolator((xpos, ypos, zpos), Gdata[:,:,:,1])
    g_fz_func = interp.RegularGridInterpolator((xpos, ypos, zpos), Gdata[:,:,:,2])

    # Build interpolating functions for yukawa-modified gravity
    yuk_fx_funcs = []
    yuk_fy_funcs = []
    yuk_fz_funcs = []
    for lambind, yuklambda in enumerate(lambdas):
        fx_func = interp.RegularGridInterpolator((xpos, ypos, zpos), yukdata[lambind,:,:,:,0])
        fy_func = interp.RegularGridInterpolator((xpos, ypos, zpos), yukdata[lambind,:,:,:,1])
        fz_func = interp.RegularGridInterpolator((xpos, ypos, zpos), yukdata[lambind,:,:,:,2])
        yuk_fx_funcs.append(fx_func)
        yuk_fy_funcs.append(fy_func)
        yuk_fz_funcs.append(fz_func)

    gfuncs = [g_fx_func, g_fy_func, g_fz_func]
    yukfuncs = [yuk_fx_funcs, yuk_fy_funcs, yuk_fz_funcs]
    lims = [xlim, ylim, zlim]

    return gfuncs, yukfuncs, lambdas, lims


wvnum_upp = 1. / 5.   # um^-1, define an upper limit for noise model fit
wvnum_low = 1. / 800.  # um^-1, define an lower limit for noise model fit
wvnum_sig = 1. / 50.   # um^-1, expected signal (to remove from noise model estimate)


def generate_alpha_lambda_limit(data_manifold, gfuncs, yukfuncs, lambdas, \
                                lims, confidence_level=0.95, sig_period=50., \
                                short_period=5., long_period=500., noise_func=const,\
                                plot=False, save=False, \
                                savepath=''):
    '''Fits a data manifold against simulations of modified gravity

       INPUTS: data_manifold, data output from force_v_pos_manifold.py
               gfuncs, 3 element list with 3D interpolating functions
                       for regular gravity [fx, fy, fz]
               yukfuncs, 3 x Nlambda array with 3D interpolating function
                         for modified gravity with indexing: 
                         [[y0_fx, y1_fx, ...], [y0_fy, ...], [y0_fz, ...]]
               lambdas, np.array with all lambdas from the simulation
               lims, 3 element with tuples for (min, max) of coordinate
                     limits in interpolation
               confidence_level, determines final sensitivity
               sig_period, period in [um] of expected signal
               short_period, cut off for short period fluctuations in noise model
               long_period,     ''       long         ''
               plot, boolean specifying whether to plot stuff
               save, boolean specifying to save or not
               savepath, path to write limit data. Must be non-empty string
                         for the saving to work

       OUTPUTS: lambdas, same as input
                alphas, alpha corresponding to confidence level
    '''

    # For the confidence interval, compute the inverse CDF of a 
    # chi^2 distribution at given confidence level and compare to 
    # liklihood ratio via a goodness of fit parameter.
    # Refer to scipy.stats documentation to understand chi2
    chi2dist = stats.chi2(1)
    # factor of 0.5 from Wilks's theorem: -2 log (Liklihood) ~ chi^2(1)
    con_val = 0.5 * chi2dist.ppf(confidence_level)



    wvnum_sig = 1. / sig_period
    wvnum_high = 1. / short_period
    wvnum_low = 1. / long_period

    #lambdas = lambdas[:10]
    alphas = np.zeros_like(lambdas)

    min_chisq = []

    #lambdas = lambdas[::-1]
    # HARDCODED NUMBERS BEWARE

    colors = bu.get_color_map(len(lambdas))

    xarr = np.sort( np.array(data_manifold.keys()) )
    zarr = np.sort( np.array(data_manifold[xarr[0]].keys()) )

    per = 0.0
    for lambind, yuklambda in enumerate(lambdas):
        sys.stdout.flush()
        newper = (float(lambind) / float(len(lambdas))) * 100.
        if newper > per + 1.0:
            print int(per),
            sys.stdout.flush()
            per = newper
        chi_sqs = np.zeros(len(testalphas))

        for alphaind, testalpha in enumerate(testalphas):
            N = 0
            chi_sq = 0

            #start = time.time()
            for xpos in xarr:
                if (xpos < lims[0][0]*1e6) or (xpos > lims[0][1]*1e6):
                    #print 'skipped x'
                    continue
                # HARDCODED SHITEEE
                newxpos = minsep + (maxthrow - xpos)
                for zpos in zarr:
                    if (zpos < lims[2][0]*1e6) or (zpos > lims[2][1]*1e6):
                        #print 'skipped z'
                        continue
                    for resp in [0,1,2]:
                        bins = data_manifold[xpos][zpos][resp][0] 
                        force = data_manifold[xpos][zpos][resp][1]

                        fft = np.fft.rfft(force)
                        wvnum = np.fft.rfftfreq( len(force), d=(bins[1]-bins[0]) )
                        asd = np.abs(fft)

                        ones = np.ones_like(bins)
                        pts = np.stack((newxpos*ones, bins, zpos*ones), axis=-1)
                        gforce = gfuncs[resp](pts*1e-6)
                        yukforce = yukfuncs[resp][lambind](pts*1e-6)

                        gfft = np.fft.rfft(gforce)
                        yukfft = np.fft.rfft(yukforce)

                        sigarg = np.argmin( np.abs(wvnum - wvnum_sig) )
                        hinds = wvnum < wvnum_high
                        linds = wvnum > wvnum_low
                        inds = hinds * linds
                        inds[sigarg-1:sigarg+2] = False

                        #plt.plot(bins, force)
                        #plt.plot(bins, yukforce*1e10)
                        #plt.show()

                        try:
                            noise_popt, _ = opti.curve_fit(noise_func, wvnum[inds], asd[inds])
                            noise_asd = noise_func(wvnum, *noise_popt)
                        except:
                            noise_asd = np.mean(asd[inds]) * np.ones_like(wvnum)
                        
                        
                        #diff = fft - (gfft + 10**testalpha * yukfft)
                        diff = fft - (gfft + testalpha * yukfft)

                        chi_sq += np.sum( np.abs(diff)**2 / noise_asd**2 )
                        N += len(fft)
            #stop = time.time()
            #print 'Single Loop: ', stop-start
            red_chi_sq = chi_sq / (N - 1)
            chi_sqs[alphaind] = red_chi_sq
                   
        #fitalphas = 10**testalphas
        fitalphas = testalphas

        max_chi = np.max(chi_sqs)
        max_alpha = np.max(testalphas)

        p0 = [max_chi/max_alpha**2, 0, 1]

        #if lambind == 0:
        #    p0 = [0.15e9, 0, 5]
        #else:
        #    p0 = p0_old

        if plot:
            plt.plot(fitalphas, chi_sqs, color = colors[lambind])
    
        try:
            popt, pcov = opti.curve_fit(parabola, fitalphas, chi_sqs, \
                                            p0=p0, maxfev = 100000)
        except:
            print "Couldn't fit"
            popt = [0,0,0]
            popt[2] = np.mean(chi_sqs)

        #p0_old = popt

        con_val = con_val + np.min(chi_sqs)

        # Select the positive root for the non-diagonalized data
        soln1 = ( -1.0 * popt[1] + np.sqrt( popt[1]**2 - \
                        4 * popt[0] * (popt[2] - con_val)) ) / (2 * popt[0])
        soln2 = ( -1.0 * popt[1] - np.sqrt( popt[1]**2 - \
                        4 * popt[0] * (popt[2] - con_val)) ) / (2 * popt[0])
        if soln1 > soln2:
            alpha_con = soln1
        else:
            alpha_con = soln2

        alphas[lambind] = alpha_con

    if plot:
        plt.title('Goodness of Fit for Various Lambda', fontsize=16)
        plt.xlabel('Alpha Parameter [arb]', fontsize=14)
        plt.ylabel('$\chi^2$', fontsize=18)

        plt.show()

    if save:
        if savepath == '':
            print 'No save path given, type full path here'
            savepath = raw_input('path: ')

        np.save(savepath, [lambdas, alphas])


    return lambdas, alphas

        

# Load Data and recall indexing
# outdic[ax1pos][ax2pos][resp(0,1,2)][bins(0) or dat(1)]
data_manifold = pickle.load( open(data_manifold_path, 'rb') )


# Load modified gravity and build functions
gfuncs, yukfuncs, lambdas, lims = build_mod_grav_funcs(theory_data_dir)


if len(user_lims):
    lims = user_lims

newlambdas, alphas = generate_alpha_lambda_limit(data_manifold, gfuncs, yukfuncs, lambdas, \
                                lims, confidence_level=0.95, sig_period=50., \
                                short_period=5., long_period=500., \
                                plot=True, save=True, savepath=savepath)



### Load limits to plot against

limitdata_path = '/home/charles/opt_lev_analysis/scripts/gravity_sim/gravity_sim_v2/data/limitdata_20160928_datathief_nodecca2.txt'
#limitdata_path = '/home/charles/limit_nodecca2.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/home/charles/opt_lev_analysis/scripts/gravity_sim/gravity_sim_v2/data/limitdata_20160914_datathief.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'



fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)

ax.loglog(newlambdas, alphas, linewidth=2, label='95% CL')
ax.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
ax.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
ax.grid()

ax.set_xlabel('$\lambda$ [m]')
ax.set_ylabel('$\\alpha$')

ax.legend(numpoints=1, fontsize=9)

ax.set_title(figtitle)

plt.tight_layout(w_pad=1.2, h_pad=1.2, pad=1.2)

plt.show()
