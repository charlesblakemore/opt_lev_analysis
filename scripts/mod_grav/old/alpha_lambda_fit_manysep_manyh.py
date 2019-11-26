import numpy as np
import matplotlib.pyplot as plt
import dill as pickle

import bead_util as bu
import cant_util as cu
import grav_util as gu

import scipy.signal as signal
import scipy.optimize as optimize 
import scipy.stats as stats

import sys, time


### NOTE: RESPONSE AXIS AND CANTILEVER SWEEP AXIS ARE CURRENTLY
### HARD-CODED IN THE ANALYSIS BELOW. EVENTUALLY, WE MIGHT WANT
### TO FIT THE RESPONSE IN MULTIPLE DIRECTIONS, ESPECIALLY IF 
### THE DOMINANT BACKGROUND IS INDEED PATCH POTENTIALS AS THE 
### DIPOLE MAY HAVE A PREFERRED ORIENTATION

save_path = '/sensitivities/alpha_lambda_sens95cl_20170903_flickernoise_minchi1.p'
extra_lab = ' Many Sep, Many Height'

# Choose whether to force the min_chisq to 1 or divide the noise
# by an arbitrary factor to accomplish the same thing.
# We really need to devise a more accurate noise model...
setmin_chisq = True
noise_fac = 1.

# Parameters for signal band rejection in the development
# of a noise model based on the response at neighboring 
# spatial frequencies
band = 1   # um
bandwidth = 1. / band
show_data_at_modulation = True
dispfit = True

resp_ax = 0
vel_mult = 0

### Load theoretical modified gravity curves

gpath = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/'
filname = 'attractorv2_1um_manysep_fullthrow_force_curves.p'

fcurve_obj = gu.Grav_force_curve(path=gpath+filname, make_splines=True, \
                                 spline_order=3)

# Choice of bead radius for theoretical curves
RBEAD = 2.37e-6

confidence_level = 0.95

# For the confidence interval, compute the inverse CDF of a 
# chi^2 distribution at given confidence level and compare to 
# liklihood ratio via a goodness of fit parameter.
# Refer to scipy.stats documentation to understand chi2
chi2dist = stats.chi2(1)
# factor of 0.5 from Wilks's theorem: -2 log (Liklihood) ~ chi^2(1)
con_val = 0.5 * chi2dist.ppf(confidence_level) 



### Load backgrounds

procd_data = '/processed_data/grav_data/manysep_20170903.p'
procd_dirs = pickle.load( open(procd_data, 'rb') )
for objind, obj in enumerate(procd_dirs):
    obj.get_closest_sep_and_pos()
    obj.bead_height = 10.0    # height from dipole measurements

cal_facs = procd_dirs[0].conv_facs

background_data = cu.Force_v_pos()
background_data.load_dir_objs(procd_dirs, organize=True)

assert type(background_data.dat) != str

dat = background_data.dat
seps = background_data.seps
heights = background_data.heights

print("looping over %i seps" % len(seps))
print("looping over %i heights" % len(heights))

#seps = seps[:5]
#heights = heights[:5]


### Load limits to plot against

#limitdata_path = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160928_datathief_nodecca2.txt'
limitdata_path = '/home/charles/limit_nodecca2.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160914_datathief.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'




### Construct a noise model

# Various fitting functions
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def line(x, a, b):
    return a * x + b

def const(x, a):
    return a

def flicker(x, a):
    return a * (1. / x)

wvnum_upp = 1. / 5.   # um^-1, define an upper limit for noise model fit
wvnum_low = 1. / 800.  # um^-1, define an lower limit for noise model fit
wvnum_sig = 1. / 50.   # um^-1, expected signal (to remove from noise model estimate)

lambdas = fcurve_obj.lambdas
alphas = np.zeros_like(lambdas)
diagalphas = np.zeros_like(lambdas)

min_chisq = []
diagmin_chisq = []

lambdas = lambdas[::-1]
testalphas = np.linspace(0, 12, 1000)

colors = bu.get_color_map(len(lambdas))

per = 0.0
for ind, yuklambda in enumerate(lambdas):
    newper = (float(ind) / float(len(lambdas))) * 100.
    if newper > per + 1.0:
        print(int(per), end=' ')
        sys.stdout.flush()
        per = newper
    chi_sqs = np.zeros(len(testalphas))
    diag_chi_sqs = np.zeros(len(testalphas))

    start = time.time()

    for alphaind, testalpha in enumerate(testalphas):

        N = 0
        diagN = 0
        chi_sq = 0
        diag_chi_sq = 0
    
        for sep in seps:
            for height in heights:

                bins = dat[sep][height][0][resp_ax,vel_mult][0]
                force = dat[sep][height][0][resp_ax,vel_mult][1]*cal_facs[resp_ax]

                diagbins = dat[sep][height][1][resp_ax,vel_mult][0]
                diagforce = dat[sep][height][1][resp_ax,vel_mult][1]

                fft = np.fft.rfft(force)
                wvnum = np.fft.rfftfreq( len(force), d=(bins[1]-bins[0]) )
                asd = np.abs(fft)

                diagfft = np.fft.rfft(diagforce)
                diagwvnum = np.fft.rfftfreq( len(diagforce), d=(diagbins[1]-diagbins[0]) )
                diagasd = np.abs(diagfft)

                
                SEP = np.sqrt(sep**2 + height**2)

                gforce = fcurve_obj.mod_grav_force(bins*1e-6, sep=SEP, alpha=1., \
                                       yuklambda=yuklambda, rbead=RBEAD, nograv=True)
                diaggforce = fcurve_obj.mod_grav_force(diagbins*1e-6, sep=SEP, alpha=1., \
                                           yuklambda=yuklambda, rbead=RBEAD, nograv=True)

                #gforce = signal.detrend(gforce)
                #diaggforce = signal.detrend(diaggforce)

                gfft = np.fft.rfft(gforce)
                gasd = np.abs(gfft)
                diaggfft = np.fft.rfft(diaggforce)
                diaggasd = np.abs(diaggfft)

                sigarg = np.argmin( np.abs( wvnum - wvnum_sig ) )
                dsigarg = np.argmin( np.abs( diagwvnum - wvnum_sig ) )

                # Select only the wavenumbers we want in our noise model
                inds = wvnum < wvnum_upp
                inds = wvnum[inds] > wvnum_low
                inds[sigarg-1:sigarg+2] = False

                dinds = diagwvnum < wvnum_upp
                dinds = diagwvnum[dinds] > wvnum_low
                dinds[dsigarg-1:dsigarg+2] = False

                # Estimate a noise model by fitting the ASD to a line
                popt, pcov = optimize.curve_fit(flicker, wvnum[inds], asd[inds], \
                                                  p0 = [1e-15])
                diagpopt, diagpcov = optimize.curve_fit(flicker, diagwvnum[dinds], \
                                                  diagasd[dinds], p0 = [1e-15])

                #noise_asd = np.zeros_like(wvnum) + np.mean(asd[inds])
                noise_asd = flicker(wvnum, *popt)
                #noise_asd = asd

                #diagnoise_asd = np.zeros_like(diagwvnum) + np.mean(diagasd[inds])
                diagnoise_asd = flicker(diagwvnum, *diagpopt)
                #diagnoise_asd = diagasd

                #plt.loglog(wvnum, noise_asd)
                #plt.loglog(wvnum, asd)
                #plt.show()

                # Scale the noise by the arbitraily defined noise fac
                noise_asd_s = noise_asd / noise_fac
                diagnoise_asd_s = diagnoise_asd / noise_fac
            
                chi_sq += np.sum( np.abs(fft - 10**testalpha * gfft)**2 / noise_asd_s**2 )
                diag_chi_sq += np.sum( np.abs(diagfft - 10**testalpha * diaggfft)**2 / diagnoise_asd_s**2 )

                N += len(fft)
                diagN += len(diagfft)
        
        red_chi_sq = chi_sq / (N - 1)
        diag_red_chi_sq = diag_chi_sq / (diagN - 1)
        chi_sqs[alphaind] = red_chi_sq
        diag_chi_sqs[alphaind] = diag_red_chi_sq

    stop = time.time()
    print("time for single lambda: ", stop-start)


    if setmin_chisq:
        print(np.min(chi_sqs))
        print(np.min(diag_chi_sqs))
        min_chisq.append(np.min(chi_sqs))
        diagmin_chisq.append(np.min(diag_chi_sqs))

        chi_sqs = chi_sqs / np.min(chi_sqs)
        diag_chi_sqs = diag_chi_sqs / np.min(diag_chi_sqs)

    #plt.plot(10**testalphas, chi_sqs)
    #plt.show()

    fitalphas = 10**testalphas

    #if yuklambda < 1.0e-6:
    #    plt.plot(fitalphas, chi_sqs)
    #    plt.plot(fitalphas, parabola(fitalphas, 0.15, 0, 0.5))
    #    plt.show()

    if ind == 0:
        p0 = [0.15*1e9, 0, 0.5]
        dp0 = [0.15*1e9, 0, 0.5]
    else:
        p0 = p0_old
        dp0 = dp0_old

    plt.plot(fitalphas, chi_sqs, color = colors[ind])

    try:
        popt, pcov = optimize.curve_fit(parabola, fitalphas, chi_sqs, p0 = p0, maxfev = 100000)
        diagpopt, diagpcov = optimize.curve_fit(parabola, fitalphas, diag_chi_sqs, p0 = dp0, maxfev = 100000)
    except:
        print("Couldn't fit")
        popt = p0_old
        popt[2] = np.mean(chi_sqs)
        diagpopt = dp0_old
        diagpopt[2] = np.mean(diag_chi_sqs)
    
    p0_old = popt
    dp0_old = diagpopt
    
    # Select the positive root for the non-diagonalized data
    soln1 = ( -1.0 * popt[1] + np.sqrt( popt[1]**2 - \
                    4 * popt[0] * (popt[2] - con_val)) ) / (2 * popt[0])
    soln2 = ( -1.0 * popt[1] - np.sqrt( popt[1]**2 - \
                    4 * popt[0] * (popt[2] - con_val)) ) / (2 * popt[0])
    if soln1 > soln2:
        alpha_con = soln1
    else:
        alpha_con = soln2

    # Select the positive root for the diagonalized data
    diagsoln1 = ( -1.0 * diagpopt[1] + np.sqrt( diagpopt[1]**2 - \
                        4 * diagpopt[0] * (diagpopt[2] - con_val)) ) / (2 * diagpopt[0])
    diagsoln2 = ( -1.0 * diagpopt[1] - np.sqrt( diagpopt[1]**2 - \
                        4 * diagpopt[0] * (diagpopt[2] - con_val)) ) / (2 * diagpopt[0])
    if diagsoln1 > diagsoln2:
        diag_alpha_con = diagsoln1
    else:
        diag_alpha_con = diagsoln2

    #diag_alpha_con = np.sqrt( (con_val - diagpopt[2]) / diagpopt[0] ) + diagpopt[1]
    alphas[ind] = alpha_con
    diagalphas[ind] = diag_alpha_con 

    

outdic = {}
outdic['min_chisq'] = min_chisq
outdic['diagmin_chisq'] = diagmin_chisq
outdic['lambdas'] = lambdas
outdic['alphas'] = alphas
outdic['diagalphas'] = diagalphas
pickle.dump( outdic, open(save_path, 'wb') )


plt.title('Goodness of Fit for Various Lambda', fontsize=16)
plt.xlabel('Alpha Parameter [arb]', fontsize=14)
plt.ylabel('$\chi^2$', fontsize=18)

plt.show()





f3, axarr3 = plt.subplots(1,2,sharex='all',sharey='all',figsize=(10,5),dpi=100)
plt.suptitle('Sensitivity' + extra_lab, fontsize=18)
axarr3[0].set_title('Raw Data', fontsize=16)
axarr3[0].loglog(lambdas, alphas, linewidth=2, label='95% CL')
axarr3[0].loglog(limitdata[:,0], limitdata[:,1], '--', \
                 label=limitlab, linewidth=3, color='r')
axarr3[0].loglog(limitdata2[:,0], limitdata2[:,1], '--', \
                 label=limitlab2, linewidth=3, color='k')
axarr3[0].grid()

axarr3[0].set_ylabel('alpha [arb]')
axarr3[0].set_xlabel('lambda [m]')

axarr3[1].set_title('Diagonalized Data', fontsize=16)
axarr3[1].loglog(lambdas, diagalphas, linewidth=2, label='95% CL')
axarr3[1].loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
axarr3[1].loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
axarr3[1].grid()

axarr3[1].set_xlabel('lambda [m]')

axarr3[0].legend(numpoints=1, fontsize=9)
axarr3[1].legend(numpoints=1, fontsize=9)






plt.show()
