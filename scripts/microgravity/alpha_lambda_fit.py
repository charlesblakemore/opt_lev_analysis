import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu
import cant_util as cu
import grav_util as gu

import scipy.signal as signal
import scipy.optimize as optimize 
import scipy.stats as stats

import sys

band = 10   # um
bandwidth = 1. / band
show_data_at_modulation = True
dispfit = True

### Load theoretical modified gravity curves

gpath = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/'
filname = 'attractorv2_1um_manysep_fullthrow_force_curves.p'

fcurve_obj = gu.Grav_force_curve(path=gpath+filname, make_splines=True, spline_order=3)

SEP = 5.0e-6
RBEAD = 2.43e-6

confidence_level = 0.95



### Load backgrounds

background_data = cu.Force_v_pos()
background_data.load('/force_v_pos/20170718_grav_background.p')

bins = background_data.bins
force = background_data.force
errs = background_data.errs

diagbins = background_data.diagbins
diagforce = background_data.diagforce
diagerrs = background_data.diagerrs

wvnum = np.fft.rfftfreq(len(force), d=(bins[1]-bins[0]))
datfft = np.fft.rfft(force)
datasd = np.abs(datfft)

diagwvnum = np.fft.rfftfreq(len(diagforce), d=(diagbins[1]-diagbins[0]))
diagdatfft = np.fft.rfft(diagforce)
diagdatasd = np.abs(diagdatfft)




### Load limits to plot against

limitdata_path = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160928_datathief_nodecca2.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160914_datathief.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'




### Construct a noise model

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def line(x, a, b):
    return a * x + b

def const(x, a):
    return a

wvnum_upp = 1. / 20.   # um^-1, define an upper limit for noise model fit
wvnum_low = 1. / 100.  # um^-1, define an lower limit for noise model fit
wvnum_sig = 1. / 50.   # um^-1, expected signal (to remove from noise model estimate)

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
popt, pcov = optimize.curve_fit(const, wvnum[inds], datasd[inds], p0 = [1e-15])
diagpopt, diagpcov = optimize.curve_fit(const, diagwvnum[dinds], diagdatasd[dinds], p0 = [1e-15])

noise_asd = np.zeros_like(wvnum) + popt[0]
#noise_asd = wvnum * popt[0] + popt[1]

diagnoise_asd = np.zeros_like(diagwvnum) + diagpopt[0]
#diagnoise_asd = diagwvnum * diagpopt[0] + diagpopt[1]


if show_data_at_modulation:
    filt_datfft = np.zeros_like(datfft)
    for ind, wvnumm in enumerate(wvnum):
        if np.abs(wvnumm - wvnum_sig) < 0.5*bandwidth:
            filt_datfft[ind] = datfft[ind]

    rdat = np.fft.irfft(datfft)
    filt_dat = np.fft.irfft(filt_datfft)
    print len(bins), len(filt_dat), len(rdat)
    plt.plot(bins, rdat)
    plt.plot(bins, filt_dat)
    plt.show()
        



lambdas = fcurve_obj.lambdas
alphas = np.zeros_like(lambdas)
simp_alphas = np.zeros_like(lambdas)
diagalphas = np.zeros_like(lambdas)
simp_diagalphas = np.zeros_like(lambdas)

lambdas = lambdas[::-1]
testalphas = np.linspace(0, 10, 10000)

# For the confidence interval, compute the inverse CDF of a 
# chi^2 distribution at 0.95 and compare to liklihood ratio
# via a goodness of fit parameter
chi2dist = stats.chi2(1)
# factor of 0.5 from Wilke's theorem: -2 log (Liklihood) ~ chi^2(1)
con_val = 0.5 * chi2dist.ppf(confidence_level) 

colors = bu.get_color_map(len(lambdas))

for ind, yuklambda in enumerate(lambdas):
    fcurve = fcurve_obj.mod_grav_force(bins*1e-6, sep=SEP, alpha=1., \
                                       yuklambda=yuklambda, rbead=RBEAD, nograv=True)
    diagfcurve = fcurve_obj.mod_grav_force(diagbins*1e-6, sep=SEP, alpha=1., \
                                           yuklambda=yuklambda, rbead=RBEAD, nograv=True)

    fcurve = signal.detrend(fcurve)
    diagfcurve = signal.detrend(diagfcurve)

    fft = np.fft.rfft(fcurve)
    asd = np.sqrt(fft.conj() * fft)
    diagfft = np.fft.rfft(diagfcurve)
    diagasd = np.sqrt(diagfft.conj() * diagfft)

    scale = 0.7
    #scale = np.abs( np.max(datasd) )
    dscale = 0.7
    #dscale = np.abs( np.max(diagdatasd) )

    datfft_s = datfft / scale
    fft_s = fft / scale
    noise_asd_s = noise_asd / scale

    diagdatfft_s = diagdatfft / dscale
    diagfft_s = diagfft / dscale
    diagnoise_asd_s = diagnoise_asd / dscale

    chi_sqs = []
    diag_chi_sqs = []
    for derpalpha in testalphas:
        chi_sq = np.sum( np.abs(datfft - 10**derpalpha * fft)**2 / noise_asd_s**2 )
        diag_chi_sq = np.sum( np.abs(diagdatfft - 10**derpalpha * diagfft)**2 / diagnoise_asd_s**2 )
        red_chi_sq = chi_sq / (len(datfft) - 1)
        diag_red_chi_sq = diag_chi_sq / (len(diagdatfft) - 1)
        chi_sqs.append(red_chi_sq)
        diag_chi_sqs.append(diag_red_chi_sq)

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
        print "Couldn't fit"
        popt = p0_old
        popt[2] = np.mean(chi_sqs)
        diagpopt = dp0_old
        diagpopt[2] = np.mean(diag_chi_sqs)
    
    p0_old = popt
    dp0_old = diagpopt
    
    soln1 = ( -1.0 * popt[1] + np.sqrt( popt[1]**2 - 4 * popt[0] * (popt[2] - con_val)) ) / (2 * popt[0])
    soln2 = ( -1.0 * popt[1] - np.sqrt( popt[1]**2 - 4 * popt[0] * (popt[2] - con_val)) ) / (2 * popt[0])
    if soln1 > soln2:
        alpha_con = soln1
    else:
        alpha_con = soln2

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

    #print (alpha_95con, alpha_95con * 1e9),
    #sys.stdout.flush()

    maxind = np.argmax(asd)
    diagmaxind = np.argmax(diagasd)
    
    simp_alpha = datasd[maxind] / asd[maxind]
    simp_diagalpha = diagdatasd[diagmaxind] / diagasd[maxind]

    simp_alphas[ind] = simp_alpha
    simp_diagalphas[ind] = simp_diagalpha

    

plt.title('Goodness of Fit for Various Lambda', fontsize=16)
plt.xlabel('Alpha Parameter [arb]', fontsize=14)
plt.ylabel('$\chi^2$', fontsize=18)

plt.show()




f, axarr = plt.subplots(1,2,sharex='all',sharey='all',figsize=(10,5),dpi=100)

axarr[0].set_title('Raw Data', fontsize=16)
axarr[0].errorbar(bins, force*1e15, errs*1e15, fmt='.-', ms=10, color = 'r')
axarr[0].set_ylabel('Force [fN]')
axarr[0].set_xlabel('Position along cantilever [um]')

axarr[1].set_title('Diagonalized Data', fontsize=16)
axarr[1].errorbar(diagbins, diagforce*1e15, diagerrs*1e15, fmt='.-', ms=10, color = 'r')
axarr[1].set_xlabel('Position along cantilever [um]')






f3, axarr3 = plt.subplots(1,2,sharex='all',sharey='all',figsize=(10,5),dpi=100)

axarr3[0].set_title('Raw Data', fontsize=16)
axarr3[0].loglog(lambdas, simp_alphas, label='Naive: Sig = Noise')
axarr3[0].loglog(lambdas, alphas, linewidth=2, label='95% CL')
axarr3[0].loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
axarr3[0].loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
axarr3[0].grid()

axarr3[0].set_ylabel('alpha [arb]')
axarr3[0].set_xlabel('lambda [m]')

axarr3[1].set_title('Diagonalized Data', fontsize=16)
axarr3[1].loglog(lambdas, simp_diagalphas, label='Naive: Sig = Noise')
axarr3[1].loglog(lambdas, diagalphas, linewidth=2, label='95% CL')
axarr3[1].loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
axarr3[1].loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
axarr3[1].grid()

axarr3[1].set_xlabel('lambda [m]')

axarr3[0].legend(numpoints=1, fontsize=9)
axarr3[1].legend(numpoints=1, fontsize=9)






plt.show()
