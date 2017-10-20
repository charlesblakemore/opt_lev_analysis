import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize as opti
import scipy.special as special

import peakdetect as pdet

import bead_util as bu
import configuration as config


fontsize = 16


xfile = '/data/20171020/beam_profiling/xprof_notape.h5'

#xfile = '/data/20171018/chopper_profiling/xprof_bright_fast.h5'

yfile = '/data/20171020/beam_profiling/yprof_notape.h5'

#yfile = '/data/20171018/chopper_profiling/yprof_bright_fast.h5'

guess = 3.0e-3    # m, guess to help fit

xfilobj = bu.DataFile()
xfilobj.load(xfile)
xfilobj.load_other_data()

yfilobj = bu.DataFile()
yfilobj.load(yfile)
yfilobj.load_other_data()


def LP01_mode(x, A, u):
    return A * special.jv(0, x * u)**2


def gauss_intensity(x, A, xo, w):
    '''Formula for the intensity of a gaussian beam
    with an arbitrary amplitude and mean, but a waist
    parameter defined as usual, and correctly for the
    intensity (rather than E-field).'''
    return A * np.exp( -2.0 * (x - xo)**2 / (w**2) )


def rebin(xvec, yvec, numbins=100):
    '''Use a cubic interpolating spline to rebin data

           INPUTS: xvec, x-axis vector to bin against
                   yvec, response to binned vector
                   numbins, number of bins to divide xvec into
                       
           OUTPUTS: xarr, rebinned x-axis
                    yarr, rebinned response
                    err, errors on rebinned response'''

    #print numbins

    minval = np.min(xvec)
    maxval = np.max(xvec)
    dx = np.abs(xvec[1] - xvec[0])

    xvec2 = np.hstack( (xvec[0] - 2.*dx, np.hstack( (xvec, xvec[-1] + 2.*dx) ) ) )
    yvec2 = np.hstack( (yvec[0], np.hstack( (yvec, yvec[-1]) ) ) )

    if numbins % 2:
        numbins += 1

    dx2 = (maxval - minval + dx) / float(numbins)
    new_minval = minval - 0.5*dx + 0.5*dx2
    new_maxval = maxval + 0.5*dx - 0.5*dx2
    xarr = np.linspace(new_minval, new_maxval, numbins)

    yarr = np.zeros_like(xarr)
    err = np.zeros_like(xarr)

    # Generate new yvals and new error bars
    for i, x in enumerate(xarr):
        inds = np.abs(xvec - x) < 0.5*dx2
        yarr[i] = np.mean( yvec[inds] )
        err[i] = np.std( yvec[inds] )
    
    return xarr, yarr, err




def fit_gauss_and_truncate(t, prof, twidth, numbins = 500):
    ''' Takes a single profile (positive and negative going) from
    a raw monitor trace of the ThorLbas WM100 chopper beam profiler.
    Assumes the postive and negative going profiles are centered,
    separates them, fits a gaussian to find the centers, then 
    offsets and truncates each one to have the same number of points
    
    INPUTS:  t, time vector
             prof, data vector with profiles
             twidth, time width guess of profile for fitting
             numbins, number of samples for final profile output

    OUTPUTS: new_t, new time vector for both profiles
             new_prof, new profile vector
    '''

    lenprof = int(len(prof) * 0.5)
    pos_prof = prof[:lenprof]
    pos_t = t[:lenprof]
    neg_prof = -1.0 * prof[lenprof:]
    neg_t = t[lenprof:]

    pos_p0 = [0.1, pos_t[np.argmax(pos_prof)], twidth] 
    neg_p0 = [0.1, neg_t[np.argmax(neg_prof)], twidth] 

    try:
        pos_popt, pos_pcov = opti.curve_fit(gauss_intensity, pos_t, pos_prof, \
                                            p0=pos_p0, maxfev=10000)
        neg_popt, neg_pcov = opti.curve_fit(gauss_intensity, neg_t, neg_prof, \
                                            p0=neg_p0, maxfev=10000)
    
    except:
        print "FAILED THIS ONE"
        plt.plot(pos_t, pos_prof)
        plt.plot(neg_t, neg_prof)
        plt.show()

    pos_cent_bin = np.argmin( np.abs(pos_t - pos_popt[1]) )
    neg_cent_bin = np.argmin( np.abs(neg_t - neg_popt[1]) )

    new_pos_bins = (pos_cent_bin-numbins/2, pos_cent_bin+numbins/2)
    new_neg_bins = (neg_cent_bin-numbins/2, neg_cent_bin+numbins/2)

    new_pos_t = pos_t[new_pos_bins[0]:new_pos_bins[1]] - pos_popt[1]
    new_neg_t = neg_t[new_neg_bins[0]:new_neg_bins[1]] - neg_popt[1]

    new_pos_prof = pos_prof[new_pos_bins[0]:new_pos_bins[1]]
    new_neg_prof = neg_prof[new_neg_bins[0]:new_neg_bins[1]]

    tot_t = np.hstack((new_pos_t, new_neg_t))
    tot_prof = np.hstack((new_pos_prof, new_neg_prof))

    sort_inds = tot_t.argsort()

    return tot_t[sort_inds], tot_prof[sort_inds]





def profile(df, raw_dat_col = 4, low_thresh = -0.05, high_thresh = 0.05):
    ''' Takes a DataFile instance, extacts digitized data from the ThorLabs
    WM100 beam profiler, computes the derivative to find the profile then
    averages many profiles from a single time steam.
    
    INPUTS:  df, DataFile instance with
             raw_dat_col, column in 'other_data' with raw WM100 monitor
             low_thresh, lower threshold to find negative going profile
             high_thresh, higher threshold to find positive going profile

    OUTPUTS: all_t, all times associated with profiles, overlain/sorted
             all_prof, all profiles overlain and sorted
    '''

    raw_dat = df.other_data[raw_dat_col]

    numpoints = len(raw_dat)
    fsamp = df.fsamp
    dt = 1.0 / fsamp
    t = np.linspace( 0, (numpoints - 1) * dt, numpoints ) 

    psd, freqs = mlab.psd(raw_dat, NFFT=len(raw_dat), Fs=fsamp)
    chopfreq = freqs[np.argmax(psd)]

    grad = np.gradient(raw_dat)

    dt_chop = 1.0 / chopfreq
    numchops = int(t[-1] / dt_chop)
    twidth = (guess / (2.0 * np.pi * 10.0e-3)) * dt_chop

    peaks = pdet.peakdetect(grad, lookahead=50, delta=0.1)

    pos_peaks = peaks[0]
    neg_peaks = peaks[1]

    tot_prof = []
    tot_t = []

    plt.plot(t, grad)
    plt.show()

    # since the chopper and ADC aren't triggered together and don't
    # have the same timebase, need to make sure only have nice pairs 
    # of peaks so we can look at forward going and backward going
    # separately. Since we know positive going should be first
    # this is quite easy to accomplish
    if neg_peaks[0][0] < pos_peaks[0][0]:
        neg_first = True
    elif neg_peaks[0][0] > pos_peaks[0][0]:
        neg_first = False
    else:
        print "Couldn't figure out positive or negative first..."
    if neg_first:
        pos_peaks = pos_peaks[:-1]
        neg_peaks = neg_peaks[1:]
    

    for ind, pos_peak in enumerate(pos_peaks):
        neg_peak = neg_peaks[ind]

        pos_peak_loc = pos_peak[0]
        neg_peak_loc = neg_peak[0]

        fit_t = t[pos_peak_loc-500:neg_peak_loc+500]
        fit_prof = grad[pos_peak_loc-500:neg_peak_loc+500]

        try:
            new_t, new_prof = fit_gauss_and_truncate(fit_t, fit_prof, \
                                                 twidth, numbins = 500)

            if len(tot_t) == 0:
                tot_t = new_t
                tot_prof = new_prof
            else:
                tot_t = np.hstack((tot_t, new_t))
                tot_prof = np.hstack((tot_prof, new_prof))

        except:
            print 'Failed to fit and return result'

    sort_inds = tot_t.argsort()

    return tot_t[sort_inds], tot_prof[sort_inds]


x_t, x_prof = profile(xfilobj, raw_dat_col = 4)
y_t, y_prof = profile(yfilobj, raw_dat_col = 4)

x_prof = x_prof / np.max(x_prof)
y_prof = y_prof / np.max(y_prof)

x_d = 2 * np.pi * 10.2 * (3.25e-2 * 0.5) * x_t
y_d = 2 * np.pi * 10.2 * (3.25e-2 * 0.5) * y_t

plt.plot(x_d, x_prof)
plt.show()

binned_x_d, binned_x_prof, x_errs = rebin(x_d, x_prof, numbins=500)
binned_y_d, binned_y_prof, y_errs = rebin(y_d, y_prof, numbins=500)

final_p0 = [1.0, 0, 0.001]
final_x_popt, final_x_pcov = opti.curve_fit(gauss_intensity, x_d, x_prof, \
                                        p0 = final_p0)
final_y_popt, final_y_pcov = opti.curve_fit(gauss_intensity, y_d, y_prof, \
                                        p0 = final_p0)

LP_p0 = [1, 0.001]
final_x_LP_popt, final_x_LP_pcov = \
        opti.curve_fit(LP01_mode, binned_x_d, binned_x_prof, p0=LP_p0)



fig1, axarr1 = plt.subplots(2,1,sharex=True,sharey=True)
axarr1[0].plot(x_d * 1e3, x_prof, label="All Data")
axarr1[0].plot(binned_x_d * 1e3, binned_x_prof, label="Avg'd Data", color='k')
axarr1[0].plot(binned_x_d * 1e3, gauss_intensity(binned_x_d, *final_x_popt),\
                   '--', color = 'r', linewidth=1.5, label="Gaussian Fit")
axarr1[0].set_xlabel("Displacement [mm]", fontsize=fontsize)
axarr1[0].set_ylabel("X Intensity [arb]", fontsize=fontsize)
axarr1[0].legend(fontsize=fontsize-4, loc=1)
plt.setp(axarr1[0].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr1[0].get_yticklabels(), fontsize=fontsize, visible=True)

axarr1[1].plot(y_d * 1e3, y_prof)
axarr1[1].plot(binned_y_d * 1e3, binned_y_prof, color='k')
axarr1[1].plot(binned_y_d * 1e3, gauss_intensity(binned_y_d, *final_y_popt),\
                   '--', color = 'r', linewidth=1.5)
axarr1[1].set_xlabel("Displacement [mm]", fontsize=fontsize)
axarr1[1].set_ylabel("Y Intensity [arb]", fontsize=fontsize)
plt.setp(axarr1[1].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr1[1].get_yticklabels(), fontsize=fontsize, visible=True)



fig2, axarr2 = plt.subplots(2,1,sharex=True,sharey=True)
axarr2[0].semilogy(binned_x_d * 1e3, np.abs(binned_x_prof), \
                   label="Avg'd Data", color='k')
axarr2[0].semilogy(binned_x_d * 1e3, gauss_intensity(binned_x_d, *final_x_popt),\
                   '--', color = 'r', linewidth=1.5, label="Gaussian Fit")
#axarr2[0].semilogy(binned_x_d * 1e3, LP01_mode(binned_x_d, *final_x_LP_popt),\
#                   '--', color = 'g', linewidth=1.5, label="LP Fit")
axarr2[0].set_xlabel("Displacement [mm]", fontsize=fontsize)
axarr2[0].set_ylabel("X Intensity [arb]", fontsize=fontsize)
axarr2[0].set_ylim(1e-4,3)
axarr2[0].legend(fontsize=fontsize-4, loc=1)
plt.setp(axarr2[0].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr2[0].get_yticklabels(), fontsize=fontsize, visible=True)

axarr2[1].semilogy(binned_y_d * 1e3, np.abs(binned_y_prof), color='k')
axarr2[1].semilogy(binned_y_d * 1e3, gauss_intensity(binned_y_d, *final_y_popt),\
                   '--', color = 'r', linewidth=1.5)
axarr2[1].set_xlabel("Displacement [mm]", fontsize=fontsize)
axarr2[1].set_ylabel("Y Intensity [arb]", fontsize=fontsize)
axarr2[0].set_ylim(1e-4,3)
plt.setp(axarr2[1].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr2[1].get_yticklabels(), fontsize=fontsize, visible=True)



plt.show()





