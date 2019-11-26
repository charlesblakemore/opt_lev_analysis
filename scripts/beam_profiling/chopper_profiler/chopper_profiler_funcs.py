import os, fnmatch

import numpy as np

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import scipy.optimize as opti
import scipy.special as special
import scipy.stats as stats

import peakdetect as pdet

import bead_util as bu
import configuration as config



def bessel_intensity(x, A, u):
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

    xvec = np.array(xvec)
    yvec = np.array(yvec)

    minval = np.min(xvec)
    maxval = np.max(xvec)
    dx = np.abs(xvec[1] - xvec[0])

    xarr = np.linspace(minval, maxval, numbins)

    yarr = np.zeros_like(xarr)
    err = np.zeros_like(xarr)

    dx2 = xarr[1] - xarr[0]
    for i, x in enumerate(xarr):
        inds = (xvec >= x - 0.5*dx2) * (xvec < x + 0.5*dx2)
        yarr[i] = np.mean(yvec[inds])
        err[i] = np.std(yvec[inds])
    '''
    xvec2 = np.hstack( (xvec[0] - dx, np.hstack( (xvec, xvec[-1] + dx) ) ) )
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
    '''
    
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
        print("FAILED THIS ONE")
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





def profile(df, raw_dat_col = 0, drum_diam=3.17e-2, return_pos=False, \
            numbins = 500, fit_intensity=False, \
            intensity_func = gauss_intensity, guess = 3.0e-3, \
            plot_peaks = False):
    ''' Takes a DataFile instance, extacts digitized data from the ThorLabs
    WM100 beam profiler, computes the derivative to find the profile then
    averages many profiles from a single time steam.
    
    INPUTS:  df, DataFile instance with profiles
             raw_dat_col, column in 'other_data' with raw WM100 monitor
             drum_diam, diameter of the optical head that rotates
             return_pos, boolean to specify if return in raw time or calibrated
                         drum position using the drum_diam argument

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

    if chopfreq > 15:
        chopfreq = 10.2

    grad = np.gradient(raw_dat)

    dt_chop = 1.0 / chopfreq
    numchops = int(t[-1] / dt_chop)
    twidth = (guess / (2.0 * np.pi * 10.0e-3)) * dt_chop

    peaks = pdet.peakdetect(grad, lookahead=50, delta=0.075)

    pos_peaks = peaks[0]
    neg_peaks = peaks[1]

    tot_prof = []
    tot_t = []

    if plot_peaks:
        for peakind, pos_peak in enumerate(pos_peaks):
            try:
                neg_peak = neg_peaks[peakind]
            except:
                continue
            plt.plot(t[pos_peak[0]], pos_peak[1], 'x', color='r')
            plt.plot(t[neg_peak[0]], neg_peak[1], 'x', color='b')
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
        print("Couldn't figure out positive or negative first...")

    if neg_first:
        pos_peaks = pos_peaks[:-1]
        neg_peaks = neg_peaks[1:]

    if len(pos_peaks) > len(neg_peaks):
        pos_peaks = pos_peaks[:-1]
    elif len(neg_peaks) > len(pos_peaks):
        neg_peaks = neg_peaks[1:]

    for ind, pos_peak in enumerate(pos_peaks):
        neg_peak = neg_peaks[ind]

        pos_peak_loc = pos_peak[0]
        neg_peak_loc = neg_peak[0]

        if pos_peak_loc < 500:
            continue

        fit_t = t[pos_peak_loc-500:neg_peak_loc+500]
        fit_prof = grad[pos_peak_loc-500:neg_peak_loc+500]

        try:
            new_t, new_prof = fit_gauss_and_truncate(fit_t, fit_prof, \
                                                     twidth, numbins = numbins)

            if len(tot_t) == 0:
                tot_t = new_t
                tot_prof = new_prof
            else:
                tot_t = np.hstack((tot_t, new_t))
                tot_prof = np.hstack((tot_prof, new_prof))

        except:
            print('Failed to fit and return result')

    sort_inds = tot_t.argsort()

    tot_d = 2 * np.pi * 10.2 * (drum_diam * 0.5) * tot_t

    new_t = tot_t[sort_inds]
    new_d = tot_d[sort_inds]
    new_prof = tot_prof[sort_inds]

    if fit_intensity:
        if return_pos:
            xvec = new_d
        else:
            xvec = new_t

        width = 0.2 * (np.max(xvec) - np.min(xvec))
        newguess = [np.max(new_prof), 0, width]

        try:
            popt, pcov = opti.curve_fit(intensity_func, xvec, \
                                        new_prof, p0 = newguess)
            return xvec, new_prof, popt
        except:
            print("Fit didn't work!")

    if return_pos:
        return new_d, new_prof
    else:
        return new_t, new_prof




def profile_directory(prof_dir, raw_dat_col = 0, drum_diam=3.25e-2, \
                      return_pos=False, plot_peaks=False):
    ''' Takes a directory path and profiles each file, and averages 
        for a final result
    
    INPUTS:  prof_dir, directory path
             raw_dat_col, column in 'other_data' with raw WM100 monitor
             drum_diam, diameter of the optical head that rotates
             return_pos, boolean to specify if return in raw time or calibrated
                         drum position using the drum_diam argument

    OUTPUTS: tot_x, all t/disp associated with profiles, overlain/sorted
             tot_prof, all profiles overlain and sorted
    '''
    prof_files = [] 
    for root, dirnames, filenames in os.walk(prof_dir):
        for filename in fnmatch.filter(filenames, '*' + config.extensions['data']):
            prof_files.append(os.path.join(root, filename))

    tot_x = []
    tor_prof = []
    nfiles = len(prof_files)
    for fil_ind, fil_path in enumerate(prof_files):
        bu.progress_bar(fil_ind, nfiles)
        prof_df = bu.DataFile()
        prof_df.load(fil_path, skip_fpga=True)
        prof_df.load_other_data()

        x, prof, popt = profile(prof_df, raw_dat_col = raw_dat_col, \
                                drum_diam = drum_diam, return_pos = return_pos, \
                                fit_intensity=True, plot_peaks=plot_peaks)


        #plt.plot(x, prof)
        #plt.show()

        #x, prof, errs = rebin(x, prof, numbins=5000)
        #plt.plot(x, prof)
        #plt.show()

        if not len(tot_x):
            tot_x = x
            tot_prof = prof
            tot_popt = [popt]
        else:
            tot_x = np.block([tot_x, x]) # = np.hstack((tot_x, x))
            tot_prof = np.block([tot_prof, prof]) # = np.hstack((tot_prof, prof))
            tot_popt.append(popt) # = np.concatenate((tot_popt, popt), axis=0)

    #tot_x = np.concatenate(tot_x)
    #tot_prof = np.concatenate(tot_x)

    tot_popt = np.array(tot_popt)
    tot_popt_mean = np.mean(tot_popt, axis=0)

    sort_inds = tot_x.argsort()

    return tot_x[sort_inds], tot_prof[sort_inds], tot_popt_mean
