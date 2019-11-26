import sys, time

import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu
import configuration as config

import scipy.signal as signal
import scipy.optimize as optimize

import sklearn.cluster as cluster

#######################################################
# Core module for handling calibrations, both the step 
# calibrations and freq-dependent TF calibrations
#
# Main data type is a BeadCalibration class
#######################################################


def step_fun(x, q, x0):
    '''Single, decreasing step function
           INPUTS: x, variable
                   q, size of step
                   x0, location of step

           OUTPUTS: q * (x <= x0)'''
    xs = np.array(x)
    return q*(xs<=x0)


def multi_step_fun(x, qs, x0s):
    '''Sum of many single, decreasing step functions
           INPUTS: x, variable
                   qs, sizes of steps
                   x0s, locations of steps

           OUTPUTS: SUM_i [qi * (x <= x0i)]'''
    rfun = 0.
    for i, x0 in enumerate(x0s):
        rfun += step_fun(x, qs[i], x0)
    return rfun



def correlation(drive, response, fsamp, fdrive, filt = False, band_width = 1):
    '''Compute the full correlation between drive and response,
       correctly normalized for use in step-calibration.

       INPUTS:   drive, drive signal as a function of time
                 response, resposne signal as a function of time
                 fsamp, sampling frequency
                 fdrive, predetermined drive frequency
                 filt, boolean switch for bandpass filtering
                 band_width, bandwidth in [Hz] of filter

       OUTPUTS:  corr_full, full and correctly normalized correlation'''

    # First subtract of mean of signals to avoid correlating dc
    drive = drive-np.mean(drive)
    response = response-np.mean(response)

    # bandpass filter around drive frequency if desired.
    if filt:
        b, a = signal.butter(3, [2.*(fdrive-band_width/2.)/fsamp, \
                             2.*(fdrive+band_width/2.)/fsamp ], btype = 'bandpass')
        drive = signal.filtfilt(b, a, drive)
        response = signal.filtfilt(b, a, response)
    
    # Compute the number of points and drive amplitude to normalize correlation
    lentrace = len(drive)
    drive_amp = np.sqrt(2)*np.std(drive)

    # Define the correlation vector which will be populated later
    corr = np.zeros(int(fsamp/fdrive))

    # Zero-pad the response
    response = np.append(response, np.zeros(int(fsamp / fdrive) - 1) )

    # Build the correlation
    n_corr = len(drive)
    for i in range(len(corr)):
        # Correct for loss of points at end
        correct_fac = 2.0*n_corr/(n_corr-i) # x2 from empirical test
        corr[i] = np.sum(drive*response[i:i+n_corr])*correct_fac

    return corr * (1.0 / (lentrace * drive_amp))



def find_step_cal_response(file_obj, bandwidth=1., include_in_phase=False, \
                           using_tabor=False, tabor_ind=3, mon_fac=100):
    '''Analyze a data step-calibraiton data file, find the drive frequency,
       correlate the response to the drive

       INPUTS:   file_obj, input file object
                 bandwidth, bandpass filter bandwidth

       OUTPUTS:  H, (response / drive)'''

    if not using_tabor:
        ecol = np.argmax(file_obj.electrode_settings['driven'])
        pcol = config.elec_map[ecol]

        # Extract the drive, detrend it, and compute an fft
        #drive = file_obj.electrode_data[ecol]
        drive = bu.trap_efield(file_obj.electrode_data)[pcol]
    elif using_tabor:
        pcol = 0
        v3 = file_obj.other_data[tabor_ind]
        v4 = file_obj.other_data[tabor_ind+1]
        zeros = np.zeros(len(v3))
        drive = bu.trap_efield([zeros, zeros, zeros, v3, v4, zeros, zeros, zeros])[pcol]

    power = np.mean(file_obj.other_data[0])
    zpos = np.mean(file_obj.pos_data[2])

    drive = signal.detrend(drive)
    drive_fft = np.fft.rfft(drive)

    # Find the drive frequency
    freqs = np.fft.rfftfreq(len(drive), d=1./file_obj.fsamp)
    drive_freq = freqs[np.argmax(np.abs(drive_fft))]

    # Extract the response and detrend
    response = file_obj.pos_data[pcol]
    response = signal.detrend(response)

    # Configure a time array for plotting and fitting
    cut_samp = config.adc_params["ignore_pts"]
    N = len(drive)
    dt = 1. / file_obj.fsamp
    t = np.linspace(0,(N+cut_samp-1)*dt, N+cut_samp)
    t = t[cut_samp:]

    # Bandpass filter the response
    b, a = signal.butter(3, [2.*(drive_freq-bandwidth/2.)/file_obj.fsamp, \
                          2.*(drive_freq+bandwidth/2.)/file_obj.fsamp ], btype = 'bandpass')
    responsefilt = signal.filtfilt(b, a, response)

    ### CORR_FUNC TESTING ###
    #test = 3.14159 * np.sin(2 * np.pi * drive_freq * t)
    #test_corr = correlation(7 * drive, test, file_obj.fsamp, drive_freq)
    #print np.sqrt(2) * np.std(test)
    #print np.max(test_corr)
    #########################

    # Compute the full, normalized correlation and extract amplitude
    corr_full = correlation(drive, response, file_obj.fsamp, drive_freq)
    #corr_full = correlation(drive, responsefilt, file_obj.fsamp, drive_freq)

    response_amp = np.max(corr_full)
    #response_amp2 = corr_full[0]
    #response_amp3 = np.sqrt(2) * np.std(responsefilt)

    # Compute the drive amplitude. Two methods included, should decide on one 
    drive_amp = np.sqrt(2) * np.std(drive) # Assume drive is sinusoidal

    def drive_fun(x, A, f, phi):
        return A * np.sin( 2 * np.pi * f * x + phi )
        
    # Estimate some parameters and try fitting a sine
    #p0_drive = [drive_amp, drive_freq, 0]
    #popt, pcov = optimize.curve_fit(drive_fun, t, drive, p0=p0_drive)

    #drive_amp2 = popt[0]

    #print drive_amp, drive_amp2

    # Include the possibility of a different sign of response
    sign = np.sign(np.mean(drive*responsefilt))

    return response_amp / drive_amp, power, zpos





def step_cal(step_cal_vec, plate_sep = 0.004, drive_freq = 41., \
             amp_gain = 1., bandwidth=1.0, first_file=0):
    '''Generates a step calibration from a list of DataFile objects
           INPUTS: fobjs, list of file objects
                   plate_sep, face-to-face separation of electrodes
                   drive_freq, electrostatic drive freq during step_cal
                   amp_gain, gain of HV amplifier if noise is a problem

           OUTPUTS: vpn, volts of response per Newton of drive
                    err, 1 std.dev. error on vpn'''


    step_cal_vec = np.array(step_cal_vec)
    
    yfit = np.abs(step_cal_vec)
    #bvec = yfit == yfit #[yfit<10.*np.mean(yfit)] #exclude cray outliers
    #yfit = yfit[bvec] 

    plt.figure(1)
    plt.ion()
    plt.plot(yfit, 'o')
    plt.show()
    guess = input('Enter a guess for volt / step: ')
    guess = float(guess)
    #plt.ioff()
    plt.close(1)

    step_inds = []
    step_qs = []
    step_sizes = []
    last_step = 0

    for i in range(len(yfit)):

        if i == 0:
            current_charge = [yfit[0]]
            continue

        std = np.std(yfit[last_step+1:i-1])

        diff = np.mean(current_charge) - yfit[i]
        diff_abs = np.abs(diff)

        #if len(step_sizes) > 0:
        #    guess = np.mean(step_sizes)

        if (diff_abs > 0.75 * guess) and (diff_abs > 2 * std):
            #big = diff_abs > 4 * guess
            #zero = (diff_abs - np.mean(current_charge)) < (2 * std)

            #if big and zero:
            #    continue
            
            current_charge = [yfit[i]]
            
            last_step = i-1

            if diff_abs > 4.5 * guess:
                step_sizes.append(diff_abs * 0.2)
                step_qs.append(np.sign(diff) * 5)
            elif diff_abs > 3.5 * guess:
                step_sizes.append(diff_abs * 0.25)
                step_qs.append(np.sign(diff) * 4)
            elif diff_abs > 2.5 * guess:
                step_sizes.append(diff_abs * 0.33333333)
                step_qs.append(np.sign(diff) * 3)
            elif diff_abs > 1.5 * guess:
                step_sizes.append(diff_abs * 0.5)
                step_qs.append(np.sign(diff) * 2)
            else:
                step_sizes.append(diff_abs)
                step_qs.append(np.sign(diff) * 1)

            step_inds.append(last_step)
        else:
            current_charge.append(yfit[i])

    vpq_guess = np.mean(step_sizes)

    def ffun(x, vpq, offset):
        qqs = vpq * np.array(step_qs)
        offarr = np.zeros(len(x)) + offset
        return multi_step_fun(x, qqs, step_inds) + offarr
    
    xfit = np.arange(len(yfit))

    p0 = [vpq_guess, 0]#Initial guess for the fit

    popt, pcov = optimize.curve_fit(ffun, xfit, yfit, p0 = p0, xtol = 1e-12)

    fitobj = Fit(popt, pcov, ffun)

    newpopt = np.copy(popt)
    newpopt[1] = 0.0

    normfitobj = Fit(newpopt / popt[0], pcov / popt[0], ffun)

    f, axarr = plt.subplots(2, sharex = True, \
                            gridspec_kw = {'height_ratios':[2,1]}, \
                            figsize=(10,4),dpi=150)#Plot fit
    normfitobj.plt_fit(xfit, (yfit - popt[1]) / popt[0], \
                       axarr[0], ylabel="Normalized Response [e]", xlabel="")
    normfitobj.plt_residuals(xfit, (yfit - popt[1]) / popt[0], axarr[1], \
                             xlabel="Integration Number")
    for x in xfit:
        if not (x-1) % 3:
            axarr[0].axvline(x=x, color='k', linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()

    happy = input("does the fit look good? (y/n): ")
    if happy == 'y':
        happy_with_fit = True
    elif happy == 'n':
        happy_with_fit = False
        f.clf()
    else:
        happy_with_fit = False
        f.clf()
        print('that was a yes or no question... assuming you are unhappy')
        sys.stdout.flush()
        time.sleep(5)


    while not happy_with_fit:
        plt.figure(1)
        plt.ion()
        plt.plot(yfit, 'o')
        plt.show()

        print("MANUAL STEP CALIBRATION")
        print("Enter guess at number of steps and charge at steps [[q1, q2, q3, ...], [x1, x2, x3, ...], vpq]")
        nstep = eval(input(": "))

        step_qs = nstep[0]
        step_inds = nstep[1]

        p0 = [nstep[2],0.0]#Initial guess for the fit
        popt, pcov = optimize.curve_fit(ffun, xfit, yfit, p0 = p0, xtol = 1e-10)

        fitobj = Fit(popt, pcov, ffun)#Store fit in object.

        newpopt = np.copy(popt)
        newpopt[1] = 0.0

        normfitobj = Fit(newpopt / popt[0], pcov / popt[0], ffun)

        plt.close(1)
        f, axarr = plt.subplots(2, sharex = True, \
                                gridspec_kw = {'height_ratios':[2,1]})#Plot fit
        normfitobj.plt_fit(xfit, (yfit - popt[1]) / popt[0], \
                           axarr[0], ylabel="Normalized Response [e]", xlabel="")
        normfitobj.plt_residuals(xfit, (yfit - popt[1]) / popt[0], axarr[1], \
                                 xlabel="Integration Number")
        plt.show()

        happy = input("does the fit look good? (y/n): ")
        if happy == 'y':
            happy_with_fit = True
        elif happy == 'n':
            f.clf()
            continue
        else:
            f.clf()
            print('that was a yes or no question... assuming you are unhappy')
            sys.stdout.flush()
            time.sleep(5)
            continue

    plt.ioff()

    print(fitobj.popt[0])

    q0_sc = ffun([0], *fitobj.popt)[0]
    q0 = int(round(q0_sc / fitobj.popt[0]))
    print('q0: ', q0)

    #Determine force calibration.
    e_charge = config.p_param['e_charge']
    #fitobj.popt = fitobj.popt * 1./(amp_gain*e_charge/plate_sep)
    #fitobj.errs = fitobj.errs * 1./(amp_gain*e_charge/plate_sep)

    fitobj.popt = fitobj.popt * (1.0 / e_charge)
    fitobj.errs = fitobj.errs * (1.0 / e_charge)

    return fitobj.popt[0], fitobj.popt[1], fitobj.errs[0], q0








class Fit:
    # Holds the optimal parameters and errors from a fit. 
    # Contains methods to plot the fit, the fit data, and the residuals.
    def __init__(self, popt, pcov, fun):
        self.popt = popt
        try:
            self.errs = pcov.diagonal()
        except ValueError:
            self.errs = "Fit failed"
        self.fun = fun

    def plt_fit(self, xdata, ydata, ax, scale = 'linear', xlabel = 'X', ylabel = 'Y', errors = []):
    
        inds = np.argsort(xdata)
        
        xdata = xdata[inds]
        ydata = ydata[inds]

        #modifies an axis object to plot the fit.
        if len(errors):
            ax.errorbar(xdata, ydata, errors, fmt = 'o')
            ax.plot(xdata, self.fun(xdata, *self.popt), 'r', linewidth = 3)

        else:    
            ax.plot(xdata, ydata, 'o')
            ax.plot(xdata, self.fun(xdata, *self.popt), 'r', linewidth = 3)

        ax.set_yscale(scale)
        ax.set_xscale(scale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([np.min(xdata), np.max(xdata)])
    
    def plt_residuals(self, xdata, ydata, ax, scale = 'linear', xlabel = 'X', ylabel = 'Residual', label = '', errors = []):
        #modifies an axis object to plot the residuals from a fit.

        inds = np.argsort(xdata)
        
        xdata = xdata[inds]
        ydata = ydata[inds]

        #print np.std( self.fun(xdata, *self.popt) - ydata )

        if len(errors):
            ax.errorbar(xdata, self.fun(xdata, *self.popt) - ydata, errors, fmt = 'o')
        else:
            ax.plot(xdata, (self.fun(xdata, *self.popt) - ydata), 'o')
        
        #ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([np.min(xdata), np.max(xdata)])

    def css(self, xdata, ydata, yerrs, p):
        #returns the chi square score at a point in fit parameters.
        return np.sum((ydata))
