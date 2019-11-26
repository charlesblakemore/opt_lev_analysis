import glob, os, sys, copy, time, math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy
import scipy.optimize as opti
import scipy.signal as signal
import scipy.interpolate as interp

import bead_util as bu
import image_util as imu
import configuration as config

import dill as pickle





def damped_osc_amp(f, A, f0, g):
    '''Fitting function for AMPLITUDE of a damped harmonic oscillator
           INPUTS: f [Hz], frequency 
                   A, amplitude
                   f0 [Hz], resonant frequency
                   g [Hz], damping factor

           OUTPUTS: Lorentzian amplitude'''
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    denom = np.sqrt((w0**2 - w**2)**2 + w**2 * g**2)
    return A / denom

def damped_osc_phase(f, A, f0, g, phase0 = 0.):
    '''Fitting function for PHASE of a damped harmonic oscillator.
       Includes an arbitrary DC phase to fit over out of phase responses
           INPUTS: f [Hz], frequency 
                   A, amplitude
                   f0 [Hz], resonant frequency
                   g [Hz], damping factor

           OUTPUTS: Lorentzian amplitude'''
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    return A * np.arctan2(-w * g, w0**2 - w**2) + phase0



def sum_3osc_amp(f, A1, f1, g1, A2, f2, g2, A3, f3, g3):
    '''Fitting function for AMPLITUDE of a sum of 3 damped harmonic oscillators.
           INPUTS: f [Hz], frequency 
                   A1,2,3, amplitude of the three oscillators
                   f1,2,3 [Hz], resonant frequency of the three oscs
                   g1,2,3 [Hz], damping factors

           OUTPUTS: Lorentzian amplitude of complex sum'''

    csum = damped_osc_amp(f, A1, f1, g1)*np.exp(1.j * damped_osc_phase(f, A1, f1, g1) ) \
           + damped_osc_amp(f, A2, f2, g2)*np.exp(1.j * damped_osc_phase(f, A2, f2, g2) ) \
           + damped_osc_amp(f, A3, f3, g3)*np.exp(1.j * damped_osc_phase(f, A3, f3, g3) )
    return np.abs(csum)


def sum_3osc_phase(f, A1, f1, g1, A2, f2, g2, A3, f3, g3, phase0=0.):
    '''Fitting function for PHASE of a sum of 3 damped harmonic oscillators.
       Includes an arbitrary DC phase to fit over out of phase responses
           INPUTS: f [Hz], frequency 
                   A1,2,3, amplitude of the three oscillators
                   f1,2,3 [Hz], resonant frequency of the three oscs
                   g1,2,3 [Hz], damping factors

           OUTPUTS: Lorentzian phase of complex sum'''

    csum = damped_osc_amp(f, A1, f1, g1)*np.exp(1.j * damped_osc_phase(f, A1, f1, g1) ) \
           + damped_osc_amp(f, A2, f2, g2)*np.exp(1.j * damped_osc_phase(f, A2, f2, g2) ) \
           + damped_osc_amp(f, A3, f3, g3)*np.exp(1.j * damped_osc_phase(f, A3, f3, g3) )
    return np.angle(csum) + phase0




def ipoly1d_func(x, *params):
    '''inverse polynomial function to fit against

           INPUTS: x, independent variable
                   params, N-parameter array

           OUTPUTS: ip(x) = params[0] * (x) ** (-deg) + ... 
                                             + params[deg-1] * (x) ** -1
    '''
    out = x - x  
    deg = len(params)
    for ind, p in enumerate(params):
        out += np.abs(p) * (x)**(ind - deg)
    return out


def ipoly1d(ipolyparams):
    return lambda x: ipoly1d_func(x, *ipolyparams)



def ipolyfit(xs, ys, deg):
    mean = np.mean(ys)
    params = np.array([mean * 0.001 for p in range(deg)])
    popt, _ = opti.curve_fit(ipoly1d_func, xs, ys, p0=params, maxfev=10000)
    return popt



def make_extrapolator(interpfunc, pts=10, order=1, inverse=(False, False)):
    '''Make a functional object that does nth order polynomial extrapolation
       of a scipy.interpolate.interp1d object (should also work for other 1d
       interpolating objects).

           INPUTS: interpfunc, inteprolating function to extrapolate
                   pts, points to include in linear regression
                   order, order of the polynomial to use in extrapolation
                   inverse, boolean specifying whether to use inverse poly
                            1st index for lower range, 2nd for upper

           OUTPUTS: extrapfunc, function object with extrapolation
    '''
    
    xs = interpfunc.x
    ys = interpfunc.y

    if inverse[0]:
        lower_params = ipolyfit(xs[:pts], ys[:pts], order)
        lower = ipoly1d(lower_params)
                
    else:
        lower_params = np.polyfit(xs[:pts], ys[:pts], order)
        lower = np.poly1d(lower_params)

    if inverse[1]:
        upper_params = ipolyfit(xs[-pts:], ys[-pts:], order)
        upper = ipoly1d(upper_params) 
    else:
        upper_params = np.polyfit(xs[-pts:], ys[-pts:], order)
        upper = np.poly1d(upper_params) 

    def extrapfunc(x):

        ubool = x >= xs[-1]
        lbool = x <= xs[0]

        midval = interpfunc( x[ np.invert(ubool + lbool) ] )
        uval = upper( x[ubool] )
        lval = lower( x[lbool] )

        return np.concatenate((lval, midval, uval))

    return extrapfunc
        
        

    


def build_uncalibrated_H(fobjs, average_first=True, dpsd_thresh = 8e-1, mfreq = 1., \
                         plot_qpd_response=False, drop_bad_bins=True):
    '''Generates a transfer function from a list of DataFile objects
           INPUTS: fobjs, list of file objects
                   average_first, boolean specifying whether to average responses
                                  for a given drive before computing H
                   dpsd_thresh, threshold above which to compute H
                   mfreq, minimum frequency to consider
                   fix_HF, boolean to specify whether to try fixing spectral
                           leakage at high frequency due to drift in a timebase

           OUTPUTS: Hout, dictionary with 3x3 complex valued matrices as values
                          and frequencies as keys'''

    print("BUILDING H...")
    sys.stdout.flush()

    Hout = {}
    Hout_noise = {}

    Hout_amp = {}
    Hout_phase = {}
    Hout_fb = {}

    Hout_counts = {}


    avg_drive_fft = {}
    avg_data_fft = {}
    avg_fb_fft = {}
    avg_side_fft = {}

    avg_amp_fft = {}
    avg_phase_fft = {}

    counts = {}

    if plot_qpd_response:
        ampfig, amp_axarr = plt.subplots(5,3,sharex=True,sharey=True,dpi=150,figsize=(6,8))
        phasefig, phase_axarr = plt.subplots(5,3,sharex=True,sharey=True,dpi=150,figsize=(6,8))
        sidefig, side_axarr = plt.subplots(4,3,sharex=True,sharey=True,dpi=150,figsize=(6,7))
        fbfig, fb_axarr = plt.subplots(3,3,sharex=True,sharey=True,dpi=150,figsize=(6,6))
        posfig, pos_axarr = plt.subplots(3,3,sharex=True,sharey=True,dpi=150,figsize=(6,6))
        drivefig, drive_axarr = plt.subplots(8,3,sharex=True,sharey=True,dpi=150,figsize=(6,8))

    filind = 0
    for fobj in fobjs:

        dfft = np.fft.rfft(fobj.electrode_data) #fft of electrode drive in daxis. 
        data_fft = np.fft.rfft(fobj.pos_data)
        amp_fft = np.fft.rfft(fobj.amp)
        phase_fft = np.fft.rfft(fobj.phase)
        fb_fft = np.fft.rfft(fobj.pos_fb)

        left = fobj.amp[2] + fobj.amp[3]
        right = fobj.amp[0] + fobj.amp[1]
        top = fobj.amp[2] + fobj.amp[0]
        bot = fobj.amp[3] + fobj.amp[1]
        side_fft = np.fft.rfft(np.array([right, left, top, bot]))

        N = np.shape(fobj.pos_data)[1]#number of samples
        fsamp = fobj.fsamp

        fft_freqs = np.fft.rfftfreq(N, d=1.0/fsamp)

        dpsd = np.abs(dfft)**2 * 2./(N*fobj.fsamp) #psd for all electrode drives   
        inds = np.where(dpsd>dpsd_thresh)#Where the dpsd is over the threshold for being used.
        eind = np.unique(inds[0])[0]

        if eind not in avg_drive_fft:
            avg_drive_fft[eind] = np.zeros(dfft.shape, dtype=np.complex128)
            avg_data_fft[eind] = np.zeros(data_fft.shape, dtype=np.complex128)
            avg_amp_fft[eind] = np.zeros(amp_fft.shape, dtype=np.complex128)
            avg_phase_fft[eind] = np.zeros(phase_fft.shape, dtype=np.complex128)
            avg_fb_fft[eind] = np.zeros(fb_fft.shape, dtype=np.complex128)
            avg_side_fft[eind] = np.zeros(side_fft.shape, dtype=np.complex128)
            counts[eind] = 0.

        avg_drive_fft[eind] += dfft
        avg_data_fft[eind] += data_fft
        avg_amp_fft[eind] += amp_fft
        avg_phase_fft[eind] += phase_fft
        avg_fb_fft[eind] += fb_fft
        avg_side_fft[eind] += side_fft

        counts[eind] += 1.

    for eind in list(counts.keys()):
        avg_drive_fft[eind] = avg_drive_fft[eind] / counts[eind]
        avg_data_fft[eind] = avg_data_fft[eind] / counts[eind]
        avg_amp_fft[eind] = avg_amp_fft[eind] / counts[eind]
        avg_phase_fft[eind] = avg_phase_fft[eind] / counts[eind]
        avg_fb_fft[eind] = avg_fb_fft[eind] / counts[eind]
        avg_side_fft[eind] = avg_side_fft[eind] / counts[eind]

    poslabs = {0: 'X', 1: 'Y', 2: 'Z'}
    sidelabs = {0: 'Right', 1: 'Left', 2: 'Top', 3: 'Bottom'}
    quadlabs = {0: 'Top Right', 1: 'Bottom Right', 2: 'Top Left', 3: 'Bottom Left', 4: 'Backscatter'}


    for eind in list(avg_drive_fft.keys()):
        # First find drive-frequency bins above a fixed threshold
        dpsd = np.abs(avg_drive_fft[eind])**2 * 2. / (N*fsamp)
        inds = np.where(dpsd > dpsd_thresh)

        # Extract the frequency indices
        finds = inds[1]

        # Ignore DC and super low frequencies
        mfreq = 1.0
        b = finds > np.argmin(np.abs(fft_freqs - mfreq))

        freqs = fft_freqs[finds[b]]

        # Compute FFT of each response divided by FFT of each drive.
        # This is way more information than we need for a single drive freq
        # and electrode pair, but it allows a nice vectorization
        Hmatst = np.einsum('ij, kj -> ikj', \
                             avg_data_fft[eind], 1. / avg_drive_fft[eind])

        outind = config.elec_map[eind]

        if plot_qpd_response:
            
            for elec in [0,1,2,3,4,5,6,7]:
                drive_axarr[elec,outind].loglog(fft_freqs, \
                                                np.abs(avg_drive_fft[eind][elec]), alpha=0.75)
                drive_axarr[elec,outind].loglog(fft_freqs[inds[1]], \
                                                np.abs(avg_drive_fft[eind][elec])[inds[1]], alpha=0.75)
                if outind == 0:
                    drive_axarr[elec,outind].set_ylabel('Elec ' + str(elec))
                if elec == 7:
                    drive_axarr[elec,outind].set_xlabel('Frequency [Hz]')


            for resp in [0,1,2,3,4]:
                amp_axarr[resp,outind].loglog(fft_freqs[inds[1]], \
                                              np.abs(avg_amp_fft[eind][resp])[inds[1]], alpha=0.75)
                phase_axarr[resp,outind].loglog(fft_freqs[inds[1]], \
                                                np.abs(avg_phase_fft[eind][resp])[inds[1]], alpha=0.75)
                if outind == 0:
                    amp_axarr[resp,outind].set_ylabel(quadlabs[resp])
                    phase_axarr[resp,outind].set_ylabel(quadlabs[resp])
                if resp == 4:
                    amp_axarr[resp,outind].set_xlabel('Frequency [Hz]')
                    phase_axarr[resp,outind].set_xlabel('Frequency [Hz]')

                if resp in [0,1,2,3]:
                    side_axarr[resp,outind].loglog(fft_freqs[inds[1]], \
                                                   np.abs(avg_side_fft[eind][resp])[inds[1]], alpha=0.75)
                    if outind == 0:
                        side_axarr[resp,outind].set_ylabel(sidelabs[resp])
                    if resp == 3:
                        side_axarr[resp,outind].set_xlabel('Frequency [Hz]')

                if resp in [0,1,2]:
                    pos_axarr[resp,outind].loglog(fft_freqs[inds[1]], \
                                                  np.abs(avg_data_fft[eind][resp])[inds[1]], alpha=0.75)
                    fb_axarr[resp,outind].loglog(fft_freqs[inds[1]], \
                                                  np.abs(avg_fb_fft[eind][resp])[inds[1]], alpha=0.75)
                    if outind == 0:
                        pos_axarr[resp,outind].set_ylabel(poslabs[resp])
                        fb_axarr[resp,outind].set_ylabel(poslabs[resp] + ' FB')
                    if resp == 2:
                        pos_axarr[resp,outind].set_xlabel('Frequency [Hz]')
                        fb_axarr[resp,outind].set_xlabel('Frequency [Hz]')
            
            drivefig.suptitle('Drive Amplitude vs. Frequency', fontsize=16)
            ampfig.suptitle('ASD of Demod. Carrier Amp vs. Frequency', fontsize=16)
            phasefig.suptitle('ASD of Demod. Carrier Phase vs. Frequency', fontsize=16)
            sidefig.suptitle('ASD of Sum of Neighboring QPD Carrier Amplitudes', fontsize=16)
            posfig.suptitle('ASD of XYZ vs. Frequency', fontsize=16)
            fbfig.suptitle('ASD of XYZ Feedback vs. Frequency', fontsize=16)
            
            for axarr in [amp_axarr, phase_axarr, side_axarr, pos_axarr, fb_axarr, drive_axarr]:
                for drive in [0,1,2]:
                    axarr[0,drive].set_title(poslabs[drive] + ' Drive')
                plt.tight_layout()

            for fig in [ampfig, phasefig, sidefig, posfig, fbfig, drivefig]:
                fig.subplots_adjust(top=0.90)
            

        Hmat_fb = np.einsum('ij, kj -> ikj', \
                            avg_fb_fft[eind], 1. / avg_drive_fft[eind])
        Hmat_amp = np.einsum('ij, kj -> ikj', \
                             avg_amp_fft[eind], 1. / avg_drive_fft[eind])
        Hmat_phase = np.einsum('ij, kj -> ikj', \
                               avg_phase_fft[eind], 1. / avg_drive_fft[eind])

        # Extract the TF, (response / drive), where the drive was above a 
        # fixed threshold.
        Hmatst_good = Hmatst[:,:,finds[b]]
        Hmat_amp_good = Hmat_amp[:,:,finds[b]]
        Hmat_phase_good = Hmat_phase[:,:,finds[b]]

        # Generate an integer by which to roll the data_fft to compute the noise
        # limit of the TF measurement
        shift = int(0.5 * (finds[b][1] - finds[b][0]))
        randadd = np.random.choice(np.arange(-int(0.1*shift), \
                                             int(0.1*shift)+1, 1))
        shift = shift + randadd
        rolled_data_fft = np.roll(avg_data_fft[eind], shift, axis=-1)

        # Compute the Noise TF
        Hmatst_noise = np.einsum('ij, kj -> ikj', \
                                 rolled_data_fft, 1. / avg_drive_fft[eind])
        Hmatst_noise = Hmatst_noise[:,:,finds[b]]

        # Map the 3x7xNfreq arrays to dictionaries with keys given by the drive
        # frequencies and values given by 3x3 complex-values TF matrices
        outind = config.elec_map[eind]
        for i, freq in enumerate(freqs):
            if freq not in Hout:
                if i != 0 and drop_bad_bins:
                    sep = freq - freqs[i-1]
                    # Clause to ignore this particular frequency response if an
                    # above threshold response is found not on a drive bin. Sometimes
                    # random noise components pop up or some power leaks to a 
                    # neighboring bin
                    if sep < 0.9 * (freqs[1] - freqs[0]):
                        continue
                Hout[freq] = np.zeros((3,3), dtype=np.complex128)
                Hout_noise[freq] = np.zeros((3,3), dtype=np.complex128)
                Hout_amp[freq] = np.zeros((5,3), dtype=np.complex128)
                Hout_phase[freq] = np.zeros((5,3), dtype=np.complex128)

            # Add the response from this drive freq/electrode pair to the TF matrix
            Hout[freq][:,outind] += Hmatst_good[:,eind,i]
            Hout_noise[freq][:,outind] += Hmatst_noise[:,eind,i]
            Hout_amp[freq][:,outind] += Hmat_amp[:,eind,i]
            Hout_phase[freq][:,outind] += Hmat_phase[:,eind,i]

    if plot_qpd_response:
        plt.show()

    first_mats = []
    freqs = list(Hout.keys())
    freqs.sort()
    for freq in freqs[:1]:
        first_mats.append(Hout[freq])
    first_mats = np.array(first_mats)

    init_phases = np.mean(np.angle(first_mats), axis=0)
    for drive in [0,1,2]:
        if np.abs(init_phases[drive,drive]) > 2.0:
            print("Correcting phase shift for drive channel", drive)
            sys.stdout.flush()
            for freq in freqs:
                Hout[freq][:,drive] = Hout[freq][:,drive] * (-1)

    out_dict = {'Hout': Hout, 'Hout_amp': Hout_amp, 'Hout_phase': Hout_phase, \
                'Hout_noise': Hout_noise}

    return out_dict













def calibrate_H(Hout, vpn, step_cal_drive_channel = 0, drive_freq = 41.,\
                plate_sep = 0.004):
    '''Calibrates a transfer function with a given charge step calibration.
       This inherently assumes all the gains are matched between the step response
       and transfer function measurement
           INPUTS: Hout, dictionary transfer function to calibrate
                   vpn, volts per Newton for step cal response channel
                   drive_freq, drive frequency for step response
                   plate_sep, face to face electrode separation in meters
                              to compute amplitude of driving force

           OUTPUTS: Hout_cal, calibrated transfer function'''

    print("CALIBRATING H FROM SINGLE-CHARGE STEP...")
    sys.stdout.flush()
    freqs = np.array(list(Hout.keys()))
    freqs.sort()
    ind = np.argmin(np.abs(freqs-drive_freq))

    j = step_cal_drive_channel

    # Compute Vresponse / Vdrive on q = q0:
    npfreqs = np.array(freqs)
    freqs_to_avg = npfreqs[:ind]

    resps = []
    for freq in freqs_to_avg:
        resps.append(np.abs(Hout[freq][j,j]))

    qfac = np.mean(resps)  
    qfac = qfac * plate_sep # convert V -> E-field -> Force

    q = qfac / vpn
    e_charge = config.p_param["e_charge"]
    outstr = "Charge-step calibration implies "+\
             "%0.2f charge during H measurement" % (q / e_charge)

    print(outstr)

    Hout_cal = {}
    for freq in freqs:
        # Normalize transfer functions by charge number
        # and convert to force with capacitor plate separation
        # F = q*E = q*(V/d) so we take
        # (Vresp / Vdrive) * d / q = Vresp / Fdrive
        Hout_cal[freq] = np.copy(Hout[freq]) * (plate_sep / q)

    return Hout_cal, q / e_charge








        

def build_Hfuncs(Hout_cal, fit_freqs = [10.,600], fpeaks=[400.,400.,200.], \
                 weight_peak=False, weight_lowf=False, lowf_weight_fac=0.1, \
                 lowf_thresh=120., plot_without_fits=False,\
                 weight_phase=False, plot_fits=False, plot_inits=False, \
                 grid = False, fit_osc_sum=False, deweight_peak=False, \
                 interpolate = False, max_freq=600, num_to_avg=5):
    # Build the calibrated transfer function array
    # i.e. transfer matrices at each frequency and fit functions to each component

    keys = list(Hout_cal.keys())
    keys.sort()

    keys = np.array(keys)

    mats = []
    for freq in keys:
        mat = Hout_cal[freq]
        mats.append(mat)

    mats = np.array(mats)
    fits = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   

    if plot_fits or plot_without_fits:
        f1, axarr1 = plt.subplots(3,3, sharex='all', sharey='all')
        f2, axarr2 = plt.subplots(3,3, sharex='all', sharey='all')
        f1.suptitle("Magnitude of Transfer Function", fontsize=18)
        f2.suptitle("Phase of Transfer Function", fontsize=18)

        if fit_osc_sum:
            f3, axarr3 = plt.subplots(3,3, sharex='all', sharey='all')
            f4, axarr4 = plt.subplots(3,3, sharex='all', sharey='all')
            f3.suptitle("Magnitude of Transfer Function", fontsize=18)
            f4.suptitle("Phase of Transfer Function", fontsize=18)

    for drive in [0,1,2]:
        for resp in [0,1,2]:
            # Build the array of TF magnitudes and remove NaNs
            mag = np.abs(mats[:,resp,drive])
            nans = np.isnan(mag)
            for nanind, boolv in enumerate(nans):
                if boolv:
                    mag[nanind] = mag[nanind-1]

            # Build the array of TF phases and remove NaNs
            phase = np.angle(mats[:,resp,drive])
            nans2 = np.isnan(phase)
            for nanind, boolv in enumerate(nans2):
                if boolv:
                    phase[nanind] = phase[nanind-1]

            # Unwrap the phase
            unphase = np.unwrap(phase, discont=1.4*np.pi)

            if interpolate:
                b = keys < max_freq
                num = num_to_avg
                magfunc = interp.interp1d(keys[b], mag[b], kind='cubic') #, \
                                          #fill_value=(np.mean(mag[b][:num]), mag[b][-1]), \
                                          #bounds_error=False)
                phasefunc = interp.interp1d(keys[b], unphase[b], kind='cubic') #, \
                                            #fill_value=(np.mean(unphase[b][:num]), unphase[b][-1]), \
                                            #bounds_error=False)

                magfunc2 = make_extrapolator(magfunc, pts=20, order=3, inverse=(False, True))
                phasefunc2 = make_extrapolator(phasefunc, pts=10, order=2, inverse=(False, True))

                fits[resp][drive] = (magfunc2, phasefunc2)
                if plot_fits or plot_without_fits:
                    pts = np.linspace(np.min(keys) / 10., np.max(keys) * 10., len(keys) * 100)
                    

                    if grid:
                        axarr1[resp,drive].grid()
                        axarr2[resp,drive].grid()

                    axarr1[resp,drive].loglog(keys, mag)
                    if not plot_without_fits:
                        axarr1[resp,drive].loglog(pts, magfunc2(pts), color='r', linewidth=2)

                    axarr2[resp,drive].semilogx(keys, unphase / np.pi)
                    if not plot_without_fits:
                        axarr2[resp,drive].semilogx(pts, phasefunc2(pts) / np.pi, color='r', linewidth=2)
                continue

            if not interpolate:
                # Make initial guess based on high-pressure thermal spectra fits
                #therm_fits = dir_obj.thermal_cal_fobj.thermal_cal
                if (drive == 2) or (resp == 2):
                    # Z-direction is considerably different than X or Y
                    Amp = 1e24
                    f0 = 200
                    g = f0 * 5.0
                    fit_freqs = [1.,600.]
                    fpeak = fpeaks[2]
                else:
                    Amp = 1e27
                    f0 = 400
                    g = f0 * 0.3
                    fit_freqs = [1.,600.]
                    fpeak = fpeaks[resp]

                # Construct initial paramter arrays
                p0_mag = [Amp, f0, g]
                p0_phase = [1., f0, g]  # includes arbitrary smearing amplitude

                b1 = keys > fit_freqs[0]
                b2 = keys < fit_freqs[1]
                b = b1 * b2

                # Construct weights if desired
                npkeys = np.array(keys)
                weights = np.zeros(len(npkeys)) + 1.
                if (weight_peak or deweight_peak):
                    if weight_peak:
                        fac = -0.7
                    else:
                        if drive != resp:
                            fac = 1.0
                        else:
                            fac = 1.0
                    weights = weights + fac * np.exp(-(npkeys-fpeak)**2 / (2 * 600) )
                if weight_lowf:
                    ind = np.argmin(np.abs(npkeys - lowf_thresh))
                    if drive != resp:
                        weights[:ind] *= lowf_weight_fac #0.01

                phase_weights = np.zeros(len(npkeys)) + 1.
                if weight_phase and (drive != 2 and resp != 2):
                    low_ind = np.argmin(np.abs(npkeys - 10.0))
                    high_ind = np.argmin(np.abs(npkeys - 10.0))
                    phase_weights[low_ind:ind] *= 0.05

                # Fit the TF magnitude
                try:
                    popt_mag, pcov_mag = opti.curve_fit(damped_osc_amp, keys[b], mag[b], \
                                                   sigma=weights[b], p0=p0_mag, maxfev=1000000)
                except:
                    popt_mag = p0_mag

                # Fit the TF phase with varying phi(DC): -pi, 0 and pi and
                # select the sign based on sum of residuals
                phase_fits = {}
                phase_resids = [0.,0.,0.]
                bounds = ([0.1, -np.inf, -np.inf], [10, np.inf, np.inf])
                for pmult in np.arange(-1,2,1):
                    # Wrap fitting in a try/except block since trying all 3
                    # phi(DC) will inevitably lead to some bad fits
                    try:
                        fitfun = lambda x,a,b,c:damped_osc_phase(x,a,b,c,phase0=np.pi*pmult)
                        popt, pcov = opti.curve_fit(fitfun, keys[b], unphase[b], \
                                               p0=p0_phase, bounds=bounds,
                                               sigma=phase_weights[b])
                    except:
                        #print "bad fit...", drive, resp, np.pi*pmult
                        popt = p0_phase

                    # Save the fits and the residuals
                    phase_fits[pmult] = np.copy(popt)
                    phase_resids[pmult] = np.sum( np.abs( \
                                            damped_osc_phase(keys[b], *popt, phase0=np.pi*pmult) \
                                                         - unphase[b]) )

                lowkey = np.argmin(np.abs(keys[b]-10.0))
                highkey = np.argmin(np.abs(keys[b]-100.0))
                avg = np.mean(unphase[b][lowkey:highkey])

                mult = np.argmin(np.abs(avg - np.array([0, np.pi, -1.0*np.pi])))

                #print drive, resp, phase_resids
                #mult = np.argmin(phase_resids)
                if mult == 2:
                    mult = -1

                popt_phase = phase_fits[mult]
                phase0 = np.pi * mult

                fits[resp][drive] = (popt_mag, popt_phase, phase0)

                if plot_fits or plot_without_fits:

                    fitmag = damped_osc_amp(keys[b], popt_mag[0], \
                                        popt_mag[1], popt_mag[2])
                    fitphase = damped_osc_phase(keys[b], popt_phase[0], \
                                            popt_phase[1], popt_phase[2], phase0=phase0)

                    if plot_inits:
                        maginit = damped_osc_amp(keys[b], p0_mag[0], p0_mag[1], p0_mag[2])
                        phaseinit = damped_osc_phase(keys[b], p0_phase[0], p0_phase[1], \
                                             p0_phase[2], phase0=phase0)

                    if grid:
                        axarr1[resp,drive].grid()
                        axarr2[resp,drive].grid()

                    axarr1[resp,drive].loglog(keys, mag)
                    if not plot_without_fits:
                        axarr1[resp,drive].loglog(keys[b], fitmag, color='r', linewidth=3)
                    if plot_inits:
                        axarr1[resp,drive].loglog(keys[b], maginit, color='k', linewidth=2)

                    axarr2[resp,drive].semilogx(keys, unphase / np.pi)
                    if not plot_without_fits:
                        axarr2[resp,drive].semilogx(keys[b], fitphase / np.pi, \
                                                    color='r', linewidth=3)
                    if plot_inits:
                        axarr2[resp,drive].semilogx(keys[b], phaseinit / np.pi, \
                                                    color='k', linewidth=2)



    if fit_osc_sum:

        sum_fits = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        fitx = fits[0][0]
        fity = fits[1][1]
        fitz = fits[2][2]
        fx = fitx[0][1]
        fy = fity[0][1]
        fz = fitz[0][1]
        for drive in [0,1,2]:
            for resp in [0,1,2]:
                # Build the array of TF magnitudes and remove NaNs
                mag = np.abs(mats[:,resp,drive])
                nans = np.isnan(mag)
                for nanind, boolv in enumerate(nans):
                    if boolv:
                        mag[nanind] = mag[nanind-1]

                # Build the array of TF phases and remove NaNs
                phase = np.angle(mats[:,resp,drive])
                nans2 = np.isnan(phase)
                for nanind, boolv in enumerate(nans2):
                    if boolv:
                        phase[nanind] = phase[nanind-1]

                # Unwrap the phase
                unphase = np.unwrap(phase, discont=1.4*np.pi)

                phase0 = fits[resp][drive][2]
                fit_phase_func = lambda f,a1,a2,a3:\
                                 sum_3osc_phase(f, a1, fx, fitx[0][2], \
                                                a2, fy, fity[0][2], \
                                                a3, fz, fitz[0][2], \
                                                phase0=phase0)

                fit_amp_func = lambda f,a1,g1,a2,g2,a3,g3:\
                                 sum_3osc_amp(f, a1, fx, g1, a2, fy, g2, a3, fz, g3)

                b1 = keys > 1.
                b2 = keys < 501.
                b = b1 * b2

                p0_mag = np.array([fitx[0][0], fitx[0][2], \
                                   fity[0][0], fity[0][2], \
                                   fitz[0][0], fitz[0][2], ])

                p0_phase = np.array([1.,  1., 1.])

                mask = np.array([0.1, 1., 0.1, 1., 0.1, 1.])
                if resp == drive:
                    mask[resp*2] = 1.
                else:
                    mask[resp*2] = 0.5

                p0_mag = mask * p0_mag
                #p0_phase = mask * p0_phase

                popt_mag, pcov_mag = opti.curve_fit(fit_amp_func, keys[b], mag[b], \
                                                   p0=p0_mag, maxfev=1000000)

                popt_phase, pcov_phase = opti.curve_fit(fit_phase_func, keys[b], unphase[b], \
                                                       p0=p0_phase, maxfev=1000000)

                sum_fits[resp][drive] = (popt_mag, popt_phase, phase0)
                #except:
                #    raw_input("Sum fit for Drive: %i, and Response: %i, FAILED" % (drive, resp))

                if plot_fits:

                    fitmag = fit_amp_func(keys[b], *popt_mag)
                    fitphase = fit_phase_func(keys[b], *popt_phase)

                    if grid:
                        axarr3[resp,drive].grid()
                        axarr4[resp,drive].grid()

                    axarr3[resp,drive].loglog(keys, mag)
                    axarr3[resp,drive].loglog(keys[b], fitmag, color='r', linewidth=3)

                    axarr4[resp,drive].semilogx(keys, unphase / np.pi)
                    axarr4[resp,drive].semilogx(keys[b], fitphase / np.pi, color='r', linewidth=3)

    if plot_fits:

        for drive in [0,1,2]:
            axarr1[0, drive].set_title("Drive direction \'%i\'"%drive)
            axarr1[2, drive].set_xlabel("Frequency [Hz]")

            axarr2[0, drive].set_title("Drive direction \'%i\'"%drive)
            axarr2[2, drive].set_xlabel("Frequency [Hz]")

            if fit_osc_sum:

                axarr3[0, drive].set_title("Drive direction \'%i\'"%drive)
                axarr3[2, drive].set_xlabel("Frequency [Hz]")

                axarr4[0, drive].set_title("Drive direction \'%i\'"%drive)
                axarr4[2, drive].set_xlabel("Frequency [Hz]")

        for response in [0,1,2]:
            axarr1[response, 0].set_ylabel("Resp \'%i\' [V/N]" %response)
            axarr2[response, 0].set_ylabel("Resp \'%i\' [$\pi\cdot$rad]" %response)

            if fit_osc_sum:

                axarr3[response, 0].set_ylabel("Resp \'%i\' [V/N]" %response)
                axarr4[response, 0].set_ylabel("Resp \'%i\' [rad]" %response)

        plt.show()

    if interpolate:
        def outfunc(resp, drive, x):
            amp = fits[resp][drive][0](x)
            phase = fits[resp][drive][1](x)
            return amp * np.exp(1.0j * phase)

    else:
        def outfunc(resp, drive, x):
            amp = damped_osc_amp(x, *fits[resp][drive][0])
            phase = damped_osc_phase(x, *fits[resp][drive][1], phase0=fits[resp][drive][2])
            return amp * np.exp(1.0j * phase)

    return outfunc

#################





def make_tf_array(freqs, Hfunc):
    '''Makes a 3x3xNfreq complex-valued array for use in diagonalization
           INPUTS: freqs, array of frequencies
                   Hfunc, output from build_Hfuncs()

           OUTPUTS: Harr, array output'''

    Nfreq = len(freqs)
    Harr = np.zeros((Nfreq,3,3),dtype=np.complex128)

    for drive in [0,1,2]:
        for resp in [0,1,2]:
            Harr[:,drive,resp] = Hfunc(resp, drive, freqs)
    Hout = np.linalg.inv(Harr)
    return Hout



def plot_tf_array(freqs, Harr):
    '''Plots a 3x3xNfreq complex-valued array for use in diagonalization
           INPUTS: freqs, array of frequencies
                   Harr, output from build_Hfuncs()

           OUTPUTS: none'''

    mfig, maxarr = plt.subplots(3,3,figsize=(8,8),sharex=True,sharey=True)
    pfig, paxarr = plt.subplots(3,3,figsize=(8,8),sharex=True,sharey=True)

    for drive in [0,1,2]:
        for resp in [0,1,2]:
            mag = np.abs(Harr[:,drive,resp])
            phase = np.angle(Harr[:,drive,resp])
            maxarr[drive,resp].loglog(freqs, mag)
            paxarr[drive,resp].semilogx(freqs, phase)

    for ind in [0,1,2]:
        maxarr[ind,0].set_ylabel('Mag [abs]')
        paxarr[ind,0].set_ylabel('Phase [rad]')

        maxarr[2,ind].set_xlabel('Frequency [Hz]')
        paxarr[2,ind].set_xlabel('Frequency [Hz]')

    plt.tight_layout()
    plt.show()

