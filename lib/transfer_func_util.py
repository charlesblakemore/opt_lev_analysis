import glob, os, sys, copy, time, math, traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

import scipy
import scipy.optimize as opti
import scipy.signal as signal
import scipy.interpolate as interp
import scipy.constants as constants

import bead_util as bu
import image_util as imu
import configuration as config

import dill as pickle

from iminuit import Minuit, describe





def line(x, a, b):
    return a * x + b



def ipoly1d_func(x, *params):
    '''inverse polynomial function to fit against

           INPUTS: x, independent variable
                   params, N-parameter array

           OUTPUTS: ip(x) = params[0] * (x) ** (-deg) + ... 
                                             + params[deg-1] * (x) ** -1
    '''
    out = x - x  
    deg = len(params) - 1
    for ind, p in enumerate(params):
        out += np.abs(p) * (x)**(ind - deg)
    return out


def ipoly1d(ipolyparams):
    return lambda x: ipoly1d_func(x, *ipolyparams)


def ipolyfit(xs, ys, deg):
    mean = np.mean(ys)
    meanx = np.mean(xs)
    params = np.array([mean * meanx**p for p in range(deg + 1)])
    popt, _ = opti.curve_fit(ipoly1d_func, xs, ys, p0=params, maxfev=10000)
    return popt


def make_extrapolator(interpfunc, xs=[], ys=[], pts=(10, 10), order=(1,1), \
                      arb_power_law=(False, False), semilogx=False):
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
    
    if not len(xs) or not len(ys):
        try:
            xs = interpfunc.x
            ys = interpfunc.y
        except:
            print('Need to provide data, or interpolating function needs to contain\
                   the data as class attributes (such as interp1d objectes)')
            return

    if arb_power_law[0]:
        xx = xs[:pts[0]]
        yy = ys[:pts[0]]
        meanx = np.mean(xx)
        meany = np.mean(yy)

        if semilogx:
            popt_l, _ = opti.curve_fit(line, np.log10(xx), yy)

            p0_l = [meany / np.log10(meanx)]
            def fit_func_l(x, c):
                return popt_l[0] * np.log10(x) + c

        else:
            popt_l, _ = opti.curve_fit(line, np.log10(xx), np.log10(yy))

            p0_l = [meany / (meanx**popt_l[0])]
            def fit_func_l(x, a):
                return a * x**popt_l[0]


        popt2_l, _ = opti.curve_fit(fit_func_l, xx, yy, maxfev=100000, p0=p0_l)
        lower = lambda x: fit_func_l(x, popt2_l[0])

        # lower_params = ipolyfit(xs[:pts[0]], ys[:pts[0]], order[0])
        # lower = ipoly1d(lower_params)       
    else:
        lower_params = np.polyfit(xs[:pts[0]], ys[:pts[0]], order[0])
        lower = np.poly1d(lower_params)

    if arb_power_law[1]:
        xx = xs[-pts[1]:]
        yy = ys[-pts[1]:]
        meanx = np.mean(xx)
        meany = np.mean(yy)

        if semilogx:
            popt_u, _ = opti.curve_fit(line, np.log10(xx), yy)

            p0_u = [meany / np.log10(meanx)]
            def fit_func_u(x, c):
                return popt_u[0] * np.log10(x) + c

        else:
            popt_u, _ = opti.curve_fit(line, np.log10(xx), np.log10(yy))

            p0_u = [meany / (meanx**popt_u[0])]
            def fit_func_u(x, a):
                return a * x**popt_u[0]

        popt2_u, _ = opti.curve_fit(fit_func_u, xx, yy, maxfev=100000, p0=p0_u)
        upper = lambda x: fit_func_u(x, popt2_u[0])
        # upper_params = ipolyfit(xs[-pts[1]:], ys[-pts[1]:], order[1])
        # upper = ipoly1d(upper_params) 
    else:
        upper_params = np.polyfit(xs[-pts[1]:], ys[-pts[1]:], order[1])
        upper = np.poly1d(upper_params) 

    def extrapfunc(x):

        ubool = x > xs[-1]
        lbool = x < xs[0]

        midval = interpfunc( x[ np.invert(ubool + lbool) ] )
        uval = upper( x[ubool] )
        lval = lower( x[lbool] )

        return np.concatenate((lval, midval, uval))

    return extrapfunc
        
        

    


def build_uncalibrated_H(fobjs, average_first=True, dpsd_thresh = 8e-1, mfreq = 1., \
                         skip_qpd=False, plot_response=False, drop_bad_bins=True, \
                         new_trap=False, lines_to_remove=[60.0], zero_drive_phase=False):
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

    if not skip_qpd:
        avg_side_fft = {}

        avg_amp_fft = {}
        avg_phase_fft = {}

    counts = {}

    if plot_response:
        fbfig, fb_axarr = plt.subplots(3,3,sharex=True,sharey=True,figsize=(9,8))
        posfig, pos_axarr = plt.subplots(3,3,sharex=True,sharey='row',figsize=(9,8))
        drivefig, drive_axarr = plt.subplots(3,3,sharex=True,sharey=True,figsize=(9,8))
        if not skip_qpd:
            ampfig, amp_axarr = plt.subplots(5,3,sharex=True,sharey=True,figsize=(9,10))
            phasefig, phase_axarr = plt.subplots(5,3,sharex=True,sharey=True,figsize=(9,10))
            sidefig, side_axarr = plt.subplots(4,3,sharex=True,sharey=True,figsize=(9,9))

    filind = 0
    for fobj in fobjs:

        N = np.shape(fobj.pos_data)[1]#number of samples
        fsamp = fobj.fsamp
        
        fft_fac = bu.fft_norm(N, fsamp)

        drive = bu.trap_efield(fobj.electrode_data) #* constants.elementary_charge
        #drive = np.roll(drive, -10, axis=-1)

        # dfft = np.fft.rfft(fobj.electrode_data) #fft of electrode drive in daxis.
        dfft = np.fft.rfft( drive ) * fft_fac
        if new_trap: 
            data_fft = np.fft.rfft(fobj.pos_data_3) * fft_fac
        else:
            data_fft = np.fft.rfft(fobj.pos_data) * fft_fac

        fb_fft = np.fft.rfft(fobj.pos_fb) * fft_fac

        if not skip_qpd:
            amp_fft = np.fft.rfft(fobj.amp) * fft_fac
            phase_fft = np.fft.rfft(fobj.phase) * fft_fac

            left = fobj.amp[2] + fobj.amp[3]
            right = fobj.amp[0] + fobj.amp[1]
            top = fobj.amp[2] + fobj.amp[0]
            bot = fobj.amp[3] + fobj.amp[1]
            side_fft = np.fft.rfft(np.array([right, left, top, bot])) * fft_fac

        fft_freqs = np.fft.rfftfreq(N, d=1.0/fsamp)

        # for i in range(len(dfft)):
        #     plt.loglog(fft_freqs, np.abs(dfft[i]))
        # # plt.loglog(fft_freqs, np.abs(dfft[4]))
        # # plt.loglog(fft_freqs, np.abs(data_fft[0]))
        # plt.show()

        dpsd = np.abs(dfft)**2  #psd for all electrode drives
        dpsd_thresh = 0.1 * np.max(dpsd.flatten())
        inds = np.where(dpsd>dpsd_thresh)#Where the dpsd is over the threshold for being used.
        eind = np.unique(inds[0])[0]
        # print(eind)

        if eind not in avg_drive_fft:
            avg_drive_fft[eind] = np.zeros(dfft.shape, dtype=np.complex128)
            avg_data_fft[eind] = np.zeros(data_fft.shape, dtype=np.complex128)
            avg_fb_fft[eind] = np.zeros(fb_fft.shape, dtype=np.complex128)
            if not skip_qpd:
                avg_amp_fft[eind] = np.zeros(amp_fft.shape, dtype=np.complex128)
                avg_phase_fft[eind] = np.zeros(phase_fft.shape, dtype=np.complex128)
                avg_side_fft[eind] = np.zeros(side_fft.shape, dtype=np.complex128)
            counts[eind] = 0.

        # try:
        avg_drive_fft[eind] += dfft
        avg_data_fft[eind] += data_fft
        avg_fb_fft[eind] += fb_fft
        if not skip_qpd:
            avg_amp_fft[eind] += amp_fft
            avg_phase_fft[eind] += phase_fft
            avg_side_fft[eind] += side_fft

        # except:
        #     traceback.print_exc()
        #     print()
        #     print(fobj.fname)
        #     derpfig, derpax = plt.subplots(1,1)
        #     for i in [0,1,2,3,4]:
        #         derpax.loglog(np.abs(amp_fft[i]) * 10**i, label=str(i))
        #     derpax.legend()
        #     plt.show()
        #     input()

        counts[eind] += 1.

    for eind in list(counts.keys()):
        print(eind, counts[eind])
        avg_drive_fft[eind] = avg_drive_fft[eind] / counts[eind]
        avg_data_fft[eind] = avg_data_fft[eind] / counts[eind]
        avg_fb_fft[eind] = avg_fb_fft[eind] / counts[eind]
        if not skip_qpd:
            avg_amp_fft[eind] = avg_amp_fft[eind] / counts[eind]
            avg_phase_fft[eind] = avg_phase_fft[eind] / counts[eind]
            avg_side_fft[eind] = avg_side_fft[eind] / counts[eind]

    poslabs = {0: 'X', 1: 'Y', 2: 'Z'}
    sidelabs = {0: 'Right', 1: 'Left', 2: 'Top', 3: 'Bottom'}
    quadlabs = {0: 'Top Right', 1: 'Bottom Right', 2: 'Top Left', \
                3: 'Bottom Left', 4: 'Backscatter'}

    for eind in list(avg_drive_fft.keys()):
        # First find drive-frequency bins above a fixed threshold
        dpsd = np.abs(avg_drive_fft[eind])**2 
        inds = np.where(dpsd > dpsd_thresh)

        # Extract the frequency indices
        finds = inds[1]

        # Ignore DC and super low frequencies
        mfreq = 1.0
        b = finds > np.argmin(np.abs(fft_freqs - mfreq))

        tf_inds = finds[b]
        for linefreq in lines_to_remove:
            print('Ignoring response at line frequency: {:0.1f}'.format(linefreq))
            line_freq_ind = np.argmin(np.abs(fft_freqs - linefreq))
            tf_inds = tf_inds[tf_inds != line_freq_ind]

        freqs = fft_freqs[tf_inds]

        xlim = (np.min(fft_freqs[1:]), np.max(fft_freqs))
        # xlim = (45.0, 130.0)

        # plt.plot(freqs, np.angle(avg_drive_fft[eind][eind,tf_inds]) / np.pi)
        # plt.title('Raw Phase From Drive FFT')
        # plt.ylabel('Apparent Phase [$\\pi \\, rad$]')
        # plt.xlabel('Drive Frequency [Hz]')
        # plt.tight_layout()

        # plt.figure()
        # plt.plot(freqs, np.unwrap(np.angle(avg_drive_fft[eind][eind,tf_inds])) / np.pi)
        # plt.title('Unwrapped Phase From Drive FFT')
        # plt.ylabel('Apparent Phase [$\\pi \\, rad$]')
        # plt.xlabel('Drive Frequency [Hz]')
        # plt.tight_layout()
        # plt.show()

        # outind = config.elec_map[eind]
        outind = eind

        if plot_response:
            
            for elec in [0,1,2]: #,3,4,5,6,7]:
                drive_axarr[elec,outind].loglog(fft_freqs, \
                                                np.abs(avg_drive_fft[eind][elec]), alpha=1.0)
                drive_axarr[elec,outind].loglog(fft_freqs[tf_inds], \
                                                np.abs(avg_drive_fft[eind][elec])[tf_inds], alpha=1.0)
                if outind == 0:
                    drive_axarr[elec,outind].set_ylabel('Efield axis ' + str(elec) \
                                                            + '\n[(V/m)/$\\sqrt{\\rm Hz}$]')
                if elec == 2: #7:
                    drive_axarr[elec,outind].set_xlabel('Frequency [Hz]')


            for resp in [0,1,2,3,4]:

                if not skip_qpd:
                    amp_axarr[resp,outind].loglog(fft_freqs[tf_inds], \
                                                  np.abs(avg_amp_fft[eind][resp])[tf_inds], alpha=1.0)
                    phase_axarr[resp,outind].loglog(fft_freqs[tf_inds], \
                                                    np.abs(avg_phase_fft[eind][resp])[tf_inds], alpha=1.0)
                    if outind == 0:
                        amp_axarr[resp,outind].set_ylabel(quadlabs[resp])
                        phase_axarr[resp,outind].set_ylabel(quadlabs[resp])
                    if resp == 4:
                        amp_axarr[resp,outind].set_xlabel('Frequency [Hz]')
                        phase_axarr[resp,outind].set_xlabel('Frequency [Hz]')

                    if resp in [0,1,2,3]:
                        side_axarr[resp,outind].loglog(fft_freqs[tf_inds], \
                                                       np.abs(avg_side_fft[eind][resp])[tf_inds], alpha=1.0)
                        if outind == 0:
                            side_axarr[resp,outind].set_ylabel(sidelabs[resp])
                        if resp == 3:
                            side_axarr[resp,outind].set_xlabel('Frequency [Hz]')


                if resp in [0,1,2]:
                    # if resp == 2:
                    #     fac = 1000.0
                    # else:
                    #     fac = 1.0
                    pos_axarr[resp,outind].loglog(fft_freqs, np.abs(avg_data_fft[eind][resp]), alpha=1.0)
                    pos_axarr[resp,outind].loglog(fft_freqs[tf_inds], \
                                                  np.abs(avg_data_fft[eind][resp])[tf_inds], alpha=1.0)

                    fb_axarr[resp,outind].loglog(fft_freqs, np.abs(avg_fb_fft[eind][resp]), alpha=1.0)
                    fb_axarr[resp,outind].loglog(fft_freqs[tf_inds], \
                                                  np.abs(avg_fb_fft[eind][resp])[tf_inds], alpha=1.0)
                    if outind == 0:
                        pos_axarr[resp,outind].set_ylabel(poslabs[resp] + ' [arb]')
                        fb_axarr[resp,outind].set_ylabel(poslabs[resp] + ' FB\n[bits/$\\sqrt{\\rm Hz}$]')
                    if resp == 2:
                        pos_axarr[resp,outind].set_xlabel('Frequency [Hz]')
                        fb_axarr[resp,outind].set_xlabel('Frequency [Hz]')
            
            drivefig.suptitle('Drive Amplitude vs. Frequency', fontsize=16)
            posfig.suptitle('ASD of XYZ vs. Frequency', fontsize=16)
            fbfig.suptitle('ASD of XYZ Feedback vs. Frequency', fontsize=16)
            if not skip_qpd:
                ampfig.suptitle('ASD of Demod. Carrier Amp vs. Frequency', fontsize=16)
                phasefig.suptitle('ASD of Demod. Carrier Phase vs. Frequency', fontsize=16)
                sidefig.suptitle('ASD of Sum of Neighboring QPD Carrier Amplitudes', fontsize=16)

                figlist = [posfig, fbfig, drivefig, ampfig, phasefig, sidefig]
                axlist = [pos_axarr, fb_axarr, drive_axarr, amp_axarr, phase_axarr, side_axarr]

            else:
                figlist = [posfig, fbfig, drivefig]
                axlist = [pos_axarr, fb_axarr, drive_axarr]


            for axind, axarr in enumerate(axlist):
                # axarr[0,0].set_xlim(*xlim)
                axarr[0,0].set_xlim(34, 107)

                for drive in [0,1,2]:
                    axarr[0,drive].set_title(poslabs[drive] + ' Drive')

                for resp in [0,1,2]:
                    if (axind != 0) and (resp != 0):
                        continue

                    mag_major_locator = LogLocator(base=10.0, numticks=30)
                    mag_minor_locator = LogLocator(base=1.0, numticks=300)

                    axarr[resp,0].yaxis.set_major_locator(mag_major_locator)
                    axarr[resp,0].yaxis.set_minor_locator(mag_minor_locator)
                    axarr[resp,0].yaxis.set_minor_formatter(NullFormatter())

                for d in [0,1,2]:
                    for r in [0,1,2]:
                        axarr[r,d].grid(which='both')



        ### Compute FFT of each response divided by FFT of each drive.
        ### This is way more information than we need for a single drive freq
        ### and electrode pair, but it allows a nice vectorization
        Hmat = np.einsum('ij, kj -> ikj', \
                             avg_data_fft[eind][:,tf_inds], 1. / avg_drive_fft[eind][:,tf_inds])

        Hmat_fb = np.einsum('ij, kj -> ikj', \
                            avg_fb_fft[eind][:,tf_inds], 1. / avg_drive_fft[eind][:,tf_inds])

        if not skip_qpd:
            Hmat_amp = np.einsum('ij, kj -> ikj', \
                                 avg_amp_fft[eind][:,tf_inds], 1. / avg_drive_fft[eind][:,tf_inds])
            Hmat_phase = np.einsum('ij, kj -> ikj', \
                                   avg_phase_fft[eind][:,tf_inds], 1. / avg_drive_fft[eind][:,tf_inds])


        # Generate an integer by which to roll the data_fft to compute the noise
        # limit of the TF measurement
        if len(tf_inds) > 1:
            shift = int(0.5 * (tf_inds[1] - tf_inds[0]))
        else:
            shift = int(0.5 * tf_inds[0])
        randadd = np.random.choice(np.arange(-int(0.1*shift), \
                                             int(0.1*shift)+1, 1))
        shift = shift + randadd
        rolled_data_fft = np.roll(avg_data_fft[eind], shift, axis=-1)

        # Compute the Noise TF
        Hmat_noise = np.einsum('ij, kj -> ikj', \
                                 rolled_data_fft[:,tf_inds], 1. / avg_drive_fft[eind][:,tf_inds])

        # Map the 3x7xNfreq arrays to dictionaries with keys given by the drive
        # frequencies and values given by 3x3 complex-values TF matrices
        #outind = config.elec_map[eind]
        outind = eind
        for i, freq in enumerate(freqs):
            if freq not in Hout:
                if i != 0 and drop_bad_bins:
                    sep = freq - freqs[i-1]
                    # Clause to ignore this particular frequency response if an
                    # above threshold response is found not on a drive bin. Sometimes
                    # random noise components pop up or some power leaks to a 
                    # neighboring bin
                    if sep < 0.8 * (freqs[1] - freqs[0]):
                        continue
                Hout[freq] = np.zeros((3,3), dtype=np.complex128)
                Hout_noise[freq] = np.zeros((3,3), dtype=np.complex128)
                Hout_amp[freq] = np.zeros((5,3), dtype=np.complex128)
                Hout_phase[freq] = np.zeros((5,3), dtype=np.complex128)

            # Add the response from this drive freq/electrode pair to the TF matrix
            Hout[freq][:,outind] += Hmat[:,eind,i]
            Hout_noise[freq][:,outind] += Hmat_noise[:,eind,i]
            if not skip_qpd:
                Hout_amp[freq][:,outind] += Hmat_amp[:,eind,i]
                Hout_phase[freq][:,outind] += Hmat_phase[:,eind,i]

    if plot_response:
        for fig in figlist:
            fig.tight_layout()
            fig.subplots_adjust(top=0.90)
        plt.show()

    # first_mats = []
    freqs = list(Hout.keys())
    freqs.sort()


    if zero_drive_phase:
        # init_phases = np.angle(Hout[freqs[0]])

        first_mats = []
        for freq in freqs[1:3]:
            first_mats.append(Hout[freq])
        first_mats = np.array(first_mats)
        init_phases = np.mean(np.unwrap(np.angle(first_mats), axis=0), axis=0)

        # print(init_phases)
        # input()

        for drive in [0,1,2]:
            if np.abs(init_phases[drive,drive]) > 1.5:
                ### Check the second frequency to make sure the first isn't crazy
                if np.abs(np.angle(Hout[freqs[1]][drive,drive])) > 1.5:
                    print("Correcting phase shift for drive channel", drive)
                    sys.stdout.flush()
                    for freq in freqs:
                        Hout[freq][drive,:] = Hout[freq][drive,:] * (-1)
                        # Hout[freq][:,drive] = Hout[freq][:,drive] * (-1)
                else:
                    Hout[freqs[0]][drive,:] = Hout[freqs[0]][drive,:] * (-1)


    out_dict = {'Hout': Hout, 'Hout_amp': Hout_amp, 'Hout_phase': Hout_phase, \
                'Hout_noise': Hout_noise}

    return out_dict













def calibrate_H(Hout, vpn, step_cal_drive_channel=0, drive_freq=41., \
                verbose=False, neg_charge=True):
    '''Calibrates a transfer function with a given charge step calibration.
       This inherently assumes all the gains are matched between the step response
       and transfer function measurement
           INPUTS: Hout, dictionary transfer function to calibrate
                   vpn, volts per Newton for step cal response channel
                   drive_freq, drive frequency for step response
                   step_cal_drive_channel, pretty self-explanatory
                   verbose, print some words that could be helpful

           OUTPUTS: Hout_cal, calibrated transfer function'''

    print("CALIBRATING H FROM SINGLE-CHARGE STEP...")
    sys.stdout.flush()
    freqs = np.array(list(Hout.keys()))
    freqs.sort()
    ind = np.argmin(np.abs(freqs-drive_freq))

    j = step_cal_drive_channel

    ### Compute Vresponse / Vdrive on q = q0:
    npfreqs = np.array(freqs)
    freqs_to_avg = npfreqs[:ind]

    resps = []
    for freq in freqs_to_avg:
        resps.append(np.abs(Hout[freq][j,j]))

    #test_voltages = np.zeros(8)
    #test_voltages[ind_map[j]] = 1.0
    #test_efield = np.abs(bu.trap_efield(test_voltages, nsamp=1)[j])

    mean_resp = np.mean(resps)

    ### Probably still need to sort out consistency in using the in-phase
    ### correlation and the real part of this transfer function
    mean_resp = np.abs(Hout[freqs[ind]][j,j])
    # mean_resp = Hout[freqs[ind]][j,j].real

    q_tf = mean_resp / vpn
    e_charge = constants.elementary_charge

    if neg_charge:
        q_tf *= -1.0

    if verbose:
        outstr = " Charge-step calibration implies "+\
                 " {:0.2f} charge during H measurement".format(q_tf / e_charge)
        print(outstr)

    Hout_cal = {}
    for freq in freqs:
        ### Normalize transfer functions by charge number
        ### and convert to force with capacitor plate separation
        ### F = q*E = q*(V/d) so we take
        ### (Vresp / Vdrive) * d / q = Vresp / Fdrive
        Hout_cal[freq] = np.copy(Hout[freq]) * (1.0 / np.abs(q_tf))

    return Hout_cal, q_tf / e_charge








        

def build_Hfuncs(Hout_cal, fit_freqs = [10.,600.], fpeaks=[400.,400.,200.], \
                 weight_peak=False, weight_lowf=False, lowf_weight_fac=0.1, \
                 lowf_thresh=120., linearize=False, ignore_phase=False,
                 weight_phase=False, plot=False, plot_fits=False, \
                 plot_inits=False, plot_off_diagonal=False, \
                 grid = False, fit_osc_sum=False, deweight_peak=False, \
                 interpolate = False, max_freq=600, num_to_avg=5, \
                 real_unwrap=[[0, 1, 1], [1, 0, 1], [1, 1, 0]], \
                 derpy_unwrap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], \
                 interps=[[0, 1, 1], [1, 0, 1], [1, 1, 0]], \
                 smoothing=1.0, amp_xlim=(), amp_ylim=(), \
                 phase_xlim=(), phase_ylim=()):
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

    if plot:
        # figsize = (8,6)
        figsize = (10,8)
        # f1, axarr1 = plt.subplots(3,3, sharex=True, sharey='row', figsize=figsize)
        f1, axarr1 = plt.subplots(3,3, sharex=True, sharey='row', figsize=figsize)
        f2, axarr2 = plt.subplots(3,3, sharex=True, sharey='row', figsize=figsize)

        # colors = bu.get_color_map(5, cmap='inferno')
        data_color, fit_color = ['k', 'C1']

    for drive in [0,1,2]:
        for resp in [0,1,2]:

            interpolate = interps[resp][drive]

            d_unwrap = derpy_unwrap[resp][drive]
            r_unwrap = real_unwrap[resp][drive]

            ### A shitty shit factor that I have to add because folks operating
            ### the new trap don't know how to scale things
            # if resp == 2:
            #     plot_fac = 3e-7
            # else:
            #     plot_fac = 1.0
            plot_fac = 1

            ### Build the array of TF magnitudes and remove NaNs
            mag = np.abs(mats[:,resp,drive])
            nans = np.isnan(mag)
            for nanind, boolv in enumerate(nans):
                if boolv:
                    mag[nanind] = mag[nanind-1]

            ### Build the array of TF phases and remove NaNs
            phase = np.angle(mats[:,resp,drive])
            nans2 = np.isnan(phase)
            for nanind, boolv in enumerate(nans2):
                if boolv:
                    phase[nanind] = phase[nanind-1]

            ### Unwrap the phase
            if d_unwrap:
                pos_inds = phase > np.pi / 4.0
                unphase = phase - 2.0 * np.pi * pos_inds
            else:
                unphase = np.copy(phase)

            if r_unwrap:
                unphase = np.unwrap(unphase)

            # if drive == resp == 1:
            #     plt.figure()
            #     plt.semilogx(keys, phase)
            #     plt.semilogx(keys, np.unwrap(phase))
            #     plt.semilogx(keys, unphase)
            #     plt.show()

            #     input()

            b1 = keys >= fit_freqs[0]
            b2 = keys <= fit_freqs[1]
            b = b1 * b2

            if interpolate:
                num = num_to_avg
                mw = (1.0 / np.std(mag[b][:10])) * np.ones(np.sum(b))
                pw = (1.0 / np.std(unphase[b][:10])) * np.ones(np.sum(b))
                magfunc = interp.UnivariateSpline(keys[b], mag[b], w=mw, k=2, s=smoothing)
                phasefunc = interp.UnivariateSpline(keys[b], unphase[b], w=pw, k=2, s=smoothing)
                # magfunc = interp.interp1d(keys[b], mag[b], kind='quadratic')
                # phasefunc = interp.interp1d(keys[b], unphase[b], kind='quadratic')

                if resp == 2:
                    arb_power_law_mag = (True, True)
                    arb_power_law_phase = (True, True)
                    if drive == 2:
                        pts_mag = (4, 30)
                        pts_phase = (4, 20)
                    else:
                        pts_mag = (10, 30)
                        pts_phase = (10, 20)
                else:
                    arb_power_law_mag = (False, True)
                    arb_power_law_phase = (False, True)
                    pts_mag = (10, 30)
                    pts_phase = (10, 20)

                magfunc2 = make_extrapolator(magfunc, xs=keys[b], ys=mag[b], \
                                                pts=pts_mag, order=(0, 0), \
                                                arb_power_law=arb_power_law_mag)
                phasefunc2 = make_extrapolator(phasefunc, xs=keys[b], ys=unphase[b], \
                                                pts=pts_phase, order=(0, 0), \
                                                arb_power_law=arb_power_law_phase, semilogx=True)

                mag_params = (keys[b], mag[b], pts_mag, arb_power_law_mag)
                phase_params = (keys[b], unphase[b], pts_phase, arb_power_law_phase)
                fits[resp][drive] = (mag_params, phase_params)

                if plot:
                    pts = np.linspace(np.min(keys) / 2., np.max(keys) * 2., len(keys) * 100)
            
                    axarr1[resp,drive].loglog(keys, mag * plot_fac, 'o', ms=6, color=data_color)
                    axarr2[resp,drive].semilogx(keys, unphase, 'o', ms=6, color=data_color)

                    if plot_fits and ((resp == drive) or plot_off_diagonal):
                        axarr1[resp,drive].loglog(pts, magfunc2(pts) * plot_fac, color=fit_color, \
                                                    linestyle='--', linewidth=2, alpha=1.0)
                        axarr2[resp,drive].semilogx(pts, phasefunc2(pts), color=fit_color, \
                                                    linestyle='--', linewidth=2, alpha=1.0)


            if not interpolate:

                magscale = np.mean(mag[b])
                phasescale = np.mean(unphase[b])

                fpeak = keys[np.argmax(mag)]
                if fpeak < 100.0:
                    fpeak = fpeaks[resp]

                ### Make initial guess based on high-pressure thermal spectra fits
                if (drive == 2) or (resp == 2):
                    ### Z-direction is considerably different than X or Y
                    g = fpeak * 2.0
                else:
                    g = fpeak * 0.15

                amp0 = np.mean( mag[b][:np.argmin(np.abs(keys[b] - 100.0))] ) \
                                * ((2.0 * np.pi * fpeak)**2)

                ### Construct initial paramter arrays
                p0_mag = [amp0/magscale, fpeak, g]
                p0_phase = [1., fpeak, g]  ### includes arbitrary smearing amplitude

                ### Construct weights if desired
                npkeys = np.array(keys)
                mag_weights = np.zeros_like(npkeys) + 1.
                phase_weights = np.zeros(len(npkeys)) + 1.

                if (weight_peak or deweight_peak):
                    if weight_peak:
                        fac = -0.7
                    else:
                        if drive != resp:
                            fac = 1.0
                        else:
                            fac = 1.0
                    mag_weights = mag_weights + fac * np.exp(-(npkeys-fpeak)**2 / (2 * 50) )
                    phase_weights = phase_weights + fac * np.exp(-(npkeys-fpeak)**2 / (2 * 50) )

                if weight_lowf:
                    ind = np.argmin(np.abs(npkeys - lowf_thresh))
                    # if drive != resp:
                    mag_weights[:ind] *= lowf_weight_fac #0.01
                    phase_weights[:ind] *= lowf_weight_fac #0.01

                # mag_weights *= amp0 / ((2.0 * np.pi * fpeak)**2)
                # mag_weights *= np.sqrt(mag)

                # plt.figure()
                # plt.loglog(keys[b], mag[b]/magscale)
                # plt.loglog(keys[b], mag_weights[b])
                # plt.show()


                lowkey = np.argmin(np.abs(keys[b]-10.0))
                highkey = np.argmin(np.abs(keys[b]-100.0))
                avg = np.mean(unphase[b][lowkey:highkey])

                mult = np.argmin(np.abs(avg - np.array([0, np.pi, -1.0*np.pi])))
                if mult == 2:
                    mult = -1
                phase0 = np.pi * mult

                # def NLL(amp, f0, g):
                #     mag_term = ( bu.damped_osc_amp(keys[b], amp, f0, g) - mag[b] )**2 / mag_weights[b]**2
                #     phase_term = ( bu.damped_osc_phase(keys[b], 1.0, f0, g, phase0=phase0) - unphase[b] )**2 \
                #                         / phase_weights[b]**2
                #     return np.sum(mag_term + phase_term)

                # if linearize:
                #     def NLL(amp, f0, g):
                #         num = ( np.log(bu.damped_osc_amp(keys[b], amp, f0, g)) - np.log(mag[b]) )**2 
                #         denom = np.log(mag_weights[b])**2
                #         return np.sum(num / denom)
                if ignore_phase:
                    def NLL(amp, f0, g):
                        num = ( bu.damped_osc_amp(keys[b], amp, f0, g) - mag[b]/magscale )**2 
                        denom = mag_weights[b]**2
                        return np.sum(num / denom)
                else:
                    def NLL(amp, f0, g):
                        num1 = ( bu.damped_osc_amp(keys[b], amp, f0, g) - mag[b]/magscale )**2
                        num2 = ( bu.damped_osc_phase(keys[b], 1.0, f0, g, phase0=phase0) \
                                        - unphase[b])**2 
                        denom1 = mag_weights[b]**2
                        denom2 = phase_weights[b]**2
                        return np.sum( (num1 / denom1) + (num2 / denom2) )


                m = Minuit(NLL,
                           amp = amp0/magscale, # set start parameter
                           # fix_amp = 'True', # you can also fix it
                           limit_amp = (0.0, np.inf),
                           f0 = fpeak, # set start parameter
                           # fix_f0 = 'True', 
                           limit_f0 = (0.0, np.inf),
                           g = g, # set start parameter
                           # fix_g = "True", 
                           limit_g = (0, np.inf),
                           errordef = 1,
                           print_level = 1, 
                           pedantic=False)
                m.migrad(ncall=500000)

                # plt.figure()
                # m.draw_mnprofile('f0')
                # plt.figure()
                # m.draw_mncontour('amp', 'f0')
                # input()

                popt_mag = [m.values['amp']*magscale, m.values['f0'], m.values['g']]
                popt_phase = [1.0, m.values['f0'], m.values['g']]

                print()
                print(popt_mag)
                print()

                fits[resp][drive] = (popt_mag, popt_phase, phase0)

                # if drive == resp:
                #     print()
                #     print(drive)
                #     print(popt_mag)
                #     print(pcov_mag)
                #     print(popt_phase)
                #     print(pcov_phase)
                #     print()

                if plot:
                    pts = np.linspace(np.min(keys) / 2., np.max(keys) * 2., len(keys) * 100)

                    axarr1[resp,drive].loglog(keys, mag, 'o', ms=6, color=data_color)
                    axarr2[resp,drive].semilogx(keys, unphase, 'o', ms=6, color=data_color)

                    if plot_fits:
                        fitmag = bu.damped_osc_amp(pts, *popt_mag)
                        axarr1[resp,drive].loglog(pts, fitmag, ls='-', \
                                                  color=fit_color, linewidth=2)

                        fitphase = bu.damped_osc_phase(pts, *popt_phase, phase0=phase0)
                        axarr2[resp,drive].semilogx(pts, fitphase, ls='-', \
                                                    color=fit_color, linewidth=2)

                    if plot_inits:
                        maginit = bu.damped_osc_amp(pts, *p0_mag)
                        axarr1[resp,drive].loglog(pts, maginit, ls='-', color='k', linewidth=2)

                        phaseinit = bu.damped_osc_phase(pts, *p0_phase, phase0=phase0)
                        axarr2[resp,drive].semilogx(pts, phaseinit, \
                                                    ls='-', color='k', linewidth=2)


    if plot:

        ax_to_pos = {0: 'X', 1: 'Y', 2: 'Z'}
        for drive in [0,1,2]:
            # axarr1[0, drive].set_title("Drive direction {:s}".format(ax_to_pos[drive]))
            axarr1[0, drive].set_title("Drive $\\widetilde{{F}}_{:s}$".format(ax_to_pos[drive]))
            axarr1[2, drive].set_xlabel("Frequency [Hz]")
            # axarr1[2, drive].set_xticks([1, 10, 100])
            axarr1[2, drive].set_xticks([1, 10, 100, 1000])
            if amp_xlim:
                axarr1[2, drive].set_xlim(*amp_xlim)

            # axarr2[0, drive].set_title("Drive direction {:s}".format(ax_to_pos[drive]))
            axarr2[0, drive].set_title("Drive $\\widetilde{{F}}_{:s}$".format(ax_to_pos[drive]))
            axarr2[2, drive].set_xlabel("Frequency [Hz]")
            # axarr2[2, drive].set_xticks([1, 10, 100])
            axarr2[2, drive].set_xticks([1, 10, 100, 1000])
            # axarr2[2, drive].set_xlim(2.5, 1800)
            if phase_xlim:
                axarr2[2, drive].set_xlim(*phase_xlim)


        for response in [0,1,2]:

            mag_major_locator = LogLocator(base=10.0, numticks=30)
            mag_minor_locator = LogLocator(base=10.0, numticks=30)

            # axarr1[response, 0].set_ylabel("Resp {:s} [Arb/N]".format(ax_to_pos[response]))
            axarr1[response, 0].set_ylabel("$| \\widetilde{{R}}_{:s} / \\widetilde{{F}}_i |$ [Arb/N]"\
                                                .format(ax_to_pos[response]))
            # axarr1[response, 0].set_yticks([1e9, 1e11, 1e13])
            axarr1[response, 0].yaxis.set_major_locator(mag_major_locator)
            # axarr1[response, 0].set_yticks([1e8, 1e10, 1e12, 1e14], minor=True)
            axarr1[response, 0].yaxis.set_minor_locator(mag_minor_locator)
            axarr1[response, 0].yaxis.set_minor_formatter(NullFormatter())
            # if response != 2:
            #     axarr1[response, 0].set_yticks([1e9, 1e10, 1e11])
            #     axarr1[response, 0].set_ylim(3e9, 1.3e11)

            if amp_ylim:
                if type(amp_ylim) == tuple:
                    axarr1[response, 0].set_ylim(*amp_ylim)
                elif type(amp_ylim) == list:
                    axarr1[response, 0].set_ylim(*(amp_ylim[response]))
                else:
                    print('custom y-axis limits provided are not of the right type')

            # axarr2[response, 0].set_ylabel("Resp {:s} [$\\pi\\cdot$rad]".format(ax_to_pos[response]))
            axarr2[response, 0].set_ylabel("$ \\angle \\, \\widetilde{{R}}_{:s} / \\widetilde{{F}}_i $ [rad]"\
                                                .format(ax_to_pos[response]))
            axarr2[response, 0].set_yticks([-2*np.pi, -1*np.pi, 0, 1*np.pi, 2*np.pi])
            axarr2[response, 0].set_yticklabels(['-2$\\pi$', '-$\\pi$', '0', '$\\pi$', '2$\\pi$'])
            axarr2[response, 0].set_ylim(-1.3*np.pi, 1.3*np.pi)

            if phase_ylim:
                axarr2[response, 0].set_ylim(*phase_ylim)


        f1.suptitle("Magnitude of Transfer Function", fontsize=18)
        f2.suptitle("Phase of Transfer Function", fontsize=18)

        f1.tight_layout()
        f2.tight_layout()

        f1.subplots_adjust(wspace=0.065, hspace=0.1, top=0.9)
        f2.subplots_adjust(wspace=0.065, hspace=0.1, top=0.9)


        if grid:
            for d in [0,1,2]:
                for r in [0,1,2]:
                    axarr1[r,d].grid(True, which='both')
                    axarr2[r,d].grid(True, which='both')

        plt.show()

    return fits, interps

#################





def make_tf_array(freqs, Hfunc, suppress_off_diag=False, smoothing=1.0, \
                  adjust_phase=False, adjust_phase_dict={}):
    '''Makes a 3x3xNfreq complex-valued array for use in diagonalization
           INPUTS: freqs, array of frequencies
                   Hfunc, output from build_Hfuncs()

           OUTPUTS: Harr, array output'''

    fits, interps = Hfunc

    Nfreq = len(freqs)
    Harr = np.zeros((Nfreq,3,3),dtype=np.complex128)

    ### Sample the Hfunc at the desired frequencies
    for drive in [0,1,2]:
        for resp in [0,1,2]:
            if suppress_off_diag and (drive != resp):
                continue
            interpolate = interps[resp][drive]
            fit = fits[resp][drive]
            if interpolate:
                oldfreqs = fit[0][0]
                oldmag = fit[0][1]
                oldphase = fit[1][1]

                mw = (1.0 / np.std(oldmag[:10])) * np.ones(len(oldfreqs))
                pw = (1.0 / np.std(oldphase[:10])) * np.ones(len(oldfreqs))
                magfunc = interp.UnivariateSpline(oldfreqs, oldmag, w=mw, k=2, s=smoothing)
                phasefunc = interp.UnivariateSpline(oldfreqs, oldphase, w=pw, k=2, s=smoothing)

                mag_extrap = \
                    make_extrapolator( magfunc, xs=oldfreqs, ys=oldmag, \
                                       pts=fit[0][2], arb_power_law=fit[0][3])

                phase_extrap = \
                    make_extrapolator( phasefunc, xs=oldfreqs, ys=oldphase, \
                                       pts=fit[1][2], arb_power_law=fit[1][3], semilogx=True)
                mag = mag_extrap(freqs)
                phase = phase_extrap(freqs)

            else:
                mag = bu.damped_osc_amp(freqs, *fit[0])
                phase = bu.damped_osc_phase(freqs, *fit[1], phase0=fit[2])

            if adjust_phase:
                adjust_key = '{:d}{:d}'.format(drive, resp)
                if adjust_key in adjust_phase_dict:
                    adjust_freqs = list(adjust_phase_dict[adjust_key].keys())
                    for freq in adjust_freqs:
                        freqind = np.argmin( np.abs(freqs - freq) )
                        phase[freqind] = adjust_phase_dict[adjust_key][freq]


            Harr[:,drive,resp] = mag * np.exp(1.0j * phase)

    ### Make the TF at the DC bin equal to the TF at the first 
    ### actual frequency bin. If using analytic functions for damped
    ### harmonic oscillators, these two should already be the same.
    ### If using an interpolating function with custom extrapolation, 
    ### this avoids singular matrices because usually the z-dirction 
    ### response goes to 0 at 0 frequency
    # Harr[0,:,:] = Harr[1,:,:]
    ### THIS IS COMMENTED BECAUSE NEW DATA DOESN"T INCLUDE A DC VALUE

    ### numPy's matrix inverse can handle an array of matrices
    Hout = np.linalg.inv(Harr)

    ### If the diagonal components are suppressed, sometimes the 
    ### inversion does some weird stuff so explicitly set the 
    ### off-diagonal components to 0 again
    if suppress_off_diag:
        for drive in [0,1,2]:
            for resp in [0,1,2]:
                if drive == resp:
                    continue
                Hout[:,drive,resp] = 0.0 + 0.0j

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

