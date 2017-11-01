import glob, os, sys, copy, time, math, pprocess

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy
import scipy.optimize as opti
import scipy.signal as signal
import scipy.interpolate as interp

import cant_util as cu
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

           OUTPUTS: Lorentzian amplitude of complex sum'''
    csum = damped_osc_amp(f, A1, f1, g1)*np.exp(1.j * damped_osc_phase(f, A1, f1, g1) ) \
           + damped_osc_amp(f, A2, f2, g2)*np.exp(1.j * damped_osc_phase(f, A2, f2, g2) ) \
           + damped_osc_amp(f, A3, f3, g3)*np.exp(1.j * damped_osc_phase(f, A3, f3, g3) )
    return np.angle(csum) + phase0


    


def build_uncalibrated_H(fobjs, average_first=True, dpsd_thresh = 8e-2, mfreq = 1., \
                         fix_HF=False):
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

    print "BUILDING H..."
    sys.stdout.flush()

    Hout = {}
    Hout_noise = {}

    Hout_counts = {}

    if average_first:
        avg_drive_fft = {}
        avg_data_fft = {}
        counts = {}

        for fobj in fobjs:

            dfft = np.fft.rfft(fobj.electrode_data) #fft of electrode drive in daxis. 
            data_fft = np.fft.rfft(fobj.pos_data)

            N = np.shape(fobj.pos_data)[1]#number of samples
            fsamp = fobj.fsamp

            fft_freqs = np.fft.rfftfreq(N, d=1.0/fsamp)

            dpsd = np.abs(dfft)**2 * 2./(N*fobj.fsamp) #psd for all electrode drives   
            inds = np.where(dpsd>dpsd_thresh)#Where the dpsd is over the threshold for being used.
            eind = np.unique(inds[0])[0]

            if eind not in avg_drive_fft:
                avg_drive_fft[eind] = np.zeros(dfft.shape, dtype=np.complex128)
                avg_data_fft[eind] = np.zeros(data_fft.shape, dtype=np.complex128)
                counts[eind] = 0.

            avg_drive_fft[eind] += dfft
            avg_data_fft[eind] += data_fft
            counts[eind] += 1.

        for eind in counts.keys():
            avg_drive_fft[eind] = avg_drive_fft[eind] / counts[eind]
            avg_data_fft[eind] = avg_data_fft[eind] / counts[eind]

        for eind in avg_drive_fft.keys():
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

            # Extract the TF, (response / drive), where the drive was above a 
            # fixed threshold.
            Hmatst_good = Hmatst[:,:,finds[b]]

            # Generate a integer by which to roll the data_fft to compute the noise
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
                    if i != 0 and fix_HF:
                        sep = freq - freqs[i-1]
                        # Clause to ignore this particular frequency response if an
                        # above threshold response is found not on a drive bin. Sometimes
                        # random noise components pop up or some power leaks to a 
                        # neighboring bin
                        if sep < 0.9 * (freqs[1] - freqs[0]):
                            continue
                    Hout[freq] = np.zeros((3,3), dtype=np.complex128)
                    Hout_noise[freq] = np.zeros((3,3), dtype=np.complex128)

                # Add the response from this drive freq/electrode pair to the TF matrix
                Hout[freq][:,outind] += Hmatst_good[:,eind,i]
                Hout_noise[freq][:,outind] += Hmatst_noise[:,eind,i]

            # This next bit of code goes over the computed transfer function and cleans it
            # up a little. Drive frequencies were chosen with a paricular linear spacing
            # so if we somehow found a drive/response pair that was closely or incorrectly
            # spaced, we sum the TF matrices over those closely spaced frequencies.
            # 
            # Often, this error seemed to result when the code would find an above threshold
            # drive in one channel at a particular frequency, but NOT in the other channels
            # thus it would generate a pair of matrices like
            #                      [[0., 0., a]                        [[d, g, 0.]
            # Hout = { ..., 41 Hz:  [0., 0., b]     ,   41.0000001 Hz:  [e, h, 0.]    , .... }
            #                       [0., 0., c]]                        [f, i, 0.]]
            # And thus we sum over these closely spaced matrices and take the average of 
            # frequencies as a new key


    if not average_first:
        for obj in dir_obj.fobjs:
            einds = obj.H.electrodes
            finds = obj.H.finds
            freqs = obj.fft_freqs[finds]

            for i, freq in enumerate(freqs):
                if freq not in Hout:
                    Hout[freq] = np.zeros((3,3), dtype=np.complex128)
                    Hout_noise[freq] = np.zeros((3,3), dtype=np.complex128)
                    Hout_counts[freq] = np.zeros(3)

                outind = emap(einds[i])
                Hout[freq][:,outind] += obj.H.Hmats[:,einds[i],i]
                Hout_noise[freq][:,outind] += obj.noiseH.Hmats[:,einds[i],i]
                Hout_counts[freq][outind] += 1

        # Compute the average transfer function
        for key in Hout.keys():
            for i in [0,1,2]:
                Hout[key][:,i] = Hout[key][:,i] / Hout_counts[key][i]
                Hout_noise[key][:,i] = Hout_noise[key][:,i] / Hout_counts[key][i]


    #if fix_HF:
    #    keys = Hout.keys()
    #    keys.sort()
    #
    #    freqsep = keys[1] - keys[0]
    #    freqsep = freqsep * 0.9
    #
    #    curr_sum = np.zeros((3,3), dtype=np.complex128)
    #    curr_freqs = []
    #    count = 0
    #    fixing = False
    #
    #    for i, key in enumerate(keys):
    #
    #        if key == keys[0]:
    #            continue
    #
    #        if ((keys[i] - keys[i-1]) < freqsep) and not fixing:
    #            fixing = True
    #            curr_freqs.append(keys[i-1])
    #            curr_sum += Hout[keys[i-1]]
    #            count += 1
    #
    #        if fixing:
    #            curr_freqs.append(keys[i])
    #            curr_sum += Hout[keys[i]]
    #            count += 1
    #            if i == len(keys) - 1:
    #                continue
    #            else:
    #                if keys[i+1] - keys[i] >= freqsep:
    #                    fixing = False
    #
    #                for freq in curr_freqs:
    #                    del Hout[freq]
    #
    #                    newfreq = np.mean(curr_freqs)
    #                    Hout[newfreq] = curr_sum
    #                    curr_freqs = []
    #                    curr_sum = np.zeros((3,3), dtype=np.complex128)



    first_mats = []
    freqs = Hout.keys()
    freqs.sort()
    for freq in freqs[:1]:
        first_mats.append(Hout[freq])
    first_mats = np.array(first_mats)

    init_phases = np.mean(np.angle(first_mats), axis=0)
    for drive in [0,1,2]:
        if np.abs(init_phases[drive,drive]) > 2.0:
            print "Correcting phase shift for drive channel", drive
            sys.stdout.flush()
            for freq in freqs:
                Hout[freq][:,drive] = Hout[freq][:,drive] * (-1)

    return Hout, Hout_noise













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

    print "CALIBRATING H FROM SINGLE-CHARGE STEP..."
    sys.stdout.flush()
    freqs = np.array(Hout.keys())
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
    print outstr

    Hout_cal = {}
    for freq in freqs:
        # Normalize transfer functions by charge number
        # and convert to force with capacitor plate separation
        # F = q*E = q*(V/d) so we take
        # (Vresp / Vdrive) * d / q = Vresp / Fdrive
        Hout_cal[freq] = np.copy(Hout[freq]) * (plate_sep / q)

    return Hout_cal








        

def build_Hfuncs(Hout_cal, fit_freqs = [10.,600], fpeaks=[400.,400.,50.], \
                 weight_peak=False, weight_lowf=False, lowf_thresh=60., \
                 weight_phase=False, plot_fits=False, plot_inits=False, \
                 grid = False, fit_osc_sum=False, deweight_peak=False, \
                 interpolate = False, max_freq=300):
    # Build the calibrated transfer function array
    # i.e. transfer matrices at each frequency and fit functions to each component

    keys = Hout_cal.keys()
    keys.sort()

    keys = np.array(keys)

    mats = []
    for freq in keys:
        mat = Hout_cal[freq]
        mats.append(mat)

    mats = np.array(mats)
    fits = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   

    if plot_fits:
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
                magfunc = interp.interp1d(keys[b], mag[b], kind='cubic', \
                                          fill_value=(mag[b][0],0), bounds_error=False)
                phasefunc = interp.interp1d(keys[b], unphase[b], kind='cubic', \
                                            fill_value=(unphase[b][0], 0), bounds_error=False)
                fits[resp][drive] = (magfunc, phasefunc)
                if plot_fits:
                    pts = np.linspace(np.min(keys), np.max(keys), len(keys) * 10)
                    

                    if grid:
                        axarr1[resp,drive].grid()
                        axarr2[resp,drive].grid()

                    axarr1[resp,drive].loglog(keys, mag)
                    axarr1[resp,drive].loglog(pts, magfunc(pts), color='r', linewidth=3)

                    axarr2[resp,drive].semilogx(keys, unphase)
                    axarr2[resp,drive].semilogx(pts, phasefunc(pts), color='r', linewidth=3)
                continue


            # Make initial guess based on high-pressure thermal spectra fits
            #therm_fits = dir_obj.thermal_cal_fobj.thermal_cal
            if (drive == 2) or (resp == 2):
                # Z-direction is considerably different than X or Y
                Amp = 1e17
                f0 = 50
                g = f0 * 5.0
                fit_freqs = [1.,200.]
                fpeak = fpeaks[2]
            else:
                Amp = 1e20
                f0 = 300
                g = f0
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
                    fac = 1.0
                weights = weights + fac * np.exp(-(npkeys-fpeak)**2 / (2 * 600) )
            if weight_lowf:
                ind = np.argmin(np.abs(npkeys - lowf_thresh))
                weights[:ind] *= 0.1
            phase_weights = np.zeros(len(npkeys)) + 1.
            if weight_phase and (drive != 2 and resp != 2):
                ind = np.argmin(np.abs(npkeys - 50.))
                phase_weights[:ind] *= 0.05

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

            #print drive, resp, phase_resids
            mult = np.argmin(phase_resids)
            if mult == 2:
                mult = -1

            popt_phase = phase_fits[mult]
            phase0 = np.pi * mult

            fits[resp][drive] = (popt_mag, popt_phase, phase0)

            if plot_fits:

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
                axarr1[resp,drive].loglog(keys[b], fitmag, color='r', linewidth=3)
                if plot_inits:
                    axarr1[resp,drive].loglog(keys[b], maginit, color='k', linewidth=2)

                axarr2[resp,drive].semilogx(keys, unphase)
                axarr2[resp,drive].semilogx(keys[b], fitphase, color='r', linewidth=3)
                if plot_inits:
                    axarr2[resp,drive].semilogx(keys[b], phaseinit, color='k', linewidth=2)



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

                    axarr4[resp,drive].semilogx(keys, unphase)
                    axarr4[resp,drive].semilogx(keys[b], fitphase, color='r', linewidth=3)

    if plot_fits:

        for drive in [0,1,2]:
            axarr1[0, drive].set_title("Drive in direction \'%i\'"%drive)
            axarr1[2, drive].set_xlabel("Frequency [Hz]")

            axarr2[0, drive].set_title("Drive in direction \'%i\'"%drive)
            axarr2[2, drive].set_xlabel("Frequency [Hz]")

            if fit_osc_sum:

                axarr3[0, drive].set_title("Drive in direction \'%i\'"%drive)
                axarr3[2, drive].set_xlabel("Frequency [Hz]")

                axarr4[0, drive].set_title("Drive in direction \'%i\'"%drive)
                axarr4[2, drive].set_xlabel("Frequency [Hz]")

        for response in [0,1,2]:
            axarr1[response, 0].set_ylabel("Resp in \'%i\' [V/N]" %response)
            axarr2[response, 0].set_ylabel("Resp in \'%i\' [rad]" %response)

            if fit_osc_sum:

                axarr3[response, 0].set_ylabel("Resp in \'%i\' [V/N]" %response)
                axarr4[response, 0].set_ylabel("Resp in \'%i\' [rad]" %response)

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
