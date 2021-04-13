import sys, time, traceback

import numpy as np

import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter

import bead_util as bu
import configuration as config

import scipy.signal as signal
import scipy.optimize as optimize
import scipy.constants as constants

import sklearn.cluster as cluster

plt.rcParams.update({'font.size': 14})

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
    # delta_x = np.mean(np.diff(xs))
    # return q*(xs<=(x0+0.75*delta_x))
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





def find_step_cal_response(file_obj, bandwidth=1., include_in_phase=False, \
                           using_tabor=False, tabor_ind=3, mon_fac=100, \
                           ecol=-1, pcol=-1, new_trap=False, plot=False, \
                           userphase=0.0, nearest=True):
    '''Analyze a data step-calibraiton data file, find the drive frequency,
       correlate the response to the drive

       INPUTS:   file_obj, input file object
                 bandwidth, bandpass filter bandwidth

       OUTPUTS:  H, (response / drive)'''

    if new_trap:
        using_tabor = False

    if not using_tabor:
        if pcol == -1:
            if ecol == -1:
                ecol = np.argmax(file_obj.electrode_settings['driven'])
            pcol = config.elec_map[ecol]

        efield = bu.trap_efield(file_obj.electrode_data, new_trap=new_trap)
        drive = efield[pcol]

        #drive = file_obj.electrode_data[ecol]
        if plot:
            fig, axarr = plt.subplots(2,1,sharex=True)
            tvec = np.arange(file_obj.nsamp) * (1.0 / file_obj.fsamp)
            colors = bu.get_color_map(len(file_obj.electrode_data), cmap='plasma')
            for i in range(len(file_obj.electrode_data)):
                if file_obj.electrode_settings['driven'][i]:
                    ext = ' - driven'
                else:
                    ext = ''
                axarr[0].plot(tvec, file_obj.electrode_data[i], color=colors[i], 
                            label='Elec. {:s}{:s}'.format(str(i), ext))
            axarr[0].set_title('Electrode and Efield Data')
            axarr[0].set_ylabel('Voltage [V]')
            axarr[0].legend(fontsize=10, ncol=2, loc='upper right')

            for i, ax in enumerate(['X', 'Y', 'Z']):
                axarr[1].plot(tvec, efield[i], label=ax)
            axarr[1].set_ylabel('Efield [V/m]')
            axarr[1].set_xlabel('Time [s]')
            axarr[1].legend(fontsize=10, loc='upper right')

            fig.tight_layout()
            fig.show()
            input()

        # plt.plot(drive)
        # plt.show()
        #drive = efield[ecol]

    elif using_tabor:
        pcol = 0
        v3 = file_obj.other_data[tabor_ind] * mon_fac
        v4 = file_obj.other_data[tabor_ind+1] * mon_fac
        zeros = np.zeros(len(v3))

        if plot:
            colors = bu.get_color_map(2, cmap='plasma')
            plt.figure()
            plt.plot(v3, color=colors[0], label='Elec. {:s}'.format(str(tabor_ind)))
            plt.plot(v4, color=colors[1], label='Elec. {:s}'.format(str(tabor_ind+1)))
            plt.title('Electrode data [V]')
            plt.legend()
            plt.tight_layout()
            plt.show()

            input()

        fac = 1.0
        if np.std(v4) < 0.5 * np.std(v3):
            # print('Only one Tabor drive channel being digitized...')
            v4 = zeros
            fac = 2.0
        elif np.std(v3) < 0.5 * np.std(v4):
            # print('Only one Tabor drive channel being digitized...')
            v3 = zeros
            fac = 2.0

        voltages = []
        for i in range(8):
            if i == tabor_ind:
                voltages.append(v3)
            elif i == (tabor_ind + 1):
                voltages.append(v4)
            else:
                voltages.append(zeros)

        drive = bu.trap_efield(voltages, new_trap=new_trap)[pcol] * fac

    # try:
    #     power = np.mean(file_obj.power)
    # except Exception:
    #     power = 0.0
    #     traceback.print_exc()

    zpos = np.mean(file_obj.pos_data[2])

    #drive = bu.detrend_poly(drive, order=1.0, plot=True)
    drive_fft = np.fft.rfft(drive)

    ### Find the drive frequency
    freqs = np.fft.rfftfreq(len(drive), d=1./file_obj.fsamp)
    drive_freq = freqs[np.argmax(np.abs(drive_fft[1:])) + 1]

    # plt.plot(drive)
    # plt.show()
    # input()

    # print(drive_freq)
    # for i in range(3):
    #     plt.plot(efield[i], label=str(i))
    # plt.legend()
    # plt.show()

    ### Extract the response and detrend
    # response = file_obj.pos_data[pcol]
    if new_trap:
        response = file_obj.pos_data_3[pcol]
    else:
        response = file_obj.pos_data[pcol]
    #response = bu.detrend_poly(response, order=1.0, plot=True)


    ### Configure a time array for plotting and fitting
    cut_samp = config.adc_params["ignore_pts"]
    N = len(drive)
    dt = 1. / file_obj.fsamp
    t = np.linspace(0,(N+cut_samp-1)*dt, N+cut_samp)
    t = t[cut_samp:]

    # print(drive_freq)
    # if drive_freq < 10.0:
    #     print(file_obj.fname)
    #     plt.plot(t, drive)
    #     plt.figure()
    #     plt.loglog(freqs, np.abs(drive_fft))
    #     plt.show()

    if drive_freq < 0.5*bandwidth:
        apply_filter = False
    else:
        apply_filter = True

    ### Bandpass filter the response
    if apply_filter:
        b, a = signal.butter(3, [2.*(drive_freq-bandwidth/2.)/file_obj.fsamp, \
                              2.*(drive_freq+bandwidth/2.)/file_obj.fsamp ], btype = 'bandpass')
        responsefilt = signal.filtfilt(b, a, response)
    else:
        responsefilt = np.copy(response)

    if plot:
        plt.figure()
        plt.loglog(freqs, np.abs(np.fft.rfft(drive)))
        plt.loglog(freqs, np.abs(np.fft.rfft(responsefilt)))
        plt.show()

        input()

    ### Compute the full, normalized correlation and extract amplitude
    corr_full = bu.correlation(drive, responsefilt, file_obj.fsamp, drive_freq)
    ncorr = len(corr_full)

    phase_ratio = userphase / (2.0 * np.pi)
    phase_inds = np.array([np.floor(phase_ratio*ncorr), np.ceil(phase_ratio*ncorr)], dtype='int')

    response_inphase = corr_full[0]
    response_max = np.max(corr_full)
    # try:
    response_userphase = np.interp([phase_ratio*ncorr], phase_inds, corr_full[phase_inds])[0]
    # except:
    #     response_userphase = corr_full[phase_inds[0]]

    ### Compute the drive amplitude, assuming it's a sine wave 
    drive_amp = np.sqrt(2) * np.std(drive) # Assume drive is sinusoidal
    # print(drive_amp)

    outdict = {}
    outdict['inphase']          = response_inphase   / drive_amp
    outdict['max']              = response_max       / drive_amp
    outdict['userphase']        = response_userphase / drive_amp
    outdict['userphase_nonorm'] = response_userphase
    outdict['drive']            = drive_amp
    outdict['drive_freq']       = drive_freq

    return outdict





def step_cal(step_cal_vec, nsec=10, amp_gain = 1., new_trap = False, \
             auto_try = 0.0, max_step_size=10, plot_residual_histograms=False):
    '''Generates a step calibration from a list of DataFile objects
           INPUTS: fobjs, list of file objects
                   plate_sep, face-to-face separation of electrodes
                   drive_freq, electrostatic drive freq during step_cal
                   amp_gain, gain of HV amplifier if noise is a problem

           OUTPUTS: vpn, volts of response per Newton of drive
                    err, 1 std.dev. error on vpn'''


    step_cal_vec = np.array(step_cal_vec)


    # yfit = np.abs(step_cal_vec)
    yfit = np.copy(step_cal_vec)
    # yfit = yfit[first_file:]

    #bvec = yfit == yfit #[yfit<10.*np.mean(yfit)] #exclude cray outliers
    #yfit = yfit[bvec] 

    if not auto_try:
        # plt.ion()
        plt.figure(1)
        plt.plot(np.arange(len(yfit)), yfit, 'o')
        plt.xlabel('Integration number [Arb]')
        plt.ylabel('Response [Arb]')
        plt.tight_layout()
        plt.show()
        # plt.show(block=False)
        # plt.draw()
        guess = input('Enter a guess for response / step: ')
        guess = float(guess)
        plt.close(1)
        # plt.ioff()
    else:
        guess = auto_try

    guess0 = guess

    step_inds = []
    step_qs = []
    step_sizes = []
    last_step = 0

    for i in range(len(yfit)):

        if i == 0:
            current_charge = [yfit[0]]
            continue

        diff = np.mean(current_charge) - yfit[i]
        diff_abs = np.abs(diff)

        cond1 = (diff_abs > 0.5 * guess)

        # if not new_trap:
        #     std = np.std(yfit[last_step+1:i-1])
        #     cond2 = diff_abs > 2.0 * std
        # else:
        #     cond2 = True
        cond2 = True

        if cond1 and cond2:
            
            current_charge = [yfit[i]]

            done = False
            for step_size in (np.arange(max_step_size) + 1)[::-1]:
                if (step_size == 1) or (diff_abs > (step_size - 0.5) * guess):
                    done = True

                if done:
                    step_sizes.append(diff_abs * (1.0 / step_size))
                    step_qs.append(np.sign(diff) * step_size)
                    break

            step_inds.append((i-1)*nsec)
            last_step = step_inds[-1]

            guess = np.mean(step_sizes + [guess0])

        else:
            current_charge.append(yfit[i])

    vpq_guess = np.mean(step_sizes)

    def ffun(x, vpq, offset):
        qqs = vpq * np.array(step_qs)
        offarr = np.zeros(len(x)) + offset
        return multi_step_fun(x, qqs, step_inds) + offarr
    
    xfit = np.arange(len(yfit)) * nsec

    p0 = [vpq_guess, 0]#Initial guess for the fit

    popt, pcov = optimize.curve_fit(ffun, xfit, yfit, p0 = p0, xtol = 1e-12)

    fitobj = Fit(popt, pcov, ffun)

    newpopt = np.copy(popt)
    newpopt[1] = 0.0

    normfitobj = Fit(newpopt / popt[0], pcov / popt[0], ffun)

    f, axarr = plt.subplots(2, sharex = True, \
                            gridspec_kw = {'height_ratios':[2,1]}, \
                            figsize=(10,5),dpi=150)#Plot fit
    normfitobj.plt_fit(xfit, (yfit - popt[1]) / popt[0], axarr[0], \
                       ms=3, ylabel="Norm. Response [$e$]", xlabel="")
    normfitobj.plt_residuals(xfit, (yfit - popt[1]) / popt[0], axarr[1], \
                             ms=3, xlabel="Time [s]")

    fit_ylim = axarr[0].get_ylim()
    for val in fit_ylim:
        if np.abs(val) > 15.0:
            fit_majorspace = 5.0
            break
        elif np.abs(val) > 4.0:
            fit_majorspace = 2.0
            break
        else:
            fit_majorspace = 1.0

    resid_ylim = axarr[1].get_ylim()
    too_small = False
    for val in resid_ylim:
        if np.abs(val) < 1.0:
            too_small = True
    # if too_small:
    axarr[1].set_ylim(-1.1, 1.1)
    resid_majorspace = 1.0
    # else:
    #     resid_majorspace = 2.0

    normfitobj.setup_discharge_ticks(axarr, fit_majorspace=fit_majorspace, \
                                     resid_majorspace=resid_majorspace)
    # for x in xfit:
    #     if not (x-1) % 3:
    #         axarr[0].axvline(x=x, color='k', linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()

    happy = input("does the fit look good? (Y/n): ")
    if happy == 'y' or happy == 'Y':
        happy_with_fit = True
    elif happy == 'n' or happy == 'N':
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
        #plt.ion()
        plt.plot(yfit, 'o')
        plt.show()

        print("MANUAL STEP CALIBRATION")
        print("Enter guess at number of steps and charge at steps [[q1, q2, q3, ...], [x1, x2, x3, ...], vpq]")
        
        try:
            nstep = eval(input(": "))
            plt.close(1)

            step_qs = nstep[0]
            step_inds = np.array(nstep[1]) * nsec

            p0 = [nstep[2],0.0]#Initial guess for the fit
            popt, pcov = optimize.curve_fit(ffun, xfit, yfit, p0 = p0, xtol = 1e-10)

        except Exception:
            plt.close(1)
            continue


        fitobj = Fit(popt, pcov, ffun)#Store fit in object.

        newpopt = np.copy(popt)
        newpopt[1] = 0.0

        normfitobj = Fit(newpopt / popt[0], pcov / popt[0], ffun)

        resids = ((yfit - popt[1]) / popt[0]) - ffun(xfit, *(newpopt/popt[0]))
        npts = len(resids)

        f, axarr = plt.subplots(2, sharex = True, \
                                gridspec_kw = {'height_ratios':[2,1]}, \
                                figsize=(10,4), dpi=150)#Plot fit

        normfitobj.plt_fit(xfit, (yfit - popt[1]) / popt[0], axarr[0], \
                           ms=3, ylabel="Norm. Response [$e$]", xlabel="")
        normfitobj.plt_residuals(xfit, (yfit - popt[1]) / popt[0], axarr[1], \
                                 ms=3, ylabel='Resid.', xlabel="Time [s]")
        axarr[1].set_ylim(-1.1, 1.1)
        normfitobj.setup_discharge_ticks(axarr)

        f.tight_layout()

        if plot_residual_histograms:
            f2, axarr2 = plt.subplots(1, 3, sharex=True, sharey=True, \
                                      figsize=(10,2.5), dpi=150)

            ind1 = int( npts / 3.0 )
            ind2 = int( 2.0 * npts / 3.0 )
            nbins = int( np.min([np.max([10, int(ind1 / 10)]), 20.0]) )

            vals1, bins1, _ = axarr2[0].hist(resids[:ind1], bins=nbins, range=(-2,2))
            vals2, bins2, _ = axarr2[1].hist(resids[ind1:ind2], bins=nbins, range=(-2,2))
            vals3, bins3, _ = axarr2[2].hist(resids[ind2:], bins=nbins, range=(-2,2))
            maxval = np.max( np.concatenate((vals1, vals2, vals3)) )

            labels = ['First 3rd', 'Middle 3rd', 'Last 3rd']
            for i in [0,1,2]:
                axarr2[i].tick_params(axis='y', which='both', right=False, \
                                     left=False, labelleft=False)
                axarr2[i].set_xticks([-2,-1,0,1,2])
                axarr2[i].set_ylim(0, 1.25*maxval)
                axarr2[i].text(0, 1.125*maxval, labels[i], fontsize=14, \
                               verticalalignment='center', horizontalalignment='center')

            axarr2[1].set_xlabel('Residuals [$e^{-}$]')
            f2.tight_layout()

        plt.show()

        happy = input("does the fit look good? (Y/n): ")
        if happy == 'y' or happy == 'Y':
            happy_with_fit = True
        elif happy == 'n' or happy == 'N':
            f.clf()
            continue
        else:
            f.clf()
            print('that was a yes or no question... assuming you are unhappy')
            sys.stdout.flush()
            time.sleep(5)
            continue

    plt.close('all')

    q0_sc = ffun([0], *fitobj.popt)[0]
    q0 = int(round(q0_sc / fitobj.popt[0]))
    print('q0: ', q0)

    ### Scale response by the fundamental charge so that it's in units of
    ### (response amplitude of 1e) / (force on 1e)
    e_charge = constants.elementary_charge

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

    def plt_fit(self, xdata, ydata, ax, scale = 'linear', ms = 6, \
                    xlabel = 'X', ylabel = 'Y', errors = [], zorder=2):
    
        inds = np.argsort(xdata)
        
        xdata = xdata[inds]
        ydata = ydata[inds]

        delta_x = np.mean(np.diff(xdata))
        xfundata = np.linspace(np.min(xdata) - 2*delta_x, np.max(xdata) + 2*delta_x, \
                                int(10.0 * len(xdata)))

        #modifies an axis object to plot the fit.
        if len(errors):
            ax.errorbar(xdata, ydata, errors, fmt = 'o', ms = ms, zorder=zorder)
            ax.plot(xfundata, self.fun(xfundata, *(self.popt)), \
                        'r', linewidth = 3, zorder=zorder+1)

        else:    
            ax.plot(xdata, ydata, 'o', ms = ms, zorder=zorder)
            ax.plot(xfundata + 0.5*delta_x, self.fun(xfundata, *(self.popt)), \
                        'r', linewidth = 3, zorder=zorder+1)

        ax.set_yscale(scale)
        ax.set_xscale(scale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([np.min(xdata), np.max(xdata)])
    
    def plt_residuals(self, xdata, ydata, ax, scale = 'linear', ms = 6, \
                        xlabel = 'X', ylabel = 'Residual', label = '', errors = [], \
                        zorder=2):
        #modifies an axis object to plot the residuals from a fit.

        inds = np.argsort(xdata)
        
        xdata = xdata[inds]
        ydata = ydata[inds]

        #print np.std( self.fun(xdata, *self.popt) - ydata )

        if len(errors):
            ax.errorbar(xdata, self.fun(xdata, *self.popt) - ydata, errors, \
                        fmt = 'o', ms = ms, zorder=zorder)
        else:
            ax.plot(xdata, (self.fun(xdata, *self.popt) - ydata), 'o', \
                    ms = ms, zorder=zorder)
        
        #ax.set_xscale(scale)
        ax.set_yscale(scale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([np.min(xdata), np.max(xdata)])


    def setup_discharge_ticks(self, axarr, fit_majorspace=5, resid_majorspace=1.0, \
                              major_alpha=0.8, minor_alpha=0.3, xgrid=False):
        '''Sets up the tick marks and labels for the yaxis of the discharge
           plots. This code was being called multiple times identically so 
           now it's a class method. It's not a stanalone function because it 
           implicitly assumes axarr is length-2 array containing the axes for
           the fit data and the residuals. '''

        axarr[0].yaxis.set_major_locator(MultipleLocator(fit_majorspace))
        axarr[1].yaxis.set_major_locator(MultipleLocator(resid_majorspace))
        for i in [0,1]:
            axarr[i].yaxis.set_minor_locator(MultipleLocator(1))
            axarr[i].yaxis.set_minor_formatter(NullFormatter())
            if xgrid:
                axarr[i].grid(True, which='minor', axis='x', alpha=minor_alpha)
                axarr[i].grid(True, which='major', axis='x', alpha=major_alpha)

            ylim = axarr[i].get_ylim()
            if i == 0:
                init_sign = np.sign(ylim[np.argmax(np.abs(ylim))])
                if init_sign < 0:
                    tick_locs = np.arange(int(2.0 * ylim[0]), 10, 1)
                else:
                    tick_locs = np.arange(-10, int(2.0 * ylim[1]), 1)
            else:
                tick_locs = np.arange(-5, 6, 1)

            axarr[i].grid(False, which='both', axis='y')
            for tick in tick_locs:
                if tick == 0.0:
                    continue
                elif not tick % fit_majorspace:
                    axarr[i].axhline(tick, color='grey', alpha=major_alpha, lw=0.5)
                else:
                    axarr[i].axhline(tick, color='grey', alpha=minor_alpha, lw=0.5)
            axarr[i].axhline(0, ls='--', color='k', zorder=1)
            axarr[i].set_ylim(*ylim)




    def css(self, xdata, ydata, yerrs, p):
        #returns the chi square score at a point in fit parameters.
        return np.sum((ydata))
