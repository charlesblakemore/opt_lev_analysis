import numpy as np
import matplotlib
import bead_util as bu
import scipy
import glob, os, sys, copy, time, math, pprocess
from scipy.optimize import curve_fit
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.signal as sig
from multiprocessing import Pool


## Helper functions


def round_sig(x, sig=2):
    # round a number to a certain number of sig figs
    if x == 0:
        return 0
    else:
        return round(x, sig-int(math.floor(math.log10(x)))-1)

def trend_fun(x, a, b):
    # Define a simple linear function to de-trend datasets
    return a*x + b

def damped_osc_amp(f, A, f0, g):
    # Fitting function for the AMPLITUDE of a damped harmonic
    # oscillator potential.
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    denom = np.sqrt((w0**2 - w**2)**2 + w**2 * g**2)
    return A / denom

def damped_osc_phase(f, A, f0, g, phase0 = 0.):
    # Fitting function for the PHASE of a damped harmonic
    # oscillator potential
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    return A * np.arctan2(-w * g, w0**2 - w**2) + phase0


def sum_3osc_amp(f, A1, f1, g1, A2, f2, g2, A3, f3, g3):
    csum = damped_osc_amp(f, A1, f1, g1)*np.exp(1.j * damped_osc_phase(f, A1, f1, g1) ) \
           + damped_osc_amp(f, A2, f2, g2)*np.exp(1.j * damped_osc_phase(f, A2, f2, g2) ) \
           + damped_osc_amp(f, A3, f3, g3)*np.exp(1.j * damped_osc_phase(f, A3, f3, g3) )
    return np.abs(csum)

def sum_3osc_phase(f, A1, f1, g1, A2, f2, g2, A3, f3, g3, phase0=0.):
    csum = damped_osc_amp(f, A1, f1, g1)*np.exp(1.j * damped_osc_phase(f, A1, f1, g1) ) \
           + damped_osc_amp(f, A2, f2, g2)*np.exp(1.j * damped_osc_phase(f, A2, f2, g2) ) \
           + damped_osc_amp(f, A3, f3, g3)*np.exp(1.j * damped_osc_phase(f, A3, f3, g3) )
    return np.angle(csum) + phase0

    
def thermal_psd_spec(f, A, f0, g):
    #The position power spectrum of a microsphere normalized so that A = (volts/meter)^2*2kb*t/M
    w = 2.*np.pi*f #Convert to angular frequency.
    w0 = 2.*np.pi*f0
    num = g * w0**2
    denom = ((w0**2 - w**2)**2 + w**2*g**2)
    return 2 * (A * num / denom) # Extra factor of 2 from single-sided PSD

def step_fun(x, q, x0):
    #decreasing step function at x0 with step size q.
    xs = np.array(x)
    return q*(xs<=x0)

def multi_step_fun(x, qs, x0s):
    #Sum of step functions for fitting charge step calibration to.
    rfun = 0.
    for i, x0 in enumerate(x0s):
        rfun += step_fun(x, qs[i], x0)
    return rfun
    

sf = lambda tup: tup[0] #Sort key for sort_pts.

def sort_pts(xvec, yvec):
    #sorts yvec and xvec to put in order of increasing xvec for plotting
    zl = zip(xvec, yvec)
    zl = sorted(zl, key = sf)
    xvec, yvec = zip(*zl)
    return np.array(xvec), np.array(yvec)

def emap(eind):
    # Map from electrode number to data axis
    # Sign convention set from transfer function phase
    if eind == 1 or eind == 2:
        return 2
    elif eind == 3 or eind == 4:
        return 0
    elif eind == 5 or eind == 6:
        return 1

def emap2(drive):
    # Map from data axis back to electrode number, using nominal elecs
    # 1, 3 and 5, as TF data was taken with these electrodes
    if drive == 0:
        return 5
    elif drive == 1:
        return 3
    elif drive == 2:
        return 1







## Class for Fit Objects



class Fit:
    # Holds the optimal parameters and errors from a fit. 
    # Contains methods to plot the fit, the fit data, and the residuals.
    def __init__(self, popt, pcov, fun):
        self.popt = popt
        try:
            self.errs = np.diagonal(pcov)
        except ValueError:
            self.errs = "Fit failed"
        self.fun = fun

    def plt_fit(self, xdata, ydata, ax, scale = 'linear', xlabel = 'X', ylabel = 'Y', errors = []):
        xdata, ydata = sort_pts(xdata, ydata)
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
        xdata, ydata = sort_pts(xdata, ydata)
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
        

def thermal_fit(psd, freqs, fit_freqs = [1., 500.], temp = 300., fudge_fact = 1e-6, p0=[]):
    ## Function to fit the thermal spectra of a bead's motion
    ## First need good intitial guesses for fit parameters.

    # Boolian vector of frequencies over which the fit is performed
    fit_bool = bu.inrange(freqs, fit_freqs[0], fit_freqs[1]) 

    # guess resonant frequency from hightest part of spectrum
    f0 = freqs[np.argmax(psd[fit_bool])] 
    df = freqs[1] - freqs[0] #Frequency increment.
    
    # Guess at volts per meter using equipartition
    vpmsq = bu.bead_mass/(bu.kb*temp)*np.sum(psd[fit_bool])*df*len(psd)/np.sum(fit_bool) 
    g0 = 1./2.*f0 # Guess at damping assuming critical damping
    A0 = vpmsq*2.*bu.kb*temp/(bu.bead_mass*fudge_fact)
    if len(p0) == 0:
        p0 = [A0, f0, g0] #Initial parameter vectors 

    psd = psd.reshape((len(psd),))
    popt, pcov = curve_fit(thermal_psd_spec, freqs[fit_bool], psd[fit_bool], \
                           p0 = p0)#, sigma=weights)#, bounds = bounds)
    #print popt
    #popt[0] = popt[0]
    if not np.shape(pcov):
        print 'Warning: Bad fit'
    f = Fit(popt, pcov, thermal_psd_spec)
    return f

def sbin(xvec, yvec, bin_size):
    #Bins yvec based on binning xvec into bin_size
    fac = 1./bin_size
    bins_vals = np.around(fac*xvec)
    bins_vals /= fac
    bins = np.unique(bins_vals)
    y_binned = np.zeros_like(bins)
    y_errors = np.zeros_like(bins)
    for i, b in enumerate(bins):
        idx = bins_vals == b
        y_binned[i] = np.mean(yvec[idx])
        y_errors[i] = scipy.stats.sem(yvec[idx])
    return bins, y_binned, y_errors

def sbin_pn(xvec, yvec, bin_size, vel_mult = 0.):
    #Bins yvec based on binning xvec into bin_size for velocities*vel_mult>0.
    fac = 1./bin_size
    bins_vals = np.around(fac*xvec)
    bins_vals /= fac
    bins = np.unique(bins_vals)
    y_binned = np.zeros_like(bins)
    y_errors = np.zeros_like(bins)
    if vel_mult:
        vb = np.gradient(xvec)*vel_mult>0.
        yvec2 = yvec[vb]
    else:
        vb = yvec == yvec
        yvec2 = yvec

    for i, b in enumerate(bins):
        idx = bins_vals[vb] == b
        y_binned[i] = np.mean(yvec2[idx])
        y_errors[i] = scipy.stats.sem(yvec2[idx])
    return bins, y_binned, y_errors


def get_h5files(dir):
    files = glob.glob(dir + '/*.h5') 
    files = sorted(files, key = bu.find_str)
    return files


def simple_loader(fname, sep):
    #print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    return fobj

def diag_loader(fname, sep):
    #Generate all of the position attibutes of interest for a single file. Returns a Data_file object.
    #print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.detrend()
    fobj.get_fft()
    fobj.spatial_bin()
    return fobj

def pos_loader(fname, sep):
    #Generate all of the position attibutes of interest for a single file. Returns a Data_file object.
    #print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    #fobj.ms()
    fobj.detrend()
    fobj.spatial_bin()
    fobj.close_dat(ft=False)
    return fobj

def ft_loader(fname, sep):
    # Load files and computer FFTs. For testing out diagonalization
    #print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.detrend()
    fobj.get_fft()
    return fobj

def H_loader(fname, sep):
    #Generates transfer func data for a single file. Returns a Data_file object.
    #print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.find_H()
    fobj.detrend()
    #fobj.close_dat(p=False,ft=False,elecs=False)
    return fobj


#define a class with all of the attributes and methods necessary for processing a single data file to 
    

class Hmat:
    #this class holds transfer matricies between electrode drives and bead response.
    def __init__(self, finds, electrodes, Hmats):
        self.finds = finds #Indicies of frequences where there is an electrode being driven above threshold 
        self.electrodes = electrodes #the electrodes where there is statistically significant signal
        self.Hmats = Hmats #Transfer matrix at the frequencies 



class Data_file:
    #This is a class with all of the attributes and methods for a single data file.

    def __init__(self):
        self.fname = "Filename not assigned."
        #self.path = "Directory not assigned." #Assuming directory in filename
        self.pos_data = "bead position data not loaded"
        self.diag_pos_data = "Position Data not diagonalized"
        self.pos_data_cantfilt = "bead position data not filtered by cantilever"
        self.diag_pos_data_cantfilt = "diag position data not filtered by cantilever"
        self.image_pow_data = "Imaging power data not loaded"
        self.binned_image_pow_data = "Binned imaging power not computed"
        self.binned_image_pow_errs = "Binned imaging power errors not computed"
        self.dc_pos = "DC positions not computed"
        self.binned_pos_data = "Binned data not computed"
        self.binned_data_errors = "binded data errors not computed"
        self.cant_data = "cantilever position data no loaded"
        self.binned_cant_data = "Binned cantilever data not computed"
        self.separation = "separation not entered"
        self.Fsamp = "Fsamp not loaded"
        self.Time = "Time not loaded"
        self.temps = "temps not loaded"
        self.pressures = "pressures not loaded"
        self.synth_setting = "Synth setting not loaded"
        self.dc_supply_setting = "DC supply settings not loaded"
        self.electrode_data = "electrode data not loaded yet"
        self.electrode_settings = "Electrode settings not loaded"
        self.electrode_dc_vals = "Electrode potenitals not loaded"
        self.stage_settings = "Stage setting not loaded yet"
        self.psds = "psds not computed"
        self.data_fft = "fft not computed"
        self.diag_data_fft = "fft not diagonalized"
        self.fft_freqs = "fft freqs not computed"
        self.psd_freqs = "psd freqs not computed"
        self.thermal_cal = "Thermal calibration not computed"
        self.step_cal_response = "Step-cal response not computed"
        self.H = "bead electrode transfer function not computed"
        self.noiseH = "noise electrode transfer function not computed"
        self.Hcomponents = "bead electrode transfer function not computed"
        self.sb_spacing = "sideband spacing not computed."

    def load(self, fstr, sep, cant_cal = 8., stage_travel = 80., \
             cut_samp = 2000, elec_inds = [8, 9, 10, 11, 12, 13, 14]):
        # Methods to load the attributes from a single data file. 
        # sep is a vector of the distances of closes approach for 
        # each direction ie. [xsep, ysep, zsep] 
        dat, attribs, f = bu.getdata(fstr)
        
        self.fname = fstr
        
        dat = dat[cut_samp:, :]
        
        # Attributes coming from Labview Front pannel settings
        self.separation = sep #Manually entreed distance of closest approach
        self.Fsamp = attribs["Fsamp"] #Sampling frequency of the data
        self.Time = bu.labview_time_to_datetime(attribs["Time"]) #Time of end of file
        self.temps = attribs["temps"] #Vector of thermocouple temperatures 

        # Vector of chamber pressure readings [pirani, cold cathode, baratron]
        self.pressures = attribs["pressures"] 
        self.synth_settings = attribs["synth_settings"] #Synthesizer fron pannel settings
        self.dc_supply_settings = attribs["dc_supply_settings"] #DC power supply front pannel testings.

        # Electrode front pannel settings for all files in the directory.
        # first 8 are ac amps, second 8 are frequencies, 3rd 8 are dc vals 
        self.electrode_settings = attribs["electrode_settings"]

        # Front pannel settings applied to this particular file. 
        # Top boxes independent of the sweeps
        self.electrode_dc_vals = attribs["electrode_dc_vals"] 

        # Front pannel settings for the stage for this particular file.
        self.stage_settings = attribs['stage_settings'] 
        
        #Data vectors and their transforms
        self.pos_data = np.transpose(dat[:, 0:3]) #x, y, z bead position
        self.other_data = np.transpose(dat[:,3:7])
        self.dc_pos =  np.mean(self.pos_data, axis = -1)
        self.image_pow_data = np.transpose(dat[:, 6])

        #self.pos_data = np.transpose(dat[:,[elec_inds[1],elec_inds[3],elec_inds[5]]])
        self.cant_data = np.transpose(np.resize(sep, np.shape(np.transpose(self.pos_data)))) + stage_travel - np.transpose(dat[:, 17:20])*cant_cal
        self.electrode_data = np.transpose(dat[:, elec_inds]) #Record of voltages on the electrodes

        f.close()

    def get_stage_settings(self, axis=2):
        # Function to intelligently extract the stage settings data for a given axis
        if axis == 0:
            mask = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        elif axis == 1:
            mask = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
        elif axis == 2:
            mask = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

        return self.stage_settings[mask]

    def detrend(self):
        # Remove linear drift from data
        for i in [0,1,2]:
            dat = self.pos_data[i]
            x = np.array(range(len(dat)))
            popt, pcov = curve_fit(trend_fun, x, dat)
            self.pos_data[i] = dat - (popt[0]*x + popt[1])
            

    def ms(self):
        #mean subtracts the position data.
        ms = lambda vec: vec - np.mean(vec)
        self.pos_data  = map(ms, self.pos_data)


    def spatial_bin(self, bin_sizes = [1., 1., 1.], diag=False):
        #Method for spatially binning data based on stage z  position.
        
        if diag:
            dat = self.diag_pos_data
        else:
            dat = self.pos_data

        binned_image_pow_data = [[[], [], []], [[], [], []], [[], [], []]]
        binned_image_pow_errs = [[[], [], []], [[], [], []], [[], [], []]]

        binned_cant_data = [[[[], [], []], [[], [], []], [[], [], []]], \
                                 [[[], [], []], [[], [], []], [[], [], []]], \
                                 [[[], [], []], [[], [], []], [[], [], []]]] 
        binned_pos_data = [[[[], [], []], [[], [], []], [[], [], []]], \
                                [[[], [], []], [[], [], []], [[], [], []]], \
                                [[[], [], []], [[], [], []], [[], [], []]]]
        binned_data_errors = [[[[], [], []], [[], [], []], [[], [], []]], \
                                   [[[], [], []], [[], [], []], [[], [], []]], \
                                   [[[], [], []], [[], [], []], [[], [], []]]]

        for i, v in enumerate(dat):
            for j, pv in enumerate(self.cant_data):
                for si in np.arange(-1, 2, 1):
                    bins, y_binned, y_errors = \
                            sbin_pn(pv, v, bin_sizes[j], vel_mult = si)
                            #sbin_pn(self.cant_data[j], v, bin_sizes[j], vel_mult = si)
                    binned_cant_data[si][i][j] = bins
                    binned_pos_data[si][i][j] = y_binned 
                    binned_data_errors[si][i][j] = y_errors 

        for j, pv in enumerate(self.cant_data):
            for si in np.arange(-1, 2, 1):
                bins, ybinned, y_errors = \
                            sbin_pn(pv, self.image_pow_data, bin_sizes[j], vel_mult=si)
                binned_image_pow_data[si][j] = y_binned
                binned_image_pow_errs[si][j] = y_errors
    
        self.binned_image_pow_data = np.array(binned_image_pow_data)
        self.binned_image_pow_errs = np.array(binned_image_pow_errs)

        if diag:
            self.diag_binned_cant_data = np.array(binned_cant_data)
            self.diag_binned_pos_data = np.array(binned_pos_data)
            self.diag_binned_data_errors = np.array(binned_data_errors)

        else:
            self.binned_cant_data = np.array(binned_cant_data)
            self.binned_pos_data = np.array(binned_pos_data)
            self.binned_data_errors = np.array(binned_data_errors)


    def psd(self, NFFT = 2**16):
        #uses matplotlib mlab psd to take a psd of the microsphere position data.
        psder = lambda v: matplotlib.mlab.psd(v, NFFT = NFFT, Fs = self.Fsamp)[0]
        self.psds = np.array(map(psder, self.pos_data))
        self.psd_freqs = np.fft.rfftfreq(NFFT, d = 1./self.Fsamp)

        self.other_psds = np.array(map(psder, self.other_data))
        self.other_psd_freqs = np.fft.rfftfreq(NFFT, d = 1. / self.Fsamp) 

    def get_fft(self):
        #Uses numpy fft rfft to compute the fft of the position data
        # Does not normalize at all
        self.data_fft = np.fft.rfft(self.pos_data)
        self.fft_freqs = np.fft.rfftfreq(np.shape(self.pos_data)[1])*self.Fsamp


    def thermal_calibration(self, temp=293.):
        #Use thermal calibration calibrate voltage scale into physical units
        #Check to see if psds is computed and compute if not.
        if type(self.psds) == str:
            self.psd()

        caler = lambda v: thermal_fit(v, self.psd_freqs)

        self.thermal_cal = []
        for i in [0,1,2]:
            if i == 1:
                p0 = self.thermal_cal[0].popt
            elif i == 2:
                p0 = [1e-5, 20, 1. / 100.] 
            else:
                p0 = []

            if i == 2:
                fit_freqs = [1.,300.]
            else:
                fit_freqs = [1.,500.]
            newfit = thermal_fit(self.psds[i], self.psd_freqs, \
                                 fit_freqs = fit_freqs, p0=p0, temp=temp)

            # Since the PSD only cares about |omega| sometimes it fits to a 
            # negative frequency. The following corrects for that
            if newfit.popt[1] < 0:
                print newfit.popt[1]
                newfit.popt[1] *= (-1)

            self.thermal_cal.append(newfit)
        

    def get_thermal_cal_facs(self, temp=293.):
        # Function to return the N/V converstion factor derived
        # from the thermal calibration and treating the bead as
        # and ideal harmonic oscillator
        if type(self.thermal_cal) == str:
            self.thermal_calibration(temp=temp)
        out = []
        for i in [0,1,2]:
            fit_obj = self.thermal_cal[i]
            amp = fit_obj.popt[0]
            f = fit_obj.popt[1]
            omega = 2. * np.pi * f
            # Although the factor can be simplified, we leave it in the following
            # form to help make clear from where it is derived
            fac = (2*bu.kb*temp) / (bu.bead_mass * omega**2 * amp) \
                  * (bu.bead_mass * omega**2)**2
            out.append(fac)

        return np.sqrt(np.array(out))

    
    def plt_thermal_fit(self, coordinate = 0):
        #plots the thermal calibration and residuals
        if type(self.thermal_cal) == str:
            print "No thermal calibration"
        else:
            f, axarr = plt.subplots(3, sharex = True)
            print "plotting thermal fit"
            for i in [0,1,2]:
                if i == 0:
                    xlabel = ''
                    ylabel = 'X PSD [V^2/Hz]'
                elif i == 1:
                    xlabel = ''
                    ylabel = 'Y PSD'
                elif i == 2:
                    xlabel = 'Frequency [Hz]'
                    ylabel = 'Z PSD'

                fit_obj = self.thermal_cal[i]
                fit_obj.plt_fit(self.psd_freqs, self.psds[i], axarr[i], \
                                scale = "log", xlabel=xlabel, ylabel=ylabel) 
            #f2, axarr2 = plt.subplots(3, sharex = True)
            #print "plotting residuals"
            #for i in [0,1,2]:
            #    if i == 0:
            #        f2.suptitle('Residuals')
            #        xlabel = ''
            #        ylabel = 'X PSD [V^2/rt(Hz)]'
            #    elif i == 1:
            #        xlabel = ''
            #        ylabel = 'Y PSD'
            #    elif i == 2:
            #        xlabel = 'Frequency [Hz]'
            #        ylabel = 'Z PSD'
            #
            #    fit_obj = self.thermal_cal[i]
            #    fit_obj.plt_residuals(self.psd_freqs, self.psds[i], axarr2[i], \
            #                    scale = "log", xlabel=xlabel, ylabel=ylabel) 
            #fit_obj.plt_residuals(self.psd_freqs, self.psds[coordinate], axarr[1])

            plt.show()

    
    def find_H(self, dpsd_thresh = 6e-2, mfreq = 1.):
        #Finds the phase lag between the electrode drive and the respose at a given frequency.
        #check to see if fft has been computed. Comput if not
        if type(self.data_fft) == str:
            self.get_fft()        
        
        edat = np.roll(self.electrode_data,0,axis=-1)

        dfft = np.fft.rfft(edat) #fft of electrode drive in daxis. 
        
        N = np.shape(self.pos_data)[1]#number of samples
        dpsd = np.abs(dfft)**2*2./(N*self.Fsamp) #psd for all electrode drives
        
        inds = np.where(dpsd>dpsd_thresh)#Where the dpsd is over the threshold for being used.
        Hmatst = np.einsum('ij, kj->ikj', self.data_fft, 1./dfft) #transfer matrix between electrodes and bead motion for all frequencies
        finds = inds[1] #frequency index with significant drive
        cinds = inds[0] #colun index with significant drive

        b = finds>np.argmin(np.abs(self.fft_freqs - mfreq))

        # Find and correct for arbitrary pi phase shift if each channel's self response
        # This is equivalent to retaking transfer function data, adding appropriate
        # minus signs in the elctrode to keep the response in phase with the drive
        init_phases = np.mean(np.angle(Hmatst[:,:,finds[b]])[:,:,:2],axis=-1)
        #print init_phases
        for drive in [0,1,2]:
            j = emap2(drive)
            if np.abs(init_phases[drive,j]) > 2.0:
                #print "found out of phase", drive, np.abs(init_phases[drive,j])
                #raw_input()
                Hmatst[:,j,:] = Hmatst[:,j,:] * (-1)
            #for resp in [0,1,2]:
        data_psd = np.abs(self.data_fft)**2*2./(N*self.Fsamp)

        dat_ind = emap(cinds[b][0])

        ######################
        #### Sanity Check ####
        ######################

        #plt.loglog(self.fft_freqs, dpsd[cinds[b][0]])
        #plt.loglog(self.fft_freqs, data_psd[dat_ind])
        #plt.show()

        #print np.sqrt(2.*np.sum(dpsd[cinds[b][0],finds[b]]) / (N / self.Fsamp))
        #plt.plot(self.electrode_data[3])
        #plt.show()

        ######################

        # roll the response fft to compute a noise H
        shift = int(0.5 * (finds[b][1]-finds[b][0])) 
        randadd = np.random.choice(np.arange(-int(0.1*shift), int(0.1*shift)+1, 1))
        shift = shift+randadd

        rolled_data_fft = np.roll(self.data_fft, shift, axis=-1)

        #print finds[b]
        #print shift_for_noise
        #raw_input()

        Hmatst_noise = np.einsum('ij, kj->ikj', rolled_data_fft, 1./dfft)

        self.H = Hmat(finds[b], cinds[b], Hmatst[:, :, finds[b]])
        self.noiseH = Hmat(finds[b], cinds[b], Hmatst_noise[:, :, finds[b]])

        self.Hcomponents = (self.data_fft[:,b], dfft[:,b])



    def find_step_cal_response(self, drive_freq = 41., dpsd_thresh = 2e-2, \
                               mfreq = 1., cut_samp = 2000, pcol = 1, ecol = 3, \
                               band_width = 1.):
        #Finds the phase lag between the electrode drive and the respose at a given frequency.
        #check to see if fft has been computed. Compute if not
        if type(self.data_fft) == str:
            self.get_fft()        
        
        drive = self.electrode_data[ecol]
        response = self.pos_data[pcol]
        
        N = len(self.pos_data[0])
        dt = 1. / self.Fsamp
        t = np.linspace(0,(N+cut_samp-1)*dt, N+cut_samp)
        t = t[cut_samp:]

        b, a = sig.butter(3, [2.*(drive_freq-band_width/2.)/self.Fsamp, 2.*(drive_freq+band_width/2.)/self.Fsamp ], btype = 'bandpass')
        responsefilt = sig.filtfilt(b, a, response)

        ### CORR_FUNC TESTING ###
        #test = 3.14159 * np.sin(2 * np.pi * drive_freq * t)
        #test_corr = bu.corr_func(7 * drive, test, self.Fsamp, drive_freq)
        #print np.sqrt(2) * np.std(test)
        #print np.max(test_corr)
        #########################

        corr_full = bu.corr_func(drive, response, self.Fsamp, drive_freq)

        response_amp2 = np.max(corr_full)
        #response_amp2 = corr_full[0]

        drive_amp = np.sqrt(2) * np.std(drive)
        response_amp = np.sqrt(2) * np.std(responsefilt)

        sign = 1 #np.sign(np.mean(drive*responsefilt))

        ideal_response = sign * response_amp * np.sin(2 * np.pi * drive_freq * t)
        ideal_response2 = sign * response_amp2 * np.sin(2 * np.pi * drive_freq * t)
        ideal_drive = drive_amp * np.sin(2 * np.pi * drive_freq * t) + drive_amp

        #plt.plot(t, drive)
        #plt.plot(t, ideal_drive)
        #plt.show()

        #plt.subplot(2,1,1)
        #plt.plot(response)
        #plt.subplot(2,1,2)
        #plt.plot(ideal_response)
        #plt.plot(ideal_response2)
        #plt.plot(responsefilt)
        #plt.show()

        self.step_cal_response = sign * response_amp2 / drive_amp


    def filter_by_cantdrive(cant_axis=2, nharmonics=1, noise=False):

        cantfft = np.fft.rfft( self.cant_data[cant_axis] )
        fftsq = cantfft.conj() * cantfft
        fundind = np.argmax( fftsq )
        drive_freq = freqs[fundind]
        cantfilt = (fftsq) / (fftsq[fundind])

        for n in range(nharmonics):
            harmind = np.argmin( np.abs(n * drive_freq - freqs))
        


    def diagonalize(self, mat):
        #print "Transfer Matrix"
        #print np.abs(mat)
        diag_ffts = np.einsum('ij, jk -> ik', mat, self.data_fft)
        self.diag_data_fft = diag_ffts
        self.diag_pos_data = np.fft.irfft(diag_ffts)

        N = np.shape(self.pos_data)[1] # number of samples
        norm_fac = 2./(N*self.Fsamp) # psd normalization

        ######################
        #### SANITY CHECK ####
        ######################

        #plt.figure()
        #for i in [0,1,2]:
        #    plt.subplot(3,1,i+1)
        #    plt.loglog(self.fft_freqs, np.abs(self.data_fft[i])*np.sqrt(norm_fac))
        #plt.figure()
        #for i in [0,1,2]:
        #    plt.subplot(3,1,i+1)
        #    plt.loglog(self.fft_freqs, np.abs(diag_ffts[i])*np.sqrt(norm_fac))
        #plt.show()



    def close_dat(self, p = True, psd = True, ft = True, elecs = True):
        #Method to reinitialize the values of some lage attributes to avoid running out of memory.
        if ft:
            self.data_fft = 'fft cleared'
            self.fft_freqs = 'fft freqs cleared'

        if psd:
            self.psds = 'psds cleared'
            self.psd_freqs = 'psd freqs cleared'

        if elecs:
            self.electrode_data = 'electrode data cleared'
        
        if p:
            self.cant_data = 'cantilever position data cleared'
            self.pos_data = 'bead position data cleared'























# Define a class to hold information about a whole directory of files.


class Data_dir:
    #Holds all of the information from a directory of data files.

    def __init__(self, paths, sep, label):
        all_files = []
        if paths:
            for path in paths:
                all_files =  (all_files + get_h5files(path))[:]
        self.label = label
        self.files = sorted(all_files, key = bu.find_str)
        self.sep = sep
        self.fobjs = "Files not loaded"
        self.Hs = "Transfer functions not loaded"
        self.Hs_cal = "Transfer functions not calibrated"
        self.noiseHs = "Noise Transfer functions not loaded"
        self.Havg = "Havg not computed"
        self.Havg_cal = "Calibrated Havg not compute"
        self.Hfuncs = "Transfer function not fit yet"
        self.step_cal_vec = "Single drive TF (for charge cal) not computed"
        self.thermal_cal_file_path = "No thermal calibration file set"
        self.thermal_cal_fobj = "No thermal calibration"
        self.charge_step_calibration = "No charge step calibration"
        self.conv_facs = "Final calibration factors not computed"
        self.avg_force_v_pos = "Average force vs position not computed"
        self.avg_force_v_pos_cantfilt = "Average force vs position not computed"
        self.avg_diag_force_v_pos = "Average diagonalized force vs position not computed"
        self.avg_diag_force_v_pos_cantfilt = "Average diagonalized force vs position not computed"
        self.avg_pos_data = "Average response not computed"
        self.ave_dc_pos = "Mean positions not computed"
        self.avg_pressure = 'pressures not loaded'
        self.sigma_p = 'Pressures not loaded'
        self.drive_amplitude = 'List not Populated'
        self.gravity_signals = 'Gravity force curve not loaded'
        self.paths = paths
        if paths:
            self.out_path = paths[0].replace("/data/","/home/charles/analysis/")
        if len(self.files) == 0:
            print "#########################################"
            print "Warning: empty directory"


    def load_dir(self, loadfun, maxfiles=10000, save_dc=False):
        #Extracts information from the files using the function loadfun which return a Data_file object given a separation and a filename.
        nfiles = len(self.files[:maxfiles])
        print "#########################################"
        print "Entering Directories: ", self.paths
        print "Processing %i files:" % nfiles

        l = lambda fname: loadfun(fname, self.sep)

        #nproc = 4
        #out = pprocess.Map(reuse=1)
        #parallel_function = out.manage(pprocess.MakeReusable(loadfun))
        #[parallel_function(fname, self.sep) for fname in self.files[:nfiles]];
        #self.fobjs = out[:nfiles]

        #sys.stdout.flush()

        self.fobjs = []
        for i in range(nfiles):
            print i,
            sys.stdout.flush()
            self.fobjs.append(loadfun(self.files[i], self.sep))
        print

        #l = lambda fname: loadfun(fname, self.sep)
        #self.fobjs = map(l, self.files[:maxfiles])
        per = lambda fobj: fobj.pressures
        self.avg_pressure = np.mean(map(per, self.fobjs), axis = 0)
        self.sigma_p = np.std(map(per, self.fobjs), axis = 0)

        
        self.ave_dc_pos = np.zeros(3)
        count = 0
        for obj in self.fobjs:
            if type(obj.dc_pos) != str:
                self.ave_dc_pos += obj.dc_pos
                count += 1
        if count:
            self.ave_dc_pos = self.ave_dc_pos / count

    def thermal_calibration(self, temp=293.):
        if 'not computed' in self.thermal_cal_file_path:
            print self.thermal_cal_file_path
        else:
            cal_fobj = Data_file()
            cal_fobj.load(self.thermal_cal_file_path, [0,0,0])
            cal_fobj.detrend()
            cal_fobj.thermal_calibration(temp=temp)
            self.thermal_cal_fobj = cal_fobj





    def get_avg_force_v_pos(self, cant_axis = 2, bin_size = 0.5, \
                                 cant_indx = 24, bias = False, \
                                 baratron_indx = 2, pressures = False):

        if type(self.fobjs) == str:
            self.load_dir(pos_loader)
        if bias:
            def extractor(fobj):
                cant_dat = fobj.binned_cant_data
                pos_dat = fobj.binned_pos_data
                cantV = fobj.electrode_settings[cant_indx]
                return [cant_dat, pos_dat, cantV]
            
        elif pressures:
            def extractor(fobj):
                #extracts [cant data, pos data, baratron pressure]
                cant_dat = fobj.binned_cant_data
                pos_dat = fobj.binned_pos_data
                pressure = round_sig(fobj.pressures[baratron_indx],1)
                if pressure < 5e-5:
                    pressure = 'Base ~ 1e-6'
                else:
                    pressure = '%.1e' % pressure
                return [cant_dat, pos_dat, pressure]
        else:
            def extractor(fobj):
                cant_dat = fobj.binned_cant_data
                pos_dat = fobj.binned_pos_data
                return [cant_dat, pos_dat, 1]
        
        extracted = np.array(map(extractor, self.fobjs))
        
        self.avg_force_v_pos = {}
        for v in np.unique(extracted[:, 2]):
            new_arr = [[[], [], []], \
		       [[], [], []], \
		       [[], [], []]]
            
            for axis in [0,1,2]:
                for vel_mult in [-1,0,1]:
                    boolv = extracted[:, 2] == v

                    cant_dat_curr = []
                    for fil in extracted[boolv,0]:
                        cant_dat_curr.append(fil[vel_mult][axis,cant_axis])
                    cant_dat_curr = np.concatenate(cant_dat_curr, axis=0)

                    pos_dat_curr = []
                    for fil in extracted[boolv,1]:
                        pos_dat_curr.append(fil[vel_mult][axis,cant_axis])
                    pos_dat_curr = np.concatenate(pos_dat_curr, axis=0)

                    xout, yout, yerrs = sbin_pn(cant_dat_curr, pos_dat_curr, bin_size=bin_size)

                    new_arr[axis][vel_mult] = [xout, yout, yerrs]

            self.avg_force_v_pos[str(v)] =  np.array(new_arr)



    def get_avg_diag_force_v_pos(self, cant_axis = 2, bin_size = 0.5, \
                                 cant_indx = 24, bias = False, \
                                 baratron_indx = 2, pressures = False):

        if type(self.fobjs) == str:
            self.load_dir(pos_loader)
        if bias:
            def extractor(fobj):
                cant_dat = fobj.diag_binned_cant_data
                pos_dat = fobj.diag_binned_pos_data
                cantV = fobj.electrode_settings[cant_indx]
                return [cant_dat, pos_dat, cantV]
            
        elif pressures:
            def extractor(fobj):
                #extracts [cant data, pos data, baratron pressure]
                cant_dat = fobj.diag_binned_cant_data
                pos_dat = fobj.diag_binned_pos_data
                pressure = round_sig(fobj.pressures[baratron_indx],1)
                if pressure < 5e-5:
                    pressure = 'Base ~ 1e-6'
                else:
                    pressure = '%.1e' % pressure
                return [cant_dat, pos_dat, pressure]
        else:
            def extractor(fobj):
                cant_dat = fobj.diag_binned_cant_data
                pos_dat = fobj.diag_binned_pos_data
                return [cant_dat, pos_dat, 1]
        
        extracted = np.array(map(extractor, self.fobjs))
        
        self.avg_diag_force_v_pos = {}
        for v in np.unique(extracted[:, 2]):
            new_arr = [[[], [], []], \
		       [[], [], []], \
		       [[], [], []]]
            
            for axis in [0,1,2]:
                for vel_mult in [-1,0,1]:
                    boolv = extracted[:, 2] == v

                    cant_dat_curr = []
                    for fil in extracted[boolv,0]:
                        cant_dat_curr.append(fil[vel_mult][axis,cant_axis])
                    cant_dat_curr = np.concatenate(cant_dat_curr, axis=0)

                    pos_dat_curr = []
                    for fil in extracted[boolv,1]:
                        pos_dat_curr.append(fil[vel_mult][axis,cant_axis])
                    pos_dat_curr = np.concatenate(pos_dat_curr, axis=0)

                    xout, yout, yerrs = sbin_pn(cant_dat_curr, pos_dat_curr, bin_size=bin_size)
                    new_arr[axis][vel_mult] = [xout, yout, yerrs]


            self.avg_diag_force_v_pos[str(v)] =  np.array(new_arr)



    def get_avg_pos_data(self):
        if type(self.fobjs) == str:
            self.load_dir(ft_loader)

        avg = self.fobjs[0].pos_data
        counts = 1.
        for obj in self.fobjs[1:]:
            for i in range(len(avg)):
                avg[i] += obj.pos_data[i]
            counts += 1.
        for i in range(len(avg)):
            avg[i] = avg[i] / counts
        self.avg_pos_data = avg


    def diagonalize_avg_pos(self):
        if type(self.Havg) == str:
            self.build_Havg()
        if type(self.ave_pos_data) == str:
            self.avg_pos_data()

        ft = np.fft.rfft(self.ave_pos_data)

        # invert from response/drive -> drive/response
        H = np.linalg.inv(self.Havg_cal) 
        
        # Diagonalize and calibrate the response
        ft_diag = np.einsum('ij, jk -> ik', H, ft)

        return np.fft.irfft(ft_diag)


    def build_step_cal_vec(self, drive_freq = 41., pcol = 0, ecol = 3, files=[0,1000]):
        # Generates an array of step_cal values for the whole directory.
        # First check to make sure files are loaded and H is computed.
        if type(self.fobjs) == str: 
            self.load_dir(simple_loader)
        
        for fobj in self.fobjs:
            fobj.find_step_cal_response(drive_freq = drive_freq, \
                                        pcol = pcol, ecol = ecol)

        i = 0
        vec = []
        for fobj in self.fobjs:
            if i < files[0] or i > files[1]:
                i += 1
                continue
            vec.append(fobj.step_cal_response)
            if len(vec) >= 2:
                if np.abs(vec[-1]) > 10. * np.abs(vec[-2]):
                    vec[-1] = vec[-2]
            i += 1

        self.step_cal_vec = vec

    
    def build_uncalibrated_H(self, average_first=False, dpsd_thresh = 8e-2, mfreq = 1., \
                             fix_HF=False):
        # Loop over file objects and construct a dictionary with frequencies 
        # as keys and 3x3 transfer matrices as values

        print "BUILDING H..."
        sys.stdout.flush()

        if type(self.fobjs) == str:
            self.load_dir(H_loader)

        Hout = {}
        Hout_noise = {}

        Hout_counts = {}

        if average_first:
            avg_drive_fft = {}
            avg_data_fft = {}
            counts = {}

            for fobj in self.fobjs:

                if type(fobj.data_fft) == str:
                    fobj.get_fft()        

                dfft = np.fft.rfft(fobj.electrode_data) #fft of electrode drive in daxis. 
                data_fft = fobj.data_fft

                N = np.shape(fobj.pos_data)[1]#number of samples
                Fsamp = fobj.Fsamp

                dpsd = np.abs(dfft)**2 * 2./(N*fobj.Fsamp) #psd for all electrode drives   
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
                dpsd = np.abs(avg_drive_fft[eind])**2 * 2. / (N*Fsamp)
                inds = np.where(dpsd > dpsd_thresh)

                # Extract the frequency indices
                finds = inds[1]
                
                # Ignore DC and super low frequencies
                mfreq = 1.0
                b = finds > np.argmin(np.abs(self.fobjs[0].fft_freqs - mfreq))

                freqs = self.fobjs[0].fft_freqs[finds[b]]

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
                outind = emap(eind)
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

                if fix_HF:
                    keys = Hout.keys()
                    keys.sort()

                    freqsep = keys[1] - keys[0]
                    freqsep = freqsep * 0.9

                    curr_sum = np.zeros((3,3), dtype=np.complex128)
                    curr_freqs = []
                    count = 0
                    fixing = False

                    for i, key in enumerate(keys):

                        if key == keys[0]:
                            continue

                        if ((keys[i] - keys[i-1]) < freqsep) and not fixing:
                            fixing = True
                            curr_freqs.append(keys[i-1])
                            curr_sum += Hout[keys[i-1]]
                            count += 1

                        if fixing:
                            curr_freqs.append(keys[i])
                            curr_sum += Hout[keys[i]]
                            count += 1
                            if i == len(keys) - 1:
                                continue
                            else:
                                if keys[i+1] - keys[i] >= freqsep:
                                    fixing = False

                                    for freq in curr_freqs:
                                        del Hout[freq]

                                    newfreq = np.mean(curr_freqs)
                                    Hout[newfreq] = curr_sum
                                    curr_freqs = []
                                    curr_sum = np.zeros((3,3), dtype=np.complex128)
                                


        if not average_first:
            for obj in self.fobjs:
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


        self.Hs = Hout
        self.noiseHs = Hout_noise


    def calibrate_H(self, step_cal_drive_channel = 0, drive_freq = 41.,\
                    plate_sep = 0.004, bins_to_avg = 2):
        if type(self.charge_step_calibration) == str:
            print self.charge_step_calibration
            return
        if type(self.Hs) == str:
            self.build_uncalibrated_H()
        print "CALIBRATING H FROM SINGLE-CHARGE STEP..."
        sys.stdout.flush()
        freqs = np.array(self.Hs.keys())
        freqs.sort()
        ind = np.argmin(np.abs(freqs-drive_freq))
        
        j = step_cal_drive_channel
        bins = bins_to_avg

        # Compute Vresponse / Vdrive on q = q0:
        npfreqs = np.array(freqs)
        freqs_to_avg = npfreqs[:ind]

        resps = []
        for freq in freqs_to_avg:
            resps.append(np.abs(self.Hs[freq][j,j]))

        qfac = np.mean(resps)  
        qfac = qfac * plate_sep # convert V -> E-field -> Force
            
        fac = self.charge_step_calibration.popt[0]  # Vresponse / Ndrive on q=1

        q = qfac / fac
        outstr = "Charge-step calibration implies "+\
                 "%0.2f charge during H measurement" % (q / bu.e_charge)
        print outstr

        Hs_cal = {}
        for freq in freqs:
            # Normalize transfer functions by charge number
            # and convert to force with capacitor plate separation
            # F = q*E = q*(V/d) so we take
            # (Vresp / Vdrive) * d / q = Vresp / Fdrive
            Hs_cal[freq] = np.copy(self.Hs[freq]) * (plate_sep / q)

        self.Hs_cal = Hs_cal




    def filter_files_by_cantdrive(self, cant_axis=2, nharmonics=1, noise=True):
        # Apply to each file a filter constructed from the FFT of the 
        # cantilever drive

        print "Filtering files by cantilever drive"
        sys.stdout.flush()

        for fobj in self.fobjs:
            fobj.filter_by_cantdrive(cant_axis=cant_axis, nharmonics=nharmonics, \
                                     noise=noise)






    def diagonalize_files(self, fthresh = 40., simpleDCmat=False, plot_Happ=False, \
                          reconstruct_lowf=False, lowf_thresh=100., \
                          build_conv_facs=False, drive_freq=41., close_dat=True):
        if type(self.Hs_cal) == str:
            try:
                self.calibrate_H()
            except:
                print self.Hs_cal

        print "Diagonalizing Data:"
        sys.stdout.flush()
        
        if simpleDCmat:
            # Use the average of the low frequency response to 
            # diagonalize the data
            self.build_Havg(fthresh = fthresh)
            mat = np.linalg.inv(self.Havg_cal)

            for fobj in self.fobjs:
                fobj.diagonalize(mat)
                fobj.spatial_bin(diag=True)
                fobj.close_dat(ft=False, elecs=False)
            
            return
        
        # If not using the low-frequency average, compute the full
        # TF array, sampled at the fft_frequencies of the data
        print "  Building TF array...",
        sys.stdout.flush()
        freqs = self.fobjs[0].fft_freqs
        Nfreqs = len(freqs)

        Harr = np.zeros((Nfreqs,3,3),dtype=np.complex128)

        for drive in [0,1,2]:
            for resp in [0,1,2]:
                #print ("(%i, %i)" % (drive,resp)),
                sys.stdout.flush()
                magparams = self.Hfuncs[resp][drive][0]
                phaseparams = self.Hfuncs[resp][drive][1]
                phase0 = self.Hfuncs[resp][drive][2]
                    
                mag = damped_osc_amp(freqs, magparams[0], magparams[1], magparams[2])
                phase = damped_osc_phase(freqs, phaseparams[0], phaseparams[1], \
                                             phaseparams[2], phase0=phase0)
                Harr[:,drive,resp] = mag * np.exp(1.0j*phase)

        Harr = np.linalg.inv(Harr)

        if build_conv_facs:
            convind = np.argmin(np.abs(freqs-drive_freq)) 

            convmat = Harr[convind,:,:]
            self.conv_facs = np.abs(np.array([convmat[0,0], convmat[1,1], convmat[2,2]]))
            #print self.conv_facs, type(self.conv_facs)

        if reconstruct_lowf:
            ind = np.argmin(np.abs(freqs - lowf_thresh))
            Harr[ind:,:,:] = 0.+0.j

        if plot_Happ:
            f1, axarr1 = plt.subplots(3,3,sharex='all',sharey='all')
            f2, axarr2 = plt.subplots(3,3,sharex='all',sharey='all')
            for resp in [0,1,2]:
                for drive in [0,1,2]:
                    TF = Harr[:,resp,drive]
                    mag = np.abs(TF)
                    phase = np.angle(TF)
                    unphase = np.unwrap(phase, discont=1.4*np.pi)

                    #if type(self.conv_facs) != str:
                    #    conv_vec = np.zeros(len(freqs)) + self.conv_facs[resp]
                    #    axarr1[resp,drive].loglog(freqs, conv_vec)
                    axarr1[resp,drive].loglog(freqs, mag)
                    axarr1[resp,drive].grid()
                    axarr2[resp,drive].semilogx(freqs, unphase)
                    axarr2[resp,drive].grid()
            for drive in [0,1,2]:
                axarr1[0,drive].set_title("Drive in direction \'%i\'"%drive)
                axarr2[0,drive].set_title("Drive in direction \'%i\'"%drive)
                axarr1[2,drive].set_xlabel('Frequency [Hz]')
                axarr2[2,drive].set_xlabel('Frequency [Hz]')
            for resp in [0,1,2]:
                axarr1[resp,0].set_ylabel('Mag [Newton/Volt]')
                axarr2[resp,0].set_ylabel('Phase [rad]')
            axarr1[0,0].set_ylim(1e-17,1e-10)
            f1.suptitle("Magnitude of Transfer Function", fontsize=18)
            f2.suptitle("Phase of Transfer Function", fontsize=18)
            plt.show()

        print
        print "  Applying TF to files...",
        sys.stdout.flush()
        Nfobjs = len(self.fobjs)
        percent = 0
        for fobjind, fobj in enumerate(self.fobjs):
            if (100. * float(fobjind) / float(Nfobjs)) > percent:
                print percent,
                sys.stdout.flush()
                percent += 10
            assert np.array_equal(fobj.fft_freqs, freqs)

            diag_fft = np.einsum('ikj,ki->ji', Harr, fobj.data_fft)
            fobj.diag_pos_data = np.fft.irfft(diag_fft)
            fobj.diag_data_fft = diag_fft
            
            #for ind in [0,1,2]:
            #    plt.subplot(3,1,ind+1)
            #    plt.loglog(freqs, np.abs(fobj.data_fft[ind]))

            #plt.figure()
            #for ind in [0,1,2]:
            #    plt.subplot(3,1,ind+1)
            #    plt.loglog(freqs, np.abs(diag_fft[ind]))
            #plt.show()

            fobj.spatial_bin(diag=True)
            if close_dat:
                fobj.close_dat(ft=False, elecs=False)
        # All previous print statements have had commas so print a newline
        print


    def save_H(self, fname, cal=False):
        out = (self.Hs, self.noiseHs, self.Hs_cal, self.Hfuncs)
        pickle.dump(out, open(fname, "wb"))

    def load_H(self, fname):
        newH = pickle.load( open(fname, "rb"))
        self.Hs = newH[0]
        self.noiseHs = newH[1]
        self.Hs_cal = newH[2]
        self.Hfuncs = newH[3]


    def build_Havg(self, fthresh = 80):
        # average over frequencies f < 0.5*f_natural
        if type(self.Hs) == str:
            self.build_uncalibrated_H()
            self.calibrate_H()

        keys = self.Hs.keys()

        mats = []
        mats_cal = []
        for key in keys:
            if key < fthresh:
                mats.append(self.Hs[key])
                mats_cal.append(self.Hs_cal[key])

        mats = np.array(mats)
        mats_cal = np.array(mats_cal)

        self.Havg =  np.mean(mats, axis=0)
        self.Havg_cal = np.mean(mats_cal, axis=0)

        

    def build_Hfuncs(self, fit_freqs = [50.,600], fpeaks=[400.,400.,50.], \
                     weight_peak=False, weight_lowf=False, lowf_thresh=60., \
                     weight_phase=False, plot_fits=False, plot_inits=False, \
                     grid = False, fit_osc_sum=False):
        # Build the calibrated transfer function array
        # i.e. transfer matrices at each frequency and fit functions to each component

        if type(self.Hs_cal) == str:
            self.calibrate_H()
            
        keys = self.Hs_cal.keys()
        keys.sort()

        keys = np.array(keys)

        mats = []
        for freq in keys:
            mat = self.Hs_cal[freq]
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
                
                # Make initial guess based on high-pressure thermal spectra fits
                therm_fits = self.thermal_cal_fobj.thermal_cal
                if (drive == 2) or (resp == 2):
                    # Z-direction is considerably different than X or Y
                    Amp = 1e17
                    f0 = therm_fits[2].popt[1]
                    g = therm_fits[2].popt[2]
                    fit_freqs = [1.,200.]
                    fpeak = fpeaks[2]
                else:
                    Amp = 1e19
                    f0 = therm_fits[resp].popt[1]
                    g = therm_fits[resp].popt[2]
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
                if weight_peak:
                    weights = weights - 0.7 * np.exp(-(npkeys-fpeak)**2 / (2 * 600) )
                if weight_lowf:
                    ind = np.argmin(np.abs(npkeys - lowf_thresh))
                    weights[:ind] *= 0.2
                phase_weights = np.zeros(len(npkeys)) + 1.
                if weight_phase and (drive != 2 and resp != 2):
                    ind = np.argmin(np.abs(npkeys - 50.))
                    phase_weights[:ind] *= 0.05

                # Fit the TF magnitude
                try:
                    popt_mag, pcov_mag = curve_fit(damped_osc_amp, keys[b], mag[b], \
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
                        popt, pcov = curve_fit(fitfun, keys[b], unphase[b], \
                                               p0=p0_phase, bounds=bounds,
                                               sigma=phase_weights[b])
                    except:
                        #print "bad fit...", drive, resp, np.pi*pmult
                        popt = p0_phase
                    
                    # Save the fits and the residuals
                    phase_fits[pmult] = np.copy(popt)
                    phase_resids[pmult] = np.sum( np.abs(damped_osc_phase(keys[b], popt[0], \
                                                                popt[1], popt[2], phase0=np.pi*pmult) \
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

                    popt_mag, pcov_mag = curve_fit(fit_amp_func, keys[b], mag[b], \
                                                       p0=p0_mag, maxfev=1000000)

                    popt_phase, pcov_phase = curve_fit(fit_phase_func, keys[b], unphase[b], \
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





        self.Hfuncs = fits

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


    def plot_H(self, f1, axarr1, f2, axarr2, \
               phase=False, show=False, label=False, noise=False,\
               show_zDC=False, cal=False, lim=False, inv=False):
        # plot all the transfer functions

        if type(self.Hs) == str:
            print "need to build H's first..."
            self.build_uncalibrated_H()
            
        if noise:
            keys = self.noiseHs.keys()
        elif cal and not noise:
            keys = self.Hs_cal.keys()
        else:
            keys = self.Hs.keys()
        keys.sort()

        mats = []
        for freq in keys:
            if noise:
                mat = self.noiseHs[freq]
            elif cal and not noise:
                mat = self.Hs_cal[freq]
            else:
                mat = self.Hs[freq]

            if inv:
                mat = np.linalg.inv(mat)
            mats.append(mat)
        # Plot the magnitude of the transfer function:
        #     Makes separate plots for a given direction of drive
        #     each with three subplots detailing x, y, and z response
        #     to a drive in a particular direction
        mats = np.array(mats)

        #f1, axarr1 = plt.subplots(3,3, sharex='col', sharey='row')

        f1.suptitle("Magnitude of Transfer Function")
        for drive in [0,1,2]:
            for response in [0,1,2]:
                mag = np.abs(mats[:,response,drive])

                # check for NaNs from empty directory or incomplete
                # measurements and replace with nearest neighbor value
                nans = np.isnan(mag)
                for nanind, boolv in enumerate(nans):
                    if boolv:
                        mag[nanind] = mag[nanind-1]
                #mag[nans] = np.zeros(len(mag[nans])) + mag[nans-1]
                if label and response == 0:
                    try:
                        axarr1[response, drive].loglog(keys, mag, label = self.label)
                    except:
                        print "some zeros are bad"
                elif show_zDC and response == 2:
                    try:
                        axarr1[response, drive].loglog(keys, mag, \
                                                       label="Avg Z: %0.4f"%self.ave_dc_pos[-1])
                    except:
                        print "some zeros are bad"
                else:
                    try:
                        axarr1[response, drive].loglog(keys, mag)
                    except:
                        print "some zeros are bad"

                if show:
                    axarr1[response,drive].legend(loc=0)
            axarr1[2, drive].set_xlabel("Frequency [Hz]")
        
        for drive in [0,1,2]:
            axarr1[0, drive].set_title("Drive in direction \'%i\'"%drive)
        for response in [0,1,2]:
            axarr1[response, 0].set_ylabel("Response in \'%i\' [V/V]" %response)


        # Plot the phase of the transfer function:
        #     Same plot/subplot breakdown as before
        
        f2.suptitle("Phase of Transfer Function")
        if phase and not noise:
            for drive in [0,1,2]:
                #plt.figure(drive+4)
                for response in [0,1,2]:
                    #ax2 = plt.subplot(3,1,response+1)
                    phase = np.angle(mats[:,response,drive])

                    # Check for NaNs in phase and replace with ~0 
                    # (the semilogx doesn't like when one vector is 
                    # identically 0 so I add 1e-12)
                    nans = np.isnan(phase)
                    phase[nans] = np.zeros(len(phase[nans])) + 1e-12
                    if np.mean(phase) == 0 and np.std(phase) == 0:
                        phase = phase + 1e-12
                    unphase = np.unwrap(phase, discont=1.4*np.pi)
                    if unphase[0] < -2.5:
                        unphase = unphase + 2 * np.pi

                    if label and response == 0:
                        axarr2[response, drive].semilogx(keys, unphase, label = self.label)
                    elif show_zDC and response == 2:
                        axarr2[response, drive].semilogx(keys, unphase, \
                                   label="Avg Z: %0.4f"%self.ave_dc_pos[-1])
                    else:
                        axarr2[response, drive].semilogx(keys, unphase)

                    if lim:
                        axarr2[response, drive].ylim(-1.5*np.pi, 1.5*np.pi)

                axarr2[response, drive].set_xlabel("Frequency [Hz]")

            for drive in [0,1,2]:
                axarr2[0, drive].set_title("Drive in direction \'%i\'"%drive)
            
            for response in [0,1,2]:
                axarr2[response, 0].set_ylabel("Response in \'%i\' [rad]" %response)


                

        
    def step_cal(self, n_phi = 20, plate_sep = 0.004, \
                 drive_freq = 41., amp_gain = 1.):
        # Produce a conversion between voltage and force given a directory with single electron steps.
        # Check to see that Hs have been calculated.
        if type(self.step_cal_vec) == str:
            self.build_step_cal_vec(drive_freq = drive_freq)
        
        #phi = np.mean(np.angle(dir_obj.step_cal_vec[:n_phi])) #measure the phase angle from the first n_phi samples.
        #yfit =  np.abs(dir_obj.step_cal_vec)*np.cos(np.angle(dir_obj.step_cal_vec) - phi)

        yfit = np.abs(self.step_cal_vec)
        bvec = [yfit<10.*np.mean(yfit)] #exclude cray outliers
        yfit = yfit[bvec]

        plt.figure(1)
        plt.ion()
        plt.plot(yfit, 'o')
        plt.show()

        print "CHARGE STEP CALIBRATION"
        print "Enter guess at number of steps and charge at steps [[q1, q2, q3, ...], [x1, x2, x3, ...], vpq]"
        nstep = input(": ")
        
        #function for fit with volts per charge as only arg.
        def ffun(x, vpq, offset):
            qqs = vpq*np.array(nstep[0])
            try:
                offarr = np.zeros(len(x))
                offarr[x>nstep[-1]] += offset
            except TypeError:
                if x > nstep[-1]:
                    offarr = offset
                else:
                    offarr = 0
            return multi_step_fun(x, qqs, nstep[1]) + offarr

        xfit = np.arange(len(self.step_cal_vec))
        xfit = xfit[bvec]

        #fit
        p0 = [nstep[2],0.02]#Initial guess for the fit
        popt, pcov = curve_fit(ffun, xfit, yfit, p0 = p0, xtol = 1e-12)

        fitobj = Fit(popt, pcov, ffun)#Store fit in object.

        plt.close(1)
        plt.ioff()
        f, axarr = plt.subplots(2, sharex = True)#Plot fit
        fitobj.plt_fit(xfit, yfit, axarr[0])
        fitobj.plt_residuals(xfit, yfit, axarr[1])
        plt.show()
        
        #Determine force calibration.
        fitobj.popt = fitobj.popt * 1./(amp_gain*bu.e_charge/plate_sep)
        fitobj.errs = fitobj.errs * 1./(amp_gain*bu.e_charge/plate_sep)
        self.charge_step_calibration = fitobj

    def save_step_cal(self, fname):
        fitobj = self.charge_step_calibration
        step_cal_out = [fitobj.popt, fitobj.errs]
        pickle.dump(step_cal_out, open(fname, "wb"))

    def load_step_cal(self, fname):
        def ffun(x, vpq):
            qqs = vpq*np.array(nstep[0])
            return multi_step_fun(x, qqs, nstep[1]) + offset

        step_cal_in = pickle.load( open(fname, "rb"))
        new_fitobj = Fit(step_cal_in[0], step_cal_in[1], ffun)
        self.charge_step_calibration = new_fitobj


    def get_conv_facs(self, step_cal_drive_channel = 1, drive_freq = 41.):
        # Use transfer function to get charge step calibration in all channels
        # from the channel in which is was performed (nominally Y)
        fac = 1. / self.charge_step_calibration.popt[0]   # Gives Newton / Volt

        try:
            freqs = np.array(self.Hs.keys())
            ind = np.argmin(np.abs(drive_freq - freqs))
            freq = freqs[ind]
            mat = self.Hs[freq]

            facs = []
            j = step_cal_drive_channel

            for i in [0,1,2]:
                newfac = fac * np.abs(mat[j,j]) / np.abs(mat[i,i])
                facs.append(newfac)
            
            self.conv_facs = np.array(facs)

        except:
            self.conv_facs = [fac, fac, fac]
        

    def generate_alpha_lambda_limit(self, rbead=2.5e-06, sep=10.0e-06, offset=0., \
                                    least_squares=True, opt_filt=False, \
                                    resp_axis=1, cant_axis=1, rebin=False, bin_size=5., \
                                    diag=False, scale=1.0e18):
        if type(self.gravity_signals) == str:
            print self.gravity_signals
            return
        
        fcurves = self.gravity_signals[rbead][sep]
        simposdat = self.gravity_signals['posvec'] * 1e6
        lambdas = fcurves.keys()
        lambdas.sort()

        fac = self.conv_facs[resp_axis]

        if least_squares:
            if (type(self.avg_force_v_pos) == str) or rebin:
                self.get_avg_force_v_pos(cant_axis=cant_axis, bin_size=bin_size)
                self.get_avg_diag_force_v_pos(cant_axis=cant_axis, bin_size=bin_size)

            if diag:
                keys = self.avg_diag_force_v_pos.keys()
                if len(keys) > 1:
                    print "STUPIDITYError: Multiple Keys"
                key = keys[0]
                dat = self.avg_diag_force_v_pos[key][resp_axis,0]
            else:
                keys = self.avg_force_v_pos.keys()
                if len(keys) > 1:
                    print "STUPIDITYError: Multiple Keys"
                key = keys[0]
                dat = self.avg_force_v_pos[key][resp_axis,0]

            posdat = dat[0] + offset
            forcedat = dat[1]
            errs = dat[2]

            if not diag:
                forcedat = forcedat * fac
                errs = errs * fac

            alphas = []
            print "Fitting different alpha values..."
            sys.stdout.flush()

            poffsets = []
            for yukind, yuklambda in enumerate(lambdas):
                #per = int(100. * float(yukind) / float(len(lambdas)))
                #if not per % 1:
                #    print str(per) + ',',
                #sys.stdout.flush()

                fcurve = fcurves[yuklambda]

                fmax = np.max(forcedat) - np.min(forcedat)
                yukmax = np.max(fcurve[1]) - np.min(fcurve[1])
                alphaguess = fmax / yukmax

                Gforcefunc = interpolate.interp1d(simposdat, fcurve[0])
                yukforcefunc = interpolate.interp1d(simposdat, fcurve[1])

                #plt.figure()
                #plt.plot(pos)
                #plt.plot(posdat)
                #plt.show()
                #raw_input()

                #Gforcefunc = interpolate.interp1d(fcurve[0]*1e6, fcurve[1])
                #yukforcefunc = interpolate.interp1d(fcurve[0]*1e6, fcurve[2])

                def fitfun(x, alpha):
                    return Gforcefunc(x) + alpha * yukforcefunc(x)

                popt, pcov = optimize.curve_fit(fitfun, posdat, forcedat, p0 = alphaguess)

                #plt.plot(posdat, forcedat*1e15, 'o')
                #plt.plot(posdat, fitfun(posdat, popt[0])*1e15)
                #plt.show()

                alpha = popt[0]
                if alpha < 0:
                    alpha *= -1.

                alphas.append(alpha)

            return np.array(lambdas), np.array(alphas)


        if opt_filt:
            if type(fobj.pos_data) == str:
                print fobj.pos_data
                return
            




        
    def save_dir(self):
        #Method to save Data_dir object.
        if(not os.path.isdir(self.out_path) ):
            os.makedirs(self.out_path)
        outfile = os.path.join(self.out_path, "dir_obj.p")       
        pickle.dump(self, open(outfile, "wb"))

    def load_from_file(self):
        #Method to laod Data_dir object from a file.
        fname = os.path.join(self.out_path, "dir_obj.p")       
        temp_obj = pickle.load(open(fname, 'rb'))
        self.fobjs = temp_obj.fobjs
        self.Hs = temp_obj.Hs
        self.thermal_calibration = temp_obj.thermal_calibration
        self.charge_step_calibration = temp_obj.charge_step_calibration
        self.avg_force_v_pos = temp_obj.avg_force_v_pos

