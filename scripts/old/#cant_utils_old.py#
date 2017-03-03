import numpy as np
import matplotlib
import bead_util as bu
import scipy
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import glob
import cPickle as pickle
import copy
import scipy.signal as sig

#Define functions and classes for use processing and fitting data.
def thermal_psd_spec(f, A, f0, g, n, s):
    #The position power spectrum of a microsphere normalized so that A = (volts/meter)^2*2kb*t/M
    w = 2.*np.pi*f #Convert to angular frequency.
    w0 = 2.*np.pi*f0
    num = g
    denom = ((w0**2 - w**2)**2 + w**2*g**2)
    return A*num/denom + n + s*w

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


class Fit:
    #holds the optimal parameters and errors from a fit. Contains methods to plot the fit, the fit data, and the residuals.
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
        #ax.set_xscale(scale)
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
        

    #def plt_chi_sq(self,xdata, ydata, errs, ax):
        #plots chi square contours. 

    


        

def thermal_fit(psd, freqs, fit_freqs = [10., 400.], kelvin = 300., fudge_fact = 1e-6, noise_floor = 0., noise_slope = 0.):
    #Function to fit the thermal spectra of a bead's motion
    #First need good intitial guesses for fit parameters.
    fit_bool = bu.inrange(freqs, fit_freqs[0], fit_freqs[1]) #Boolian vector of frequencies over which the fit is performed
    f0 = freqs[np.argmax(psd[fit_bool])] #guess resonant frequency from hightest part of spectrum
    df = freqs[1] - freqs[0] #Frequency increment.
    vpmsq = bu.bead_mass/(bu.kb*kelvin)*np.sum(psd[fit_bool])*df*len(psd)/np.sum(fit_bool) #Guess at volts per meter using equipartition
    g0 = 1./2.*f0 #Guess at damping assuming critical damping
    A0 = vpmsq*2.*bu.kb*kelvin/(bu.bead_mass*fudge_fact)
    p0 = [A0, f0, g0, noise_floor, noise_slope] #Initial parameter vectors 
    popt, pcov = curve_fit(thermal_psd_spec, freqs[fit_bool], psd[fit_bool], p0 = p0)
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
    print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.close_dat(elecs=False)
    return fobj

def pos_loader(fname, sep):
    #Generate all of the position attibutes of interest for a single file. Returns a Data_file object.
    print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.ms()
    fobj.spatial_bin()
    fobj.close_dat()
    return fobj

def H_loader(fname, sep):
    #Generates transfer func data for a single file. Returns a Data_file object.
    print "Processing: ", fname
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.find_H()
    fobj.close_dat()
    return fobj

def sb_loader(fname, sep = [0,0,0], col = 1, find = 16):
    #loads the spacings between the drive frequency and the sidebands
    fobj = Data_file()
    fobj.load(fname, sep)
    fobj.ms()
    fobj.psd()
    b, a = sig.butter(4, 0.1, btype = 'high')
    psd = sig.filtfilt(b, a, np.ravel(fobj.psds[col]))
    f = fobj.psd_freqs
    df = fobj.electrode_settings[find]
    find = np.argmin((f - df)**2)
    lpsd = psd[:find]
    lf = f[:find]
    hpsd = psd[find:2*find]
    hf = -f[find:2*find] + 2.*df
    #plt.plot(lf, lpsd*hpsd)
    #plt.plot(hf, hpsd)
    #plt.plot(f[:find], rdpsd[:find]*(rdpsd[find:2*find][::-1]))
    #plt.plot(f, psd, 'o', markersize = 3)
    #plt.show()
    return fobj

#define a class with all of the attributes and methods necessary for processing a single data file to 
    

class Hmat:
    #this class holds transfer matricies between electrode drives and bead response.
    def __init__(self, finds, electrodes, Hmats):
        self.finds = finds #Indicies of frequences where there is an electrode being driven above threshold 
        self.electrodes = electrodes #the electrodes where there is statistically significant signal
        self.Hmats = Hmats #Transfer matrix at the frequencies 

    def get_3by3_matrix(self):
        # from the full 3x7 transfer matrix, return appropriate 3x3
        return





class Data_file:
    #This is a class with all of the attributes and methods for a single data file.

    def __init__(self):
        self.fname = "Filename not assigned."
        #self.path = "Directory not assigned." #Assuming directory in filename
        self.pos_data = "bead position data not loaded"
        self.binned_pos_data = "Binned data not computed"
        self.binned_data_errors = "bined data errors not computed"
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
        self.fft_freqs = "fft freqs not computed"
        self.psd_freqs = "psd freqs not computed"
        self.thermal_cal = "Thermal calibration not computed"
        self.H = "bead electrode transfer function not computed"
        self.sb_spacing = "sideband spacing not computed."

    def load(self, fstr, sep, cant_cal = 8., stage_travel = 80., cut_samp = 2000, \
             elec_inds = [8, 9, 10, 12, 13, 14, 15]):
        #Methods to load the attributes from a single data file. sep is a vector of the distances of closes approach for each direction ie. [xsep, ysep, zsep] 
        dat, attribs, f = bu.getdata(fstr)
        
        self.fname = fstr
        
        dat = dat[cut_samp:, :]
        
        #Attributes coming from Labview Front pannel settings
        self.separation = sep #Manually entreed distance of closest approach
        self.Fsamp = attribs["Fsamp"] #Sampling frequency of the data
        self.Time = bu.labview_time_to_datetime(attribs["Time"]) #Time of end of file
        self.temps = attribs["temps"] #Vector of thermocouple temperatures 
        self.pressures = attribs["pressures"] #Vector of chamber pressure readings [pirani, cold cathode]
        self.synth_settings = attribs["synth_settings"] #Synthesizer fron pannel settings
        self.dc_supply_settings = attribs["dc_supply_settings"] #DC power supply front pannel testings.
        
        self.electrode_settings = attribs["electrode_settings"] #Electrode front pannel settings for all files in the directory.fist 8 are ac amps, second 8 are frequencies, 3rd 8 are dc vals 
        self.electrode_dc_vals = attribs["electrode_dc_vals"] #Front pannel settings applied to this particular file. Top boxes independent of the sweeps
        self.stage_settings = attribs['stage_settings'] #Front pannel settings for the stage for this particular file.
        
        #Data vectors and their transforms
        self.pos_data = np.transpose(dat[:, 0:3]) #x, y, z bead position
        self.cant_data = np.transpose(np.resize(sep, np.shape(np.transpose(self.pos_data)))) + stage_travel - np.transpose(dat[:, 17:20])*cant_cal
        self.electrode_data = np.transpose(dat[:, 8:16]) #Record of voltages on the electrodes

        f.close()

    def get_stage_settings(self, axis=2):
        if axis == 0:
            mask = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        elif axis == 1:
            mask = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
        elif axis == 2:
            mask = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

        return self.stage_settings[mask]


    def ms(self):
        #mean subtracts the position data.
        ms = lambda vec: vec - np.mean(vec)
        self.pos_data  = map(ms, self.pos_data)

    def spatial_bin(self, bin_sizes = [1., 1., 4.], cant_axis = 2):
        #Method for spatially binning data based on stage z  position.
        
        self.binned_cant_data = [[[[], [], []], [[], [], []], [[], [], []]], \
                                 [[[], [], []], [[], [], []], [[], [], []]], \
                                 [[[], [], []], [[], [], []], [[], [], []]]] 
        self.binned_pos_data = [[[[], [], []], [[], [], []], [[], [], []]], \
                                [[[], [], []], [[], [], []], [[], [], []]], \
                                [[[], [], []], [[], [], []], [[], [], []]]]
        self.binned_data_errors = [[[[], [], []], [[], [], []], [[], [], []]], \
                                   [[[], [], []], [[], [], []], [[], [], []]], \
                                   [[[], [], []], [[], [], []], [[], [], []]]]

        for i, v in enumerate(self.pos_data):
            for j, pv in enumerate(self.cant_data):
                for si in np.arange(-1, 2, 1):
                    bins, y_binned, y_errors = \
                            sbin_pn(self.cant_data[j], v, bin_sizes[j], vel_mult = si)
                    self.binned_cant_data[si][i][j] = bins
                    self.binned_pos_data[si][i][j] = y_binned 
                    self.binned_data_errors[si][i][j] = y_errors 
        
        self.binned_cant_data = np.array(self.binned_cant_data)
        self.binned_pos_data = np.array(self.binned_pos_data)
        self.binned_data_errors = np.array(self.binned_data_errors)

    def psd(self, NFFT = 2**16):
        #uses matplotlib mlab psd to take a psd of the microsphere position data.
        psder = lambda v: matplotlib.mlab.psd(v, NFFT = NFFT, Fs = self.Fsamp)[0]
        self.psds = np.array(map(psder, self.pos_data))
        self.psd_freqs = np.fft.rfftfreq(NFFT, d = 1./self.Fsamp)

    def get_fft(self):
        #Uses numpy fft rfft to compute the fft of the position data
        self.data_fft = np.fft.rfft(self.pos_data)
        self.fft_freqs = np.fft.rfftfreq(np.shape(self.pos_data)[1])*self.Fsamp


    def thermal_calibration(self):
        #Use thermal calibration calibrate voltage scale into physical units
        #Check to see if psds is computed and compute if not.
        if type(self.psds) == str:
            self.psd()
            
        caler = lambda v: thermal_fit(v, self.psd_freqs) 
        self.thermal_cal = map(caler, self.psds)
    
    def plt_thermal_fit(self, coordinate = 0):
        #plots the thermal calibration and residuals
        if type(self.thermal_cal) == str:
            print "No thermal calibration"
        else:
            f, axarr = plt.subplots(2, sharex = True)
            fit_obj = self.thermal_cal[coordinate]
            fit_obj.plt_fit(self.psd_freqs, self.psds[coordinate], axarr[0]) 
            fit_obj.plt_residuals(self.psd_freqs, self.psds[coordinate], axarr[1])
            plt.show()

    
    def find_H(self, dpsd_thresh = 1e-2, mfreq = 1.):
        #Finds the phase lag between the electrode drive and the respose at a given frequency.
        #check to see if fft has been computed. Comput if not
        if type(self.data_fft) == str:
            self.get_fft()
        
        
        dfft = np.fft.rfft(self.electrode_data) #fft of electrode drive in daxis. 
        
        N = np.shape(self.pos_data)[1]#number of samples
        dpsd = np.abs(dfft)**2*2./(N*self.Fsamp) #psd for all electrode drives
        
        inds = np.where(dpsd>dpsd_thresh)#Where the dpsd is over the threshold for being used.
        Hmatst = np.einsum('ij, kj->ikj', self.data_fft, 1./dfft) #transfer matrix between electrodes and bead motion for all frequencies
        finds = inds[1] #frequency index with significant drive
        cinds = inds[0] #colun index with significant drive
        b = finds>np.argmin(np.abs(self.fft_freqs - mfreq))
        self.H = Hmat(finds[b], cinds[b], Hmatst[:, :, finds[b]])


    def plt_psd(self, col = 1):
        #plots psd
        #b, a = sig.butter(4, 0.01, btype = 'highpass')
        plt.loglog(self.psd_freqs, self.psds[col], label = str(self.electrode_settings[24]))

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



#Define a class to hold information about a whole directory of files.
class Data_dir:
    #Holds all of the information from a directory of data files.

    def __init__(self, path, sep):
        self.files = get_h5files(path)
        self.sep = sep
        self.fobjs = "Files not loaded"
        self.Hs = "Transfer functions not loaded"
        self.thermal_calibration = "No thermal calibration"
        self.charge_step_calibration = "No charge step calibration"
        self.ave_force_vs_pos = "Average force vs position not computed"
        self.ave_pressure = 'pressures not loaded'
        self.path = path
        self.out_path = path.replace("/data/","/home/charles/analysis/")
        if len(self.files) == 0:
            print "Warning: empty directory"


    def load_dir(self, loadfun):
        #Extracts information from the files using the function loadfun which return a Data_file object given a separation and a filename.
        l = lambda fname: loadfun(fname, self.sep)
        self.fobjs = map(l, self.files)
        per = lambda fobj: fobj.pressures
        self.ave_pressure = np.mean(map(per, self.fobjs), axis = 0) 

    def force_v_p(self):
        #Calculates the force vs position for all of the files in the data directory.
        #First check to make sure files are loaded and force vs position is computed.
        if type(self.fobjs) == str:
            self.load_dir(pos_loader)
        
        #self.load_dir(pos_loader)
        
    
    def avg_force_v_p(self, axis = 2, bin_size = 0.5, cant_indx = 24):
        #Averages force vs positon over files with the same potential. Returns a list of average force vs position for each cantilever potential in the directory.
        if type(self.fobjs) == str:
            self.load_dir(pos_loader)
        
        extractor = lambda fobj: [fobj.binned_cant_data[axis, 2], fobj.binned_pos_data[axis,2], fobj.electrode_settings[cant_indx]] #extracts [cant data, pos data, cant voltage]
        
        extracted = np.array(map(extractor, self.fobjs))
        self.ave_force_vs_pos = {}
        for v in np.unique(extracted[:, 2]):
            boolv = extracted[:, 2] == v
            xout, yout, yerrs = sbin(np.hstack(extracted[boolv, 0]), np.hstack(extracted[boolv, 1]), bin_size)
            self.ave_force_vs_pos[str(v)] =  [xout, yout, yerrs]

    def H_vec(self, pcol = 1, ecol = 3):
        #Generates an array of Hs for the whole directory.
        #First check to make sure files are loaded and H is computed.
        if type(self.fobjs) == str: 
            self.load_dir(H_loader)
        
        if type(self.fobjs[0].H) == str:
            self.load_dir(H_loader)
            
        Her = lambda fobj: np.mean(fobj.H.Hmats[pcol, ecol, :], axis = 0)
        self.Hs = map(Her, self.fobjs)
        
    def step_cal(self, dir_obj, n_phi = 140, plate_sep = 0.004, amp_gain = 200.):
        #Produce a conversion between voltage and force given a directory with single electron steps.
        #Check to see that Hs have been calculated.
        if type(dir_obj.Hs) == str:
            dir_obj.H_vec()
        
        phi = np.mean(np.angle(dir_obj.Hs[0:n_phi])) #measure the phase angle from the first n_phi samples.
        yfit =  np.abs(dir_obj.Hs)*np.cos(np.angle(dir_obj.Hs) - phi)
        plt.plot(yfit, 'o')
        plt.show(hold=False)
        nstep = input("Enter guess at number of steps and charge at steps [[q1, q2, q3, ...], [x1, x2, x3, ...], vpq]: ")
        
        #function for fit with volts per charge as only arg.
        def ffun(x, vpq):
            qqs = vpq*np.array(nstep[0])
            return multi_step_fun(x, qqs, nstep[1])

        xfit = np.arange(len(dir_obj.Hs))
        
        #fit
        p0 = nstep[2]#Initial guess for the fit
        popt, pcov = curve_fit(ffun, xfit, yfit, p0 = p0)

        fitobj = Fit(popt, pcov, ffun)#Store fit in object.

        f, axarr = plt.subplots(2, sharex = True)#Plot fit
        fitobj.plt_fit(xfit, yfit, axarr[0])
        fitobj.plt_residuals(xfit, yfit, axarr[1])
        plt.show()
        
        #Determine force calibration.
        fitobj.popt *= 1./(amp_gain*bu.e_charge/plate_sep)
        fitobj.errs *= 1./(amp_gain*bu.e_charge/plate_sep)
        self.charge_step_calibration = fitobj
        
        
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
        self.ave_force_vs_pos = temp_obj.ave_force_vs_pos

