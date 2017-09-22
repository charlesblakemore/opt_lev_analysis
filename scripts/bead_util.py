import h5py, os, re, glob, time, sys
import numpy as np
import datetime as dt
import dill as pickle 

import scipy.interpolate as interp
import scipy.optimize as optimize
import scipy.signal as signal
import scipy

import configuration

#######################################################
# This module has basic utility functions for analyzing bead
# data. In particular, this module has the basic data
# loading function, bead/physical constants and bead
# spectra.
#
# This version has been significantly trimmed from previous
# bead_util in an attempt to force modularization.
# Previous code for millicharge and chameleon data
# can be found by reverting opt_lev_analysis
#######################################################


####First define some functions to help with the DataFile object. 

def find_str(str):
    '''finds the index from the standard file name format'''
    idx_offset = 1e10 # Large number to ensure sorting by index first

    fname, _ = os.path.splitext(str)
    
    endstr = re.findall("\d+mV_[\d+Hz_]*[a-zA-Z]*[\d+]*", fname)
    if( len(endstr) != 1 ):
        # Couldn't find expected pattern, so return the 
        # second to last number in the string
        return int(re.findall('\d+', fname)[-1])

    # Check for an index number
    sparts = endstr[0].split("_")
    if ( len(sparts) >= 3 ):
        return idx_offset*int(sparts[2]) + int(sparts[0][:-2])
    else:
        return int(sparts[0][:-2])

def copy_attribs(attribs):
    '''copies an hdf5 attributes into a new dictionary 
       so the original file can be closed.'''
    new_dict = {}
    for k in attribs.keys():
        new_dict[k] = attribs[k]
    return new_dict



def getdata(fname, gain_error=1.0, adc_max_voltage=10., adc_res=2**16):
    '''loads a .h5 file from a path into data array and 
       attribs dictionary, converting ADC bits into 
       volatage. The h5 file is closed.'''

    #factor to convert between adc bits and voltage 
    adc_fac = (configuration.adc_params["adc_res"] - 1) / (2. * configuration.adc_params["adc_max_voltage"])

    try:
        f = h5py.File(fname,'r')
        dset = f['beads/data/pos_data']
        dat = np.transpose(dset)
        dat = dat / adc_fac
        attribs = copy_attribs(dset.attrs)
        f.close()

    except (KeyError, IOError):
        print "Warning, got no keys for: ", fname
        dat = []
        attribs = {}
        f = []

    return dat, attribs

def labview_time_to_datetime(lt):
    '''Convert a labview timestamp (i.e. time since 1904) to a  
       more useful format (python datetime object)'''
    
    ## first get number of seconds between Unix time and Labview's
    ## arbitrary starting time
    lab_time = dt.datetime(1904, 1, 1, 0, 0, 0)
    nix_time = dt.datetime(1970, 1, 1, 0, 0, 0)
    delta_seconds = (nix_time-lab_time).total_seconds()

    lab_dt = dt.datetime.fromtimestamp( lt - delta_seconds)
    
    return lab_dt

def unpack_config_dict(dic, vec):
    '''takes vector containing data atributes and puts it into a dictionary with key value pairs specified by dict where the keys of dict give the labels and the values specify the index in vec'''
    out_dict = {}
    for k in dic.keys():
        out_dict[k] = vec[dic[k]]
    return out_dict 




def spatial_bin(drive, resp, dt, nbins=100, nharmonics=10, width=0, \
                sg_filter=False, sg_params=[3,1], verbose=True):
    '''Given two waveforms drive(t) and resp(t), this function generates
       resp(drive) with a fourier method. drive(t) should be a pure tone,
       such as a single frequency cantilever drive (although the 
       existence of harmonics is fine). Ideally, the frequency with
       the dominant power should lie in a single DTFT bin.
       Behavior of this function is somewhat indeterminant when there
       is significant spectral leakage into neighboring bins.

       INPUT:   drive, single frequency drive signal, sampled with some dt
       	        resp, arbitrary response to be 'binned'
       	        dt, sample spacing in seconds [s]
                nbins, number of samples in the final resp(drive)
       	        nharmonics, number of harmonics to include in filter
       	        width, filter width in Hertz [Hz]
                sg_filter, boolean value indicating use of a Savitsky-Golay 
                            filter for final smoothing
                sg_params, parameters of the savgol filter 
                            (see scipy.signal.savgol_filter for explanation)

       OUTPUT:  drivevec, vector of drive values, monotonically increasing
                respvec, resp as a function of drivevec'''

    def fit_fun(t, A, f, phi, C):
        return A * np.sin(2 * np.pi * f * t + phi) + C

    Nsamp = len(drive)
    if len(resp) != Nsamp:
        if verbose:
            print "Data Error: x(t) and f(t) don't have the same length"
            sys.stdout.flush()
        return

    # Generate t array
    t = np.linspace(0, len(drive) - 1, len(drive)) * dt

    # Generate FFTs for filtering
    drivefft = np.fft.rfft(drive)
    respfft = np.fft.rfft(resp)
    freqs = np.fft.rfftfreq(len(drive), d=dt)

    # Find the drive frequency
    fund_ind = np.argmax( np.abs(drivefft[1:]) ) + 1
    drive_freq = freqs[fund_ind]

    meandrive = np.mean(drive)
    mindrive = np.min(drive)
    maxdrive = np.max(drive)

    # Build the notch filter
    drivefilt = np.zeros(len(drivefft))
    drivefilt[fund_ind] = 1.0

    if ( np.abs(drivefft[fund_ind-1]) > 0.01 * np.abs(drivefft[fund_ind]) or \
            np.abs(drivefft[fund_ind+1]) > 0.01 * np.abs(drivefft[fund_ind]) ):
        if verbose:
            print "More than 1\% power in neighboring bins: spatial binning may be suboptimal"
            sys.stdout.flush()

    # Expand the filter to more than a single bin. This can introduce artifacts
    # that appear like lissajous figures in the resp vs. drive final result
    if width:
        lower_ind = np.argmin(np.abs(drive_freq - 0.5 * width - freqs))
        upper_ind = np.argmin(np.abs(drive_freq + 0.5 * width - freqs))
        drivefilt[lower_ind:upper_ind+1] = cantfilt[fund_ind]

    # Generate an array of harmonics
    harms = np.array([x+2 for x in range(nharmonics)])

    # Loop over harmonics and add them to the filter
    for n in harms:
        harm_ind = np.argmin( np.abs(n * drive_freq - freqs) )
        drivefilt[harm_ind] = 1.0 
        if width:
            h_lower_ind = harm_ind - (fund_ind - lower_ind)
            h_upper_ind = harm_ind + (upper_ind - fund_ind)
            drivefilt[h_lower_ind:h_upper_ind+1] = drivefilt[harm_ind]

    # Apply the filter to both drive and response
    drivefft_filt = drivefilt * drivefft
    respfft_filt = drivefilt * respfft

    # Reconstruct the filtered data
    drive_r = np.fft.irfft(drivefft_filt) + meandrive
    resp_r = np.fft.irfft(respfft_filt)

    # Sort reconstructed data, interpolate and resample
    drivevec = np.linspace(mindrive, maxdrive, nbins)

    sortinds = drive_r.argsort()
    interpfunc = interp.interp1d(drive_r[sortinds], resp_r[sortinds], fill_value='extrapolate')

    respvec = interpfunc(drivevec)

    if sg_filter:
        respvec = signal.savgol_filter(respvec, sg_params[0], sg_params[1])

    return drivevec, respvec



class DataFile:
    '''Class holing all of the data for an individual file. 
       Contains methods to  apply calibrations to the data, 
       including image coordinate correction. Also contains 
       methods to change basis from time data to cantilever 
       position data.
    '''

    def __init__(self):
        '''Initializes the an empty DataFile object. 
           All of the attributes are filled with strings
        '''
        self.fname = "Filename not assigned."
        #Data and data parameters
        self.pos_data = []
        self.cant_data = [] 
        self.electrode_data = []
        self.fsamp = "Fsamp not loaded"
        #Conditions under which data is taken
        self.time = "Time not loaded"#loads time at end of file
        self.temps = []
        self.pressures = {}#loads to dict with keys different gauges 
        self.stage_settings = {}#loads to dict. Look at config.py for keys 
        self.electrode_settings = {}#loads to dict. The key "dc_settings" gives\
            #the dc value on electrodes 0-6. The key "driven_electrodes" is a list where the electrode index is 1 if the electrode is driven and 0 otherwise

    def load(self, fname):
        '''Loads the data from file with fname into DataFile object. 
           Does not perform any calibrations.  
        ''' 
        dat, attribs= getdata(fname)
        self.fname = fname 
        dat = dat[configuration.adc_params["ignore_pts"]:, :]
        self.pos_data = dat[:, configuration.col_labels["bead_pos"]]
        self.cant_data = dat[:, configuration.col_labels["stage_pos"]]
        self.electrode_data = dat[:, configuration.col_labels["electrodes"]]
        self.fsamp = attribs["Fsamp"]
        self.time = labview_time_to_datetime(attribs["Time"])
        self.temps = attribs["temps"] 
        self.pressures = \
	    unpack_config_dict(configuration.pressure_inds, \
            attribs["pressures"]) 

        self.stage_settings = \
            unpack_config_dict(configuration.stage_inds, \
            attribs["stage_settings"])
        #load all of the electrode settings into the correct keys
        self.electrode_settings["dc_settings"] = \
            attribs["electrode_dc_vals"][:configuration.num_electrodes]

        temp_elec_settings = np.array(attribs["electrode_settings"])

        self.electrode_settings["driven"] = \
            temp_elec_settings[configuration.electrode_settings['driven']]
        self.electrode_settings["amplitudes"] = \
            temp_elec_settings[configuration.electrode_settings['amplitudes']]
        self.electrode_settings["frequencies"] = \
            temp_elec_settings[configuration.electrode_settings['frequencies']]
        #reasign driven electrode_dc_vals because it is overwritten
        dcval_temp = \
            temp_elec_settings[configuration.electrode_settings['dc_vals2']]  
        for i, e in enumerate(self.electrode_settings["driven"]):
            if e == 1.:
                self.electrode_settings["dc_settings"][i] = dcval_temp[i]
                

    def get_force_curve(self):
        return


    def calibrate_stage_position(self):
        '''calibrates voltage in cant_data and into microns. 
           Uses stage position file to put origin of coordinate 
           system at trap in x direction with cantilever centered 
           on trap in y. Looks for stage position file with same 
           path and file name as slef.fname '''
        #First get everything into microns.
        for k in configuration.calibrate_stage_keys:
            self.stage_settings[k] *= configuration.stage_cal    


    def diagonalize(self, Harr, cantfilt=False):

        diag_fft = np.einsum('ikj,ki->ji', Harr, self.data_fft)
        self.diag_pos_data = np.fft.irfft(diag_fft)
        self.diag_data_fft = diag_fft
        if cantfilt:
            diag_fft2 = np.einsum('ikj,ki->ji', Harr, self.cantfilt * self.data_fft)
            self.diag_pos_data_cantfilt = np.fft.irfft(diag_fft2) 


