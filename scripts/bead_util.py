import h5py, os, re, glob
import numpy as np
import datetime as dt
import configuration
import dill as pickle 

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
    '''copies an hdf5 attributes into a new dictionary so the original file can be closed.'''
    new_dict = {}
    for k in attribs.keys():
        new_dict[k] = attribs[k]
    return new_dict



def getdata(fname, gain_error=1.0, adc_max_voltage=10., adc_res=2**16):
    '''loads a .h5 file from a path into data array and attribs dictionary, converting ADC bits into volatage. The h5 file is closed.'''

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
    '''Convert a labview timestamp (i.e. time since 1904) to a  more useful format (python datetime object)'''
    
    ## first get number of seconds between Unix time and Labview's
    ## arbitrary starting time
    lab_time = dt.datetime(1904, 1, 1, 0, 0, 0)
    nix_time = dt.datetime(1970, 1, 1, 0, 0, 0)
    delta_seconds = (nix_time-lab_time).total_seconds()

    lab_dt = dt.datetime.fromtimestamp( lt - delta_seconds)
    
    return lab_dt


class DataFile:
    '''Class holing all of the data for an individual file. Contains methods to  apply calibrations to the data, including image coordinate correction. Also contains methods to change basis from time data to cantilever position data.
    '''

    def __init__(self):
        '''Initializes the an empty DataFile object. All of the attributes are filled with strings
        '''
        self.fname = "Filename not assigned."
        #Data and data parameters
        self.pos_data = []
        self.cant_data = [] 
        self.electrode_data = []
        self.Fsamp = "Fsamp not loaded"
        #Conditions under which data is taken
        self.time = "Time not loaded"
        self.temps = []
        self.pressures = {} 
        self.stage_settings = []
        self.electrode_settings = []

    def load(self, fname):
        '''Loads the data from file with fname into DataFile object. Does not perform any calibrations.  
        ''' 
        dat, attribs= bu.getdata(fname)
        self.fname = fname 
        dat = dat[configuration.adc_params["ignore_pts"]:, :]
        #data and FSamp
        self.pos_data = dat[:, configuration.col_labels["bead_pos"]]
        self.cant_data = dat[:, configuration.col_labels["stage_pos"]]
        self.electrode_data = dat[:, configuration.col_labels["electrodes"]]
        self.Fsamp = attribs["Fsamp"] # Sampling frequency of the data
        #Data conditions
        self.Time = labview_time_to_datetime(attribs["Time"]) # Time of end of file
        self.temps = attribs["temps"] # Vector of thermocouple temperatures 
        ptemp = attribs["pressures"] # temporarlily hold pressures
        #loop over presssure gauges and get all gauges from right columns.
        for k in configuration.pressures.keys():
            self.pressures[k] = ptemp[configuration.pressures[k]]
         

        # Electrode front pannel settings for all files in the directory.
        # first 8 are ac amps, second 8 are frequencies, 3rd 8 are dc vals 
        self.electrode_settings = attribs["electrode_settings"]

        # Front pannel settings applied to this particular file. 
        # Top boxes independent of the sweeps
        self.electrode_dc_vals = attribs["electrode_dc_vals"] 

        # Front pannel settings for the stage for this particular file.
        self.stage_settings = attribs['stage_settings'] 
        self.stage_settings[:3]*=cant_cal #calibrate stage_settings
        # Data vectors and their transforms
        self.pos_data = np.transpose(dat[:, 0:3]) #x, y, z bead position
        self.other_data = np.transpose(dat[:,3:7])
        self.dc_pos =  np.mean(self.pos_data, axis = -1)
        self.image_pow_data = np.transpose(dat[:, 6])

        #self.pos_data = np.transpose(dat[:,[elec_inds[1],elec_inds[3],elec_inds[5]]])
        self.cant_data = np.transpose(dat[:, 17:20])*cant_cal
        # Record of voltages on the electrodes
        try:
            self.electrode_data = np.transpose(dat[:, elec_inds]) 
        except:
            print "No electrode data for %s" % self.fname

        f.close()

    def get_image_data(self, trapx_pixel, img_cal_path, make_plot=False):
        imfile = self.fname + '.npy'
        val = imu.measure_image1d(imfile, trapx_pixel, img_cal_path, make_plot=make_plot)
        self.image_data = np.abs(val)
        

    def get_stage_settings(self, axis=0):
        # Function to intelligently extract the stage settings data for a given axis
        if axis == 0:
            mask = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        elif axis == 1:
            mask = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
        elif axis == 2:
            mask = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        #print self.stage_settings[mask]
        return self.stage_settings[mask]

    def detrend(self):
        # Remove linear drift from data
        for i in [0,1,2]:
            dat = self.pos_data[i]
            x = np.array(range(len(dat)))
            popt, pcov = curve_fit(bu.trend_fun, x, dat)
            self.pos_data[i] = dat - (popt[0]*x + popt[1])
            

    def ms(self):
        #mean subtracts the position data.
        ms = lambda vec: vec - np.mean(vec)
        self.pos_data  = map(ms, self.pos_data)



    def diagonalize(self, Harr, cantfilt=False):

        diag_fft = np.einsum('ikj,ki->ji', Harr, self.data_fft)
        self.diag_pos_data = np.fft.irfft(diag_fft)
        self.diag_data_fft = diag_fft
        if cantfilt:
            diag_fft2 = np.einsum('ikj,ki->ji', Harr, self.cantfilt * self.data_fft)
            self.diag_pos_data_cantfilt = np.fft.irfft(diag_fft2) 


