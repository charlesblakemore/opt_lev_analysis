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

def unpack_config_dict(dic, vec):
    '''takes vector containing data atributes and puts it into a dictionary with key value pairs specified by dict where the keys of dict give the labels and the values specify the index in vec'''
    out_dict = {}
    for k in dic.keys():
        out_dict[k] = vec[dic[k]]
    return out_dict 


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
        self.fsamp = "Fsamp not loaded"
        #Conditions under which data is taken
        self.time = "Time not loaded"#loads time at end of file
        self.temps = []
        self.pressures = {}#loads to dict with keys different gauges 
        self.stage_settings = {}#loads to dict. Look at config.py for keys 
        self.electrode_settings = {}#loads to dict. The key "dc_settings" gives\
            #the dc value on electrodes 0-6. The key "driven_electrodes" is a list where the electrode index is 1 if the electrode is driven and 0 otherwise

    def load(self, fname):
        '''Loads the data from file with fname into DataFile object. Does not perform any calibrations.  
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
            temp_elec_settings[configuration.electrode_settings['driven']]
        self.electrode_settings["frequencies"] = \
            temp_elec_settings[configuration.electrode_settings['frequencies']]
        #reasign driven electrode_dc_vals because it is overwritten
        dcval_temp = \
            temp_elec_settings[configuration.electrode_settings['dc_vals2']]  
        for i, e in enumerate(self.electrode_settings["driven"]):
            if e == 1.:
                self.electrode_settings["dc_settings"][i] = dcval_temp[i]
                
        


    def diagonalize(self, Harr, cantfilt=False):

        diag_fft = np.einsum('ikj,ki->ji', Harr, self.data_fft)
        self.diag_pos_data = np.fft.irfft(diag_fft)
        self.diag_data_fft = diag_fft
        if cantfilt:
            diag_fft2 = np.einsum('ikj,ki->ji', Harr, self.cantfilt * self.data_fft)
            self.diag_pos_data_cantfilt = np.fft.irfft(diag_fft2) 


