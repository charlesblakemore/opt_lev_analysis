import h5py, os, re, glob, time, sys, fnmatch
import numpy as np
import datetime as dt
import dill as pickle 

from obspy.signal.detrend import polynomial

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

import scipy.interpolate as interp
import scipy.optimize as optimize
import scipy.signal as signal
import scipy

import configuration
import transfer_func_util as tf

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


#### Generic Helper functions

def progress_bar(count, total, suffix='', bar_len=50):
    '''Prints a progress bar and current completion percentage.
       This is useful when processing many files and ensuring
       a script is actually running and going through each file

           INPUTS: count, current counting index
                   total, total number of iterations to complete
                   suffix, option string to add to progress bar
                   bar_len, length of the progress bar in the console

           OUTPUTS: none
    '''
    
    if count == total - 1:
        percents = 100.0
        bar = '#' * bar_len
    else:
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()



def get_color_map( n, cmap='jet' ):
    '''Gets a map of n colors from cold to hot for use in
       plotting many curves.

           INPUTS: n, length of color array to make
                   cmap, color map for final output

           OUTPUTS: outmap, color map in rgba format'''

    cNorm  = colors.Normalize(vmin=0, vmax=n)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap) #cmap='viridis')
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

def round_sig(x, sig=2):
    '''Round a number to a certain number of sig figs

           INPUTS: x, number to be rounded
                   sig, number of sig figs

           OUTPUTS: num, rounded number'''

    neg = False
    if x == 0:
        return 0
    else:
        if x < 0:
            neg = True
            x = -1.0 * x
        num = round(x, sig-int(math.floor(math.log10(x)))-1)
        if neg:
            return -1.0 * num
        else:
            return num






#### First define some functions to help with the DataFile object. 

def count_dirs(path):
    '''Counts the number of directories (and subdirectories)
       in a given path.

       INPUTS: path, directory name to loop over

       OUTPUTS: numdir, number of directories and subdirectories
                        in the given path'''

    count = 0
    for root, dirs, files in os.walk(path):
        count += len(dirs)

    return count
    

def make_all_pardirs(path):
    '''Function to help pickle from being shit. Takes a path
       and looks at all the parent directories etc and tries 
       making them if they don't exist.

       INPUTS: path, any path which needs a hierarchy already 
                     in the file system before being used

       OUTPUTS: none
       '''

    parts = path.split('/')
    parent_dir = '/'
    for ind, part in enumerate(parts):
        if ind == 0 or ind == len(parts) - 1:
            continue
        parent_dir += part
        parent_dir += '/'
        if not os.path.isdir(parent_dir):
            os.mkdir(parent_dir)



def find_all_fnames(dirlist, ext='.h5', sort=True):
    '''Finds all the filenames matching a particular extension
       type in the directory and its subdirectories .

       INPUTS: dirlist, list of directory names to loop over
               ext, file extension you're looking for
               sort, boolean specifying whether to do a simple sort

       OUTPUTS: files, list of files names as strings'''

    was_list = True

    lengths = []
    files = []

    if type(dirlist) == str:
        dirlist = [dirlist]
        was_list = False

    for dirname in dirlist:
        for root, dirnames, filenames in os.walk(dirname):
            for filename in fnmatch.filter(filenames, '*' + ext):
                files.append(os.path.join(root, filename))
        if was_list:
            if len(lengths) == 0:
                lengths.append(len(files))
            else:
                lengths.append(len(files) - np.sum(lengths)) 
            
    if sort:
        # Sort files based on final index
        files.sort(key = find_str)

    if was_list:
        return files, lengths
    else:
        return files


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



def getdata(fname, gain_error=1.0):
    '''loads a .h5 file from a path into data array and 
       attribs dictionary, converting ADC bits into 
       volatage. The h5 file is closed.'''

    #factor to convert between adc bits and voltage 
    adc_fac = (configuration.adc_params["adc_res"] - 1) / \
               (2. * configuration.adc_params["adc_max_voltage"])

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
    '''takes vector containing data atributes and puts 
       it into a dictionary with key value pairs specified 
       by dict where the keys of dict give the labels and 
       the values specify the index in vec'''
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
                            filter for final smoothing of resp(drive)
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

    # Find the drive frequency, ignoring the DC bin
    fund_ind = np.argmax( np.abs(drivefft[1:]) ) + 1
    drive_freq = freqs[fund_ind]

    meandrive = np.mean(drive)
    mindrive = np.min(drive)
    maxdrive = np.max(drive)

    meanresp = np.mean(resp)

    # Build the notch filter
    drivefilt = np.zeros(len(drivefft))
    drivefilt[fund_ind] = 1.0

    # Error message triggered by verbose option
    if verbose:
        if ( np.abs(drivefft[fund_ind-1]) > 0.01 * np.abs(drivefft[fund_ind]) or \
             np.abs(drivefft[fund_ind+1]) > 0.01 * np.abs(drivefft[fund_ind]) ):
            print "More than 1% power in neighboring bins: spatial binning may be suboptimal"
            sys.stdout.flush()

    # Expand the filter to more than a single bin. This can introduce artifacts
    # that appear like lissajous figures in the resp vs. drive final result
    if width:
        lower_ind = np.argmin(np.abs(drive_freq - 0.5 * width - freqs))
        upper_ind = np.argmin(np.abs(drive_freq + 0.5 * width - freqs))
        drivefilt[lower_ind:upper_ind+1] = drivefilt[fund_ind]

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
    #drivefilt = np.ones_like(drivefilt)
    #drivefilt[0] = 0
    drivefft_filt = drivefilt * drivefft
    respfft_filt = drivefilt * respfft

    # Reconstruct the filtered data
    drive_r = np.fft.irfft(drivefft_filt) + meandrive
    resp_r = np.fft.irfft(respfft_filt) #+ meanresp

    #plt.plot(t, resp_r)
    #plt.figure()

    # Sort reconstructed data, interpolate and resample
    mindrive = np.min(drive_r)
    maxdrive = np.max(drive_r)

    grad = np.gradient(drive_r)

    sortinds = drive_r.argsort()
    drive_r = drive_r[sortinds]
    resp_r = resp_r[sortinds]

    ginds = grad[sortinds] < 0

    bin_spacing = (maxdrive - mindrive) * (1.0 / nbins)
    drivevec = np.linspace(mindrive+0.5*bin_spacing, maxdrive-0.5*bin_spacing, nbins)

    respvec = []
    for bin_loc in drivevec:
        inds = (drive_r > bin_loc - 0.5*bin_spacing) * (drive_r < bin_loc + 0.5*bin_spacing)
        val = np.mean( resp_r[inds] )
        respvec.append(val)

    respvec = np.array(respvec)
    
    #plt.plot(drive_r, resp_r)
    #plt.plot(drive_r[ginds], resp_r[ginds], linewidth=2)
    #plt.plot(drive_r[np.invert(ginds)], resp_r[np.invert(ginds)], linewidth=2)
    #plt.plot(drivevec, respvec, linewidth=5)
    #plt.show()

    #sortinds = drive_r.argsort()
    #interpfunc = interp.interp1d(drive_r[sortinds], resp_r[sortinds], \
    #                             bounds_error=False, fill_value='extrapolate')

    #respvec = interpfunc(drivevec)
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
        self.diag_pos_data = []
        self.cant_data = [] 
        self.electrode_data = []
        self.other_data = []
        self.fsamp = "Fsamp not loaded"
        #Conditions under which data is taken
        self.time = "Time not loaded"#loads time at end of file
        self.temps = []
        self.pressures = {} # loads to dict with keys different gauges 
        self.stage_settings = {} # loads to dict. Look at config.py for keys 
        self.electrode_settings = {} # loads to dict. The key "dc_settings" gives\
            # the dc value on electrodes 0-6. The key "driven_electrodes" 
            # is a list where the electrode index is 1 if the electrode 
            # is driven and 0 otherwise

    def load(self, fname):
        '''Loads the data from file with fname into DataFile object. 
           Does not perform any calibrations.  
        ''' 
        dat, attribs = getdata(fname)
        if len(dat) == 0:
            self.badfile = True
            return 
        else:
            self.badfile = False

        #print attribs
        self.fname = fname
        #print fname
        self.date = fname.split('/')[2]
        dat = dat[configuration.adc_params["ignore_pts"]:, :]
        self.pos_data = np.transpose(dat[:, configuration.col_labels["bead_pos"]])
        self.cant_data = np.transpose(dat[:, configuration.col_labels["stage_pos"]])
        self.electrode_data = np.transpose(dat[:, configuration.col_labels["electrodes"]])
        self.fsamp = attribs["Fsamp"]
        self.time = labview_time_to_datetime(attribs["Time"])
        try:
            self.temps = attribs["temps"]
            # Unpacks pressure gauge vector into dict with
            # labels for pressure gauge specified by configuration.pressure_inds    
            self.pressures = unpack_config_dict(configuration.pressure_inds, \
                                                attribs["pressures"]) 
        except:
            self.temps = 'Temps not loaded!'
            self.pressures = 'Pressures not loaded!'
        # Unpacks stage settings into a dictionay with keys specified by
        # configuration.stage_inds
        self.stage_settings = \
            unpack_config_dict(configuration.stage_inds, \
            attribs["stage_settings"])
        # Load all of the electrode settings into the correct keys
        # First get DC values from its own .h5 attribute
        self.electrode_settings["dc_settings"] = \
            attribs["electrode_dc_vals"][:configuration.num_electrodes]
        
        # Copy "electrode_settings" attribute into numpy array so it can be 
        # indexed by a list of indicies coming from 
        # configuration.eloectrode_settings 
        temp_elec_settings = np.array(attribs["electrode_settings"])
        # First get part with 1 if electrode was driven and 0 else 
        self.electrode_settings["driven"] = \
            temp_elec_settings[configuration.electrode_settings['driven']]
        # Now get array of amplitudes
        self.electrode_settings["amplitudes"] = \
            temp_elec_settings[configuration.electrode_settings['amplitudes']]
        # Now get array of frequencies
        self.electrode_settings["frequencies"] = \
            temp_elec_settings[configuration.electrode_settings['frequencies']]
        # reassign driven electrode_dc_vals because it is overwritten
        dcval_temp = \
            temp_elec_settings[configuration.electrode_settings['dc_vals2']]
        # If an electrode is swept the electrode_dc_vals is not used. Make 
        # sure that if an electrode is drive then the dc_setting comes from 
        # attribs["electrode_settings"]  
        for i, e in enumerate(self.electrode_settings["driven"]):
            if e == 1. and dcval_temp[i] != 0:
                self.electrode_settings["dc_settings"][i] = dcval_temp[i]
                
    def load_other_data(self):
        dat, attribs = getdata(self.fname)
        dat = dat[configuration.adc_params["ignore_pts"]:, :]
        self.other_data = np.transpose(dat[:, configuration.col_labels["other"]])


    def calibrate_stage_position(self):
        '''calibrates voltage in cant_data and into microns. 
           Uses stage position file to put origin of coordinate 
           system at trap in x direction with cantilever centered 
           on trap in y. Looks for stage position file with same 
           path and file name as self.fname '''
        # First get everything into microns.
        for k in configuration.calibrate_stage_keys:
            self.stage_settings[k] *= configuration.stage_cal    
        
        self.cant_data*=configuration.stage_cal
        
        # Now load the cantilever position file.
        # First get the path to the position file from the file base name
        # and get the extension from configuration.extensions["stage_position"].
        filename, file_extension = os.path.splitext(self.fname)
        posfname = \
            os.path.join(filename, configuration.extensions["stage_position"])
        # Load position of course stage. If file cant be found 
        try: 
            pos_arr = pickle.load(open(posfname, "rb"))
        except:
            1 + 2
            #print "shit is fucked"

    
    def detrend_poly(self, order=1, plot=False):
        '''Remove a polynomial of arbitrary order from data.

           INPUTS: order, order of the polynomial to subtract
                   plot, boolean whether to plot detrending result

           OUTPUTS: none, generates new class attribute.'''

        for resp in [0,1,2]:
            self.pos_data[resp] = polynomial(self.pos_data[resp], \
                                             order=order, plot=plot)

        if len(self.other_data):
            for ax in [0,1,2,3,4]:
                self.other_data[ax] = polynomial(self.other_data[ax], \
                                                 order=order, plot=plot)


    def high_pass_filter(self, order=1, fc=1.0):
        '''Apply a digital butterworth type filter to the data

           INPUTS: order, order of the butterworth filter
                   fc, cutoff frequency in Hertz

           OUTPUTS: none, generates new class attribute.'''

        Wn = 2.0 * fc / self.fsamp
        b, a = signal.butter(order, Wn, btype='highpass')

        for resp in [0,1,2]:
            self.pos_data[resp] = signal.filtfilt(b, a, self.pos_data[resp])


    def diagonalize(self, date='', interpolate=False, maxfreq=1000, plot=False):
        '''Diagonalizes data, adding a new attribute to the DataFile object.

           INPUTS: date, date in form YYYYMMDD if you don't want to use
                         default TF from file date
                   interpolate, boolean specifying whether to use an 
                                interpolating function or fit to damped HO
                   maxfreq, max frequency above which data is top-hat filtered

           OUTPUTS: none, generates new class attribute.'''

        tf_path = '/calibrations/transfer_funcs/'
        ext = configuration.extensions['trans_fun']
        if not len(date):
            tf_path +=  self.date
        else:
            tf_path += date

        if interpolate:
            tf_path += '_interp' + ext
        else:
            tf_path += ext

        # Load the transfer function. Note that this Hfunc maps
        # drive -> response, so we will need to invert
        try:
            Hfunc = pickle.load(open(tf_path, 'rb'))
        except:
            print "Couldn't automatically find correct TF"
            return

        # Generate FFT frequencies for given data
        N = len(self.pos_data[0])
        freqs = np.fft.rfftfreq(N, d=1.0/self.fsamp)

        # Compute TF at frequencies of interest. Appropriately inverts
        # so we can map response -> drive
        Harr = tf.make_tf_array(freqs, Hfunc)

        if plot:
            tf.plot_tf_array(freqs, Harr)

        maxfreq_ind = np.argmin( np.abs(freqs - maxfreq) )
        Harr[maxfreq_ind:,:,:] = 0.0+0.0j

        f_ind = np.argmin( np.abs(freqs - 41) )
        mat = Harr[f_ind,:,:]
        conv_facs = [0, 0, 0]
        for i in [0,1,2]:
            conv_facs[i] = np.abs(mat[i,i])
        self.conv_facs = conv_facs

        # Compute the FFT, apply the TF and inverse FFT
        data_fft = np.fft.rfft(self.pos_data)
        diag_fft = np.einsum('ikj,ki->ji', Harr, data_fft)

        if plot:
            fig, axarr = plt.subplots(3,1,sharex=True,sharey=True)
            for ax in [0,1,2]:
                axarr[ax].loglog(freqs, np.abs(data_fft[ax])*conv_facs[ax])
                axarr[ax].loglog(freqs, np.abs(diag_fft[ax]))
            plt.tight_layout()
            plt.show()

        self.diag_pos_data = np.fft.irfft(diag_fft)




    def get_force_v_pos(self, nbins=100, nharmonics=10, width=0, \
                        sg_filter=False, sg_params=[3,1], verbose=True, \
                        cantilever_drive=True, electrode_drive=False, \
                        fakedrive=False, fakefreq=50, fakeamp=80, fakephi=0):
        '''Sptially bins X, Y and Z responses against driven cantilever axis,
           or in the case of multiple axes driven simultaneously, against the
           drive with the largest amplitude.

           INPUTS: nbins, number of output bins
                   nharmonics, number of harmonics to include in fourier
                               binning procedure
                   width, bandwidth, in Hz, of notch filter
                   sg_filter, boolean specifying use of Savitsky-Golay filter
                   sg_params, parameters for Savitsky-Golay filter
                   verbose, boolean for some extra text outputs
                   cantilever_drive, boolean to specify binning against cant
                   electrode_drive, boolean to bin against an electrode drive
                                    for reconstruction testing
                   fakedrive, boolean to use a fake drive signal
                   fakefreq, frequency of fake drive
                   fakeamp, fake amplitude in microns
                   fakephi, fake phase for fake drive

           OUTPUTS: none, generates new class attribute'''

        if cantilever_drive and not fakedrive:
            # First, find which axes were driven. If multiple are found,
            # it takes the axis with the largest amplitude
            indmap = {0: 'x', 1: 'y', 2: 'z'}
            driven = [0,0,0]
            for ind, key in enumerate(['x driven','y driven','z driven']):
                if self.stage_settings[key]:
                    driven[ind] = 1
            if np.sum(driven) > 1:
                amp = [0,0,0]
                for ind, val in enumerate(driven):
                    if val: 
                        key = indmap[ind] + ' amp'
                        amp[ind] = self.stage_settings[key]
                cant_ind = np.argmax(np.abs(amp))
            else:
                cant_ind = np.argmax(np.abs(driven))
            drivevec = self.cant_data[cant_ind]

        elif cantilever_drive and fakedrive:
            numsamp = len(self.pos_data[0])
            dt = 1.0 / self.fsamp
            t = np.linspace(0, numsamp - 1, numsamp) * dt
            drivevec = fakeamp * np.sin(2.0 * np.pi * fakefreq * t + fakephi) + fakeamp

        if electrode_drive:
            elec_ind = np.argmax(self.electrode_settings['driven'])
            drivevec = self.electrode_data[elec_ind]
            

        # Bin responses against the drive. If data has been diagonalized,
        # it bins the diagonal data as well
        dt = 1. / self.fsamp
        binned_data = [[0,0], [0,0], [0,0]]
        if len(self.diag_pos_data):
            diag_binned_data = [[0,0], [0,0], [0,0]]
        for resp in [0,1,2]:
            bins, binned_vec = spatial_bin(drivevec, self.pos_data[resp], dt, \
                                           nbins = nbins, nharmonics = nharmonics, \
                                           width = width, sg_filter = sg_filter, \
                                           sg_params = sg_params, verbose = verbose)
            binned_data[resp][0] = bins
            binned_data[resp][1] = binned_vec
            
            if len(self.diag_pos_data):
                diag_bins, diag_binned_vec = \
                            spatial_bin(drivevec, self.diag_pos_data[resp], dt, \
                                        nbins = nbins, nharmonics = nharmonics, \
                                        width = width, sg_filter = sg_filter, \
                                        sg_params = sg_params, verbose = verbose)

                diag_binned_data[resp][0] = diag_bins
                diag_binned_data[resp][1] = diag_binned_vec

        self.binned_data = binned_data
        if len(self.diag_pos_data):
            self.diag_binned_data = diag_binned_data
                

