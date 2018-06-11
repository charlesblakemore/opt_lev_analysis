import h5py, os, re, glob, time, sys, fnmatch
import numpy as np
import datetime as dt
import dill as pickle 

from obspy.signal.detrend import polynomial

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.mlab as mlab

import scipy.interpolate as interp
import scipy.optimize as optimize
import scipy.signal as signal
import scipy

import configuration
import transfer_func_util as tf

from bead_util_funcs import *

#######################################################
# This module contains the DataFile class which stores
# information from a single hdf5 file. Includes class
# methods to calibrate/diagonalize data, bin data into 
# force vs position, plot cantilever drive data, create 
# harmonic notch filters from cantilever drive data, 
# extracting data/errors at the notch filter fft bins
#
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
# ----------------------------------------------------
### Imports a number of utility functions from a 
### companion module called bead_util_funcs.
### When using in scripts, it is only necessary to 
### import this module, not both.
# ----------------------------------------------------
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
#
#######################################################



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
        self.cant_calibrated = False

    def load_only_attribs(self, fname):
        dat, attribs = getdata(fname)
        if len(dat) == 0:
            self.badfile = True
            return 
        else:
            self.badfile = False

        self.fname = fname
        self.date = fname.split('/')[2]

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


    def load(self, fname, load_FPGA = True):
        '''Loads the data from file with fname into DataFile object. 
           Does not perform any calibrations.  
        ''' 
        dat, attribs = getdata(fname)
        if len(dat) == 0:
            self.badfile = True
            return 
        else:
            self.badfile = False
        
        self.time = attribs["Time"]   # unix epoch time in ns (time.time() * 10**9)
        self.fsamp = attribs["Fsamp"]
        self.nsamp = len(dat[:,0])

        self.daqmx_time = np.linspace(0,self.nsamp-1,self.nsamp) * (1.0/self.fsamp) \
                               * (10**9) + self.time

        fpga_fname = fname[:-3] + '_fpga.h5'
        if load_FPGA:
            fpga_dat = get_fpga_data(fpga_fname, verbose=True, timestamp=self.time)

            fpga_dat = sync_and_crop_fpga_data(fpga_dat, self.time, self.nsamp)

            self.pos_data = fpga_dat['xyz']
            self.pos_time = fpga_dat['xyz_time']
        
            # Load quadrant and backscatter amplitudes and phases
            self.amp = fpga_dat['amp']
            self.phase = fpga_dat['phase']
            self.quad_time = fpga_dat['quad_time']
        #print attribs
        self.fname = fname
        #print fname
        self.date = fname.split('/')[2]
        dat = dat[configuration.adc_params["ignore_pts"]:, :]

        ###self.pos_data = np.transpose(dat[:, configuration.col_labels["bead_pos"]])

        self.cant_data = np.transpose(dat[:, configuration.col_labels["stage_pos"]])
        self.electrode_data = np.transpose(dat[:, configuration.col_labels["electrodes"]])
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
        if self.cant_calibrated:
            return

        # First get everything into microns.
        for k in configuration.calibrate_stage_keys:
            #print k
            self.stage_settings[k] *= configuration.stage_cal    
            
        try:
            self.cant_data*=configuration.stage_cal
            self.cant_calibrated = True
        except:
            1+2
        
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

    def get_cant_drive_ax(self):
        '''Determine the index of cant_data with the largest drive voltage,
           which is either exrtacted from stage setting or determined
           from the RMS of the stage monitor.

           INPUTS: none, uses class attributes from loaded data

           OUTPUTS: none, generates new class attribute.'''
        indmap = {0: 'x', 1: 'y', 2: 'z'}
        driven = [0,0,0]
        
        if len(self.stage_settings) == 0:
            print "No data loaded..."
            return 

        for ind, key in enumerate(['x driven','y driven','z driven']):
            if self.stage_settings[key]:
                driven[ind] = 1
        if np.sum(driven) > 1:
            amp = [0,0,0]
            for ind, val in enumerate(driven):
                if val: 
                    key = indmap[ind] + ' amp'
                    amp[ind] = self.stage_settings[key]
            drive_ind = np.argmax(np.abs(amp))
        if np.sum(driven) == 0: # handel case of external drive
            drive_fft = np.fft.rfft(self.cant_data)
            mean_sq = np.sum(np.abs(drive_fft[:, 1:])**2, axis = 1)#cut DC
            drive_ind = np.argmax(mean_sq)

        else:
            drive_ind = np.argmax(np.abs(driven))

        return drive_ind


    def build_cant_filt(self, cant_fft, freqs, nharmonics=10, width=0, harms=[]):
        '''Identify the fundamental drive frequency and make a notch filter
           with the number of harmonics requested.

           INPUTS: cant_fft, fft of cantilever drive
                   freqs, array of frequencies associated to data ffts
                   harms, number of harmonics to included
                   width, width of the notch filter in Hz

           OUTPUTS: none, generates new class attribute.'''

        # Find the drive frequency, ignoring the DC bin
        fund_ind = np.argmax( np.abs(cant_fft[1:]) ) + 1
        drive_freq = freqs[fund_ind]

        drivefilt = np.zeros(len(cant_fft))
        drivefilt[fund_ind] = 1.0

        if width:
            lower_ind = np.argmin(np.abs(drive_freq - 0.5 * width - freqs))
            upper_ind = np.argmin(np.abs(drive_freq + 0.5 * width - freqs))
            drivefilt[lower_ind:upper_ind+1] = drivefilt[fund_ind]

        if len(harms) == 0:
            # Generate an array of harmonics
            harms = np.array([x+2 for x in range(nharmonics)])
        elif 1 not in harms:
            drivefilt[fund_ind] = 0.0
            if width:
                drivefilt[lower_ind:upper_ind+1] = drivefilt[fund_ind]

        # Loop over harmonics and add them to the filter
        for n in harms:
            harm_ind = np.argmin( np.abs(n * drive_freq - freqs) )
            drivefilt[harm_ind] = 1.0 
            if width:
                h_lower_ind = harm_ind - (fund_ind - lower_ind)
                h_upper_ind = harm_ind + (upper_ind - fund_ind)
                drivefilt[h_lower_ind:h_upper_ind+1] = drivefilt[harm_ind]

        return drivefilt, fund_ind, drive_freq
    

    def get_boolean_cantfilt(self, ext_cant_drive=False, ext_cant_ind=1, \
                             nharmonics=10, harms=[], width=0):
        '''Builds a boolean notch filter for the cantilever drive

           INPUTS: ext_cant_drive, bool to specify whether cantilever drive
                                   is from an external source
                   ext_cant_ind, index being driven externally

           OUTPUTS: ginds, bool array of length NFFT (set by hdf5 file).'''

        drive_ind = self.get_cant_drive_ax()
        if ext_cant_drive:
            drive_ind = ext_cant_ind

        drivevec = self.cant_data[drive_ind]
        drivefft = np.fft.rfft(drivevec)

        freqs = np.fft.rfftfreq(len(drivevec), d=1.0/self.fsamp)

        drivefilt, fund_ind, drive_freq = \
                    self.build_cant_filt(drivefft, freqs, nharmonics=nharmonics, \
                                         harms=harms, width=width)

        # Apply filter by indexing with a boolean array
        ginds = drivefilt > 0

        return ginds, fund_ind, drive_freq, drive_ind


    def get_datffts_and_errs(self, ginds, drive_freq, noiseband=10, plot=False, diag=False):   
        '''Applies a cantilever notch filter and returns the filtered data
           with an error estimate based on the neighboring bins of the PSD

           INPUTS: ginds, boolean cantilever drive notch filter
                   drive_freq, cantilever drive freq

           OUTPUTS: datffts, ffts evalutated at ginds
                    diagdatffts, diagffts evaluated at ginds
                    daterrs, errors from surrounding bins
                    diagdaterrs, diag errors from surrounding bins
        ''' 

        datffts = [[], [], []]
        daterrs = [[], [], []]

        if diag:
            diagdatffts = [[], [], []]
            diagdaterrs = [[], [], []]

        freqs = np.fft.rfftfreq(len(self.pos_data[0]), d=1.0/self.fsamp)
        fund_ind = np.argmin(np.abs(freqs - drive_freq))
        bin_sp = freqs[1] - freqs[0]

        harm_freqs = freqs[ginds]

        if type(harm_freqs) == np.float64:
            harm_freqs = np.array([harm_freqs])
            just_one = True
        else:
            just_one = False

        for resp in [0,1,2]:

            N = len(self.pos_data[resp])

            datfft = np.fft.rfft(self.pos_data[resp]*self.conv_facs[resp])
            datffts[resp] = datfft[ginds]
            daterrs[resp] = np.zeros_like(datffts[resp])
            
            if diag:
                diagdatfft = np.fft.rfft(self.diag_pos_data[resp])
                diagdatffts[resp] = diagdatfft[ginds]
                diagdaterrs[resp] = np.zeros_like(datffts[resp])

            for freqind, freq in enumerate(harm_freqs):
                harm_ind = np.argmin(np.abs(freqs-freq))
                noise_inds = np.abs(freqs - freq) < 0.5*noiseband
                noise_inds[harm_ind] = False
                if freqind == 0:
                    noise_inds_init = noise_inds

                errval = np.median(np.abs(datfft[noise_inds]))
                if just_one:
                    daterrs[resp] = errval
                else:
                    daterrs[resp][freqind] = errval
                #daterrs[resp][freqind] = np.abs(datfft[harm_ind])
                if diag:
                    diagerrval = np.median(np.abs(diagdatfft[noise_inds]))
                    if just_one:
                        diagdaterrs[resp] = diagerrval
                    else:
                        diagdaterrs[resp][freqind] = diagerrval
                    #diagdaterrs[resp][freqind] = np.abs(diagdatfft[harm_ind])

            if plot:
                normfac = np.sqrt(bin_sp)*fft_norm(N, self.fsamp)

                plt.figure()
                plt.loglog(freqs, np.abs(datfft)*normfac, alpha=0.4)
                plt.loglog(freqs[noise_inds_init], np.abs(datfft[noise_inds_init])*normfac)
                plt.loglog(freqs[ginds], np.abs(datfft[ginds])*normfac, '.', ms=10)
                plt.ylabel('Force [N]')
                plt.xlabel('Frequency [Hz]')

                plt.figure()
                plt.plot(datfft[ginds].real * normfac, label='real')
                plt.plot(datfft[ginds].imag * normfac, label='imag')
                plt.plot(np.sqrt(2)*daterrs[resp] * normfac, label='errs')
                plt.legend()
                plt.show()

        datffts = np.array(datffts)
        daterrs = np.array(daterrs)
        if diag:
            diagdatffts = np.array(diagdatffts)
            diagdaterrs = np.array(diagdaterrs)

        if not diag:
            diagdatffts = np.zeros_like(datffts)
            diagdaterrs = np.zeros_like(daterrs)

        return datffts, diagdatffts, daterrs, diagdaterrs


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
        Harr[maxfreq_ind+1:,:,:] = 0.0+0.0j

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


    


    def plot_cant_asd(self, drive_ax, np_mlab_compare=False):
        '''Plots the ASD = sqrt(PSD) of a given cantilever drive. This is useful
           for debugging and such

           INPUTS: drive_ax, [0,1,2] axis of cantilever drive

           OUTPUTS: none, plots stuff'''
        
        if np_mlab_compare:
            fac = np.sqrt(2.0 /  (len(self.pos_data[0]) * self.fsamp))
            fftfreqs = np.fft.rfftfreq(len(self.pos_data[0]), d=1.0/self.fsamp)
            drivepsd = np.abs(np.fft.rfft(self.cant_data[drive_ax])) * fac
            plt.loglog(fftfreqs, drivepsd, label='NumPy FFT amplitude')

        drivepsd2, freqs2 = mlab.psd(self.cant_data[drive_ax], NFFT=len(self.pos_data[0]), \
                                     Fs=self.fsamp, window=mlab.window_none)
        
        plt.loglog(freqs2, np.sqrt(drivepsd2), label='Mlab PSD')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('ASD [$\mu$m/rt(Hz)]')
        if np_mlab_compare:
            plt.legend(loc=0)
        plt.show()




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
            
            cant_ind = self.get_cant_drive_ax()
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
                

