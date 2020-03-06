import h5py, os, re, glob, time, sys, fnmatch, inspect, traceback
import numpy as np
import datetime as dt
import dill as pickle 

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
from bead_data_funcs import *
from bead_properties import *
from stats_util import *

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

        fname = os.path.abspath(fname)

        dat, attribs = getdata(fname)
        if len(dat) == 0:
            self.badfile = True
            return 
        else:
            self.badfile = False

        self.fname = fname
        self.date = re.search(r"\d{8,}", fname)[0]

        self.fsamp = attribs["Fsamp"]
        self.time = attribs["Time"]

        try:
            self.temps = attribs["temps"]
            # Unpacks pressure gauge vector into dict with
            # labels for pressure gauge specified by configuration.pressure_inds    
            self.pressures = unpack_config_dict(configuration.pressure_inds, \
                                                attribs["pressures"]) 
        except Exception:
            self.temps = 'Temps not loaded!'
            self.pressures = 'Pressures not loaded!'
            traceback.print_exc()
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

        self.synth_settings = attribs["dc_supply_settings"]





    def load(self, fname, plot_raw_dat=False, plot_sync=False, load_other=False, \
             skip_mon=False, load_all_pos=False, verbose=False, skip_fpga=False):

        '''Loads the data from file with fname into DataFile object. 
           Does not perform any calibrations.  
        ''' 
        self.new_trap = False

        fname = os.path.abspath(fname)

        dat, attribs = getdata(fname)

        if plot_raw_dat:
            for n in range(20):
                plt.plot(dat[:,n], label=str(n))
            plt.legend()
            plt.show()
        
        self.fname = fname
        self.date = re.search(r"\d{8,}", fname)[0]
        #print fname

        self.time = np.int64(attribs["Time"])   # unix epoch time in ns (time.time() * 10**9)

        if self.time == 0:
            #print 'Bad time...', self.time
            self.FIX_TIME = True
        else:
            self.FIX_TIME = False

        self.fsamp = attribs["Fsamp"]
        self.nsamp = len(dat[:,0])

        self.daqmx_time = np.linspace(0,self.nsamp-1,self.nsamp) * (1.0/self.fsamp) \
                               * (10**9) + self.time

        try:
            imgrid = bool(attribs["imgrid"])
        except Exception:
            imgrid = False
            traceback.print_exc()

        # If it's not an imgrid file, process all the fpga data
        if (not imgrid) and (not skip_fpga):
            fpga_fname = fname[:-3] + '_fpga.h5'
            fpga_dat = get_fpga_data(fpga_fname, verbose=verbose, timestamp=self.time)

            try:
                encode = attribs["encode_bits"]
                if (type(encode) == str) or (type(encode) == str):
                    self.encode_bits = np.array(list(encode), dtype=int)
                else:
                    self.encode_bits = encode
            except Exception:
                self.encode_bits = []
                traceback.print_exc()

            fpga_dat = sync_and_crop_fpga_data(fpga_dat, self.time, self.nsamp, \
                                               self.encode_bits, plot_sync=plot_sync)

            # IT CAN ONLY FIX THE TIME ATTRIB IF THE PARENT SCRIPT IS EXECUTED
            # AS ROOT OR ANY SUPERUSER
            if self.FIX_TIME:
                self.time = np.int64(fpga_dat['xyz_time'][0])
                #assert self.time != 0
                #print 'fix time: 0 -> ', self.time, '\r'
                #sudo_call(fix_time, self.fname, float(self.time))

            self.sync_data = fpga_dat['sync']

            ###self.pos_data = np.transpose(dat[:, configuration.col_labels["bead_pos"]])
            self.pos_data = fpga_dat['xyz']
            if load_all_pos:
                self.pos_data_2 = fpga_dat['xy_2']
            self.pos_time = fpga_dat['xyz_time']
            self.pos_fb = fpga_dat['fb']

            self.power = fpga_dat['power']

            #print self.pos_data

            # Load quadrant and backscatter amplitudes and phases
            self.amp = fpga_dat['amp']
            self.phase = fpga_dat['phase']
            self.quad_time = fpga_dat['quad_time']

            if load_all_pos:
                # run bu.print_quadrant_indices() to see an explanation of these
                right = self.amp[0] + self.amp[1]
                left = self.amp[2] + self.amp[3]
                top = self.amp[0] + self.amp[2]
                bottom = self.amp[1] + self.amp[3]

                x2 = right - left
                y2 = top - bottom

                quad_sum = np.zeros_like(self.amp[0])
                for ind in [0,1,2,3]:
                    quad_sum += self.amp[ind]

                self.pos_data_3 = np.array([x2.astype(np.float64)/quad_sum, \
                                            y2.astype(np.float64)/quad_sum, \
                                            self.pos_data[2]])


            #self.phi_cm = np.mean(self.phase[[0, 1, 2, 3]]) 

        if not skip_mon:
            self.load_monitor_data(fname, dat, attribs)

        if load_other:
            self.load_other_data()
                



    def load_monitor_data(self, fname, dat, attribs, debug=False):

        '''Loads the data from file with fname into DataFile object. 
           Does not perform any calibrations.  
        ''' 

        #dat, attribs = getdata(fname)

        fname = os.path.abspath(fname)

        self.date = re.search(r"\d{8,}", fname)[0]
        dat = dat[configuration.adc_params["ignore_pts"]:, :]

        if debug:
            print(attribs)
            print(dat.shape)
            for i in range(dat.shape[1]):
                plt.plot(dat[:,i])
            plt.show()

        try:
            self.cant_data = np.transpose(dat[:, configuration.col_labels["stage_pos"]])
        except Exception:
            self.cant_data = []
            print("Couldn't load stage data...")
            traceback.print_exc()

        try:
            self.electrode_data = np.transpose(dat[:, configuration.col_labels["electrodes"]])
        except Exception:
            self.electrode_data = []
            print("Couldn't load electrode data...")
            traceback.print_exc()

        #freqs = np.fft.rfftfreq(self.nsamp, d=1.0/self.fsamp)
        #for ind in [0,1,2]:
        #    plt.loglog(freqs, np.abs(np.fft.rfft(self.pos_data[ind])))
        #plt.figure()
        #for ind in [3,5,1]:
        #    plt.loglog(freqs, np.abs(np.fft.rfft(self.electrode_data[ind])))
        #plt.show()

        try:
            self.temps = attribs["temps"]
            # Unpacks pressure gauge vector into dict with
            # labels for pressure gauge specified by configuration.pressure_inds    
            self.pressures = unpack_config_dict(configuration.pressure_inds, \
                                                attribs["pressures"]) 
        except Exception:
            self.temps = 'Temps not loaded!'
            self.pressures = 'Pressures not loaded!'
            print("Couldn't load environmental data...")
            traceback.print_exc()

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

        self.synth_settings = attribs["dc_supply_settings"]



    def load_other_data(self):
        dat, attribs = getdata(self.fname)
        dat = dat[configuration.adc_params["ignore_pts"]:, :]
        self.other_data = np.transpose(dat[:, configuration.col_labels["other"]])




    def load_new(self, fname, plot_raw_dat=False, skip_mon=False, \
                    verbose=False):

        '''Loads the data from file with fname into DataFile object. 
           Does not perform any calibrations.  
        ''' 
        self.new_trap = True

        fname = os.path.abspath(fname)

        dat1, dat2, dat3, dat4, dat5, attribs = getdata_new(fname)

        # if plot_raw_dat:
        #     for n in range(20):
        #         plt.plot(dat[:,n], label=str(n))
        #     plt.legend()
        #     plt.show()
        
        self.fname = fname
        self.date = re.search(r"\d{8,}", fname)[0]
        #print fname

        self.fsamp = attribs['Fsamp'] / attribs['downsamp']

        self.pos_time, self.pos_data, self.pos_data_2, self.pos_fb, self.sync_data \
                    = extract_xyz_new(dat1)
        self.quad_time, self.amp, self.phase = extract_quad_new(dat2)
        self.other_data = dat3
        self.cant_data = dat4

        self.nsamp = len(self.pos_data[0])

        self.time = self.pos_time[0]

        discharge = False
        trans_func = False
        if 'Discharge' in self.fname:
            discharge = True
            amp = np.sqrt(2) * np.std(dat5[0])
        elif 'TransFunc' in self.fname:
            trans_func = True
            amp = 0.65
        else:
            amp = 1.0

        # print(amp)

        self.electrode_settings = {}
        self.electrode_settings['driven'] = np.zeros(8)
        self.electrode_settings['amplitudes'] = np.zeros(8)
        self.electrode_settings['frequencies'] = np.zeros(8)
        self.electrode_settings['dc_settings'] = np.zeros(8)

        if len(dat5):
            tarr = np.arange(self.nsamp) * (1.0 / self.fsamp)
            dumb_tarr = np.arange(dat5.shape[1]) * (1.0 / self.fsamp)

            freqs = np.fft.rfftfreq(self.nsamp, d=1.0/self.fsamp)
            dumb_freqs = np.fft.rfftfreq(dat5.shape[1], d=1.0/self.fsamp)
            #elec_data = np.zeros((8,self.nsamp))
            elec_data = (1.0e-9) * amp * np.random.randn(8,self.nsamp)

            channels = np.copy(attribs['electrode_channel'])
            #channels.sort()


            for ind, elec_ind in enumerate(channels):
                sign = 1.0
                if ind != 0:
                    sign = -1.0

                self.electrode_settings['driven'][elec_ind] = 1.0
                self.electrode_settings['amplitudes'][elec_ind] = amp
                fft = np.fft.rfft(dat5[ind])

                if discharge:
                    max_freq = dumb_freqs[np.argmax(np.abs(fft[1:])) + 1]
                    self.electrode_settings['frequencies'][elec_ind] = max_freq
                    reconstructed = sign * amp * np.sin(2.0 * np.pi * max_freq * tarr)
                    reconstructed += (1.0e-7) * amp * np.random.randn(self.nsamp)
                    elec_data[elec_ind] = reconstructed

                elif trans_func:
                    thresh = 0.5 * np.max(np.abs(fft))
                    drive_freqs = dumb_freqs[np.abs(fft) > thresh]
                    drive_freq_inds = np.arange(len(dumb_freqs))[np.abs(fft) > thresh]
                    for freq_ind, freq in zip(drive_freq_inds, drive_freqs):
                        phase = np.angle(fft[freq_ind])
                        reconstructed = amp * np.cos(2.0 * np.pi * freq * tarr + phase)
                        elec_data[elec_ind] += reconstructed

                    # print(channels)
                    # plt.plot(tarr, elec_data[elec_ind], label=elec_ind, \
                    #             color='C'+str(ind), lw=2, ls=':')
                    # plt.plot(dumb_tarr, dat5[ind], color='C'+str(ind))
                    
                    # plt.legend()
                    # plt.show()

            # for channel in range(8):
            #     plt.plot(tarr, elec_data[channel], label=str(channel))
            # plt.show()

            # print(self.electrode_settings)
        else:
            elec_data = []

        self.electrode_data = elec_data

        # run bu.print_quadrant_indices() to see an explanation of these
        right = self.amp[0] + self.amp[1]
        left = self.amp[2] + self.amp[3]
        top = self.amp[0] + self.amp[2]
        bottom = self.amp[1] + self.amp[3]

        x2 = right - left
        y2 = top - bottom

        quad_sum = right + left


        self.pos_data_3 = np.array([x2.astype(np.float64)/quad_sum, \
                                    y2.astype(np.float64)/quad_sum, \
                                    self.phase[4]])

        self.pos_data = np.copy(self.pos_data_3)



    def calibrate_stage_position(self):
        '''calibrates voltage in cant_data and into microns. 
           Uses stage position file to put origin of coordinate 
           system at trap in x direction with cantilever centered 
           on trap in y. Looks for stage position file with same 
           path and file name as self.fname '''
        if self.cant_calibrated:
            return

        if self.new_trap:
            cal_fac = configuration.stage_cal_new
            cal_fac_z = configuration.stage_cal_new_z
        else:
            cal_fac = configuration.stage_cal
            cal_fac_z = cal_fac

            try:
                # First get everything into microns.
                for k in configuration.calibrate_stage_keys:
                    #print k
                    self.stage_settings[k] *= cal_fac
            except Exception:
                print("No 'stage_settings' attribute")
                traceback.print_exc()
            
        try:
            self.cant_data[0] *= cal_fac
            self.cant_data[1] *= cal_fac
            self.cant_data[2] *= cal_fac_z
            self.cant_calibrated = True

        except Exception:
            print("No 'cant_data' attribute to calibrate")
            traceback.print_exc()
        
        # # Now load the cantilever position file.
        # # First get the path to the position file from the file base name
        # # and get the extension from configuration.extensions["stage_position"].
        # filename, file_extension = os.path.splitext(self.fname)
        # posfname = \
        #     os.path.join(filename, configuration.extensions["stage_position"])
        # # Load position of course stage. If file cant be found 
        # try: 
        #     pos_arr = pickle.load(open(posfname, "rb"))
        # except Exception:
        #     1 + 2
        #     #print "shit is fucked"
        #     traceback.print_exc()


    def calibrate_phase(self, z_bitshift=0):
        '''Hard-coded calibration that undoes the FPGA scaling and bitshifting.
           A later version should store those bit-shifts in the original .h5 file
           and pull them out automatically.'''

        avging_fac = 1.0 / (100.0 / 2.0**7)
        cast_fac = 1.0 / 2.0**16

        bitshift_fac = 1.0 / (2.0**z_bitshift)

        newphase = []

        for det in range(5):
            newphase.append( np.array(self.phase[det]) \
                             * avging_fac * cast_fac * bitshift_fac )

        self.phase = newphase
        self.zcal = self.pos_data[2] * avging_fac * np.pi



    def get_cant_drive_ax(self, plot=True, verbose=False):
        '''Determine the index of cant_data with the largest drive voltage,
           which is either exrtacted from stage setting or determined
           from the RMS of the stage monitor.

           INPUTS: none, uses class attributes from loaded data

           OUTPUTS: drive_ind, index with largest amplitude peak in 
                                    a PSD of the stage monitor.
        '''
        indmap = {0: 'x', 1: 'y', 2: 'z'}
        driven = [0,0,0]
        
        # if len(self.stage_settings) == 0:
        #     print "No data loaded..."
        #     return 

        # for i in range(3):
        #     plt.plot(self.cant_data[i])
        # plt.show()

        for ind, key in enumerate(['x driven','y driven','z driven']):
            try:
                if self.stage_settings[key]:
                    driven[ind] = 1
            except Exception:
                if verbose:
                    traceback.print_exc()
                pass

        if np.sum(driven) > 1:
            amp = [0,0,0]
            for ind, val in enumerate(driven):
                if val: 
                    key = indmap[ind] + ' amp'
                    amp[ind] = self.stage_settings[key]
            drive_ind = np.argmax(np.abs(amp))

        if np.sum(driven) == 0: # handle case of external drive
            drive_fft = np.fft.rfft(self.cant_data)
            mean_sq = np.sum(np.abs(drive_fft[:, 1:])**2, axis = 1)#cut DC
            drive_ind = np.argmax(mean_sq)

        return drive_ind


    def generate_yuk_template(self, yukfuncs_at_lambda, p0, stage_travel = [80., 80., 80.],\
                                plt_pvec = False, plt_template = False, cf = 1E-6):
        '''generates a template for the expected force in the time domain. takes a tuple
            of interpolating funcs for the force at a particular lambda and a vector, p0,
            that gives the displacement '''

        self.calibrate_stage_position()
        pvec = np.zeros_like(self.cant_data)
        pvec[0, :] = stage_travel[0] - self.cant_data[0, :] + p0[0]
        pvec[1, :] = self.cant_data[1, :] - stage_travel[1]/2. + p0[1]
        pvec[2, :] = self.cant_data[2, :] - p0[2]
        pvec *= cf

        pts = np.stack(pvec, axis = -1)
        
        fx = yukfuncs_at_lambda[0](pts)
        fy = yukfuncs_at_lambda[1](pts)
        fz = yukfuncs_at_lambda[2](pts)

        if plt_pvec:
            plt.plot(pts[:, 0], label = "X")
            plt.plot(pts[:, 1], label = "Y")
            plt.plot(pts[:, 2], label = "Z")
            plt.legend()
            plt.show()

        if plt_template:
            plt.plot(fx, label = "fx")
            plt.plot(fy, label = "fy")
            plt.plot(fz, label = "fz")
            plt.legend()
            plt.show()

        return np.array([fx, fy, fz])


    def inject_fake_signal(self, yukfuncs_at_lambda, p0, fake_alpha = 1E10, make_plot = False):
        '''injects a fake signal into the data at a particular alpha and lambda. 
            needs p0 = [min separation at 80um extent, y difference between center of cantilever 
            and bead when cantilever is set to 40 um, bead height in nanopositiong stage coordinates].'''
        
        self.calibrate_stage_position()
        try:
            cf = self.conv_facs
        except AttributeError:
            self.diagonalize()
            cf = self.conv_facs
    
        fs = fake_alpha*self.generate_yuk_template(yukfuncs_at_lambda, p0, \
                plt_pvec = make_plot, plt_template =make_plot)

        #add fake signal to position data using 1/cf to calibrate out of force units
        self.pos_data = self.pos_data.astype(float) #cast to floats for math
        self.pos_data[0] += fs[0]/cf[0]
        self.pos_data[1] += fs[1]/cf[1]
        self.pos_data[2] += fs[2]/cf[2]

        #no calibration required for diag data    
        self.diag_pos_data[0] += fs[0]
        self.diag_pos_data[1] += fs[1]
        self.diag_pos_data[2] += fs[2]
        


    def build_drive_filt(self, drive_fft, freqs, nharmonics=10, width=0, harms=[], \
                         maxfreq=2500):
        '''Identify the fundamental drive frequency and make a notch filter
           with the number of harmonics requested.

           INPUTS: drive_fft, fft of cantilever drive
                   freqs, array of frequencies associated to data ffts
                   harms, number of harmonics to included
                   width, width of the notch filter in Hz

           OUTPUTS: none, generates new class attribute.'''

        # Find the drive frequency, ignoring the DC bin
        maxind = np.argmin( np.abs(freqs - maxfreq) )

        fund_ind = np.argmax( np.abs(drive_fft[1:maxind]) ) + 1
        drive_freq = freqs[fund_ind]

        drivefilt = np.zeros(len(drive_fft))
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
    




    def get_boolean_cantfilt(self, ext_cant=(False,1), ext_cant_drive=False, ext_cant_ind=1, \
                             nharmonics=10, harms=[], width=0, maxfreq=2500, \
                             drive_harms=5):
        '''Builds a boolean notch filter for the cantilever drive

           INPUTS: ext_cant, tuple with bool specifying if an external drive
                             was used for the cantilever, and the axis of that
                             external drive (so we know which mon signal)

           OUTPUTS: ginds, bool array of length NFFT (set by hdf5 file).'''

        drive_ind = self.get_cant_drive_ax()
        if ext_cant[0]:
            drive_ind = ext_cant[1]

        drivevec = self.cant_data[drive_ind]
        drivefft = np.fft.rfft(drivevec)

        freqs = np.fft.rfftfreq(len(drivevec), d=1.0/self.fsamp)

        drivefilt, fund_ind, drive_freq = \
                    self.build_drive_filt(drivefft, freqs, nharmonics=nharmonics, \
                                          harms=harms, width=width, maxfreq=maxfreq)

        all_inds = np.arange(len(freqs)).astype(np.int)
        drive_ginds = []
        for i in range(drive_harms):
            if width != 0:
                drive_ginds += list( all_inds[np.abs(freqs - (i+1)*drive_freq) < 0.5*width] )
            else:
                drive_ginds.append( np.argmin(np.abs(freqs - (i+1)*drive_freq)) )

        # Apply filter by indexing with a boolean array
        bool_ginds = drivefilt > 0
        ginds = np.arange(len(drivefilt)).astype(np.int)[bool_ginds]

        outdic = {'ginds': ginds, 'fund_ind': fund_ind, 'drive_freq': drive_freq, \
                  'drive_ind': drive_ind, 'drive_ginds': drive_ginds}

        return outdic





    def get_boolean_elecfilt(self, elec_ind, nharmonics=10, harms=[], width=0, \
                             maxfreq=2500):
        '''Builds a boolean notch filter for the cantilever drive

           INPUTS: ext_cant, tuple with bool specifying if an external drive
                             was used for the cantilever, and the axis of that
                             external drive (so we know which mon signal)

           OUTPUTS: ginds, bool array of length NFFT (set by hdf5 file).'''

        drivevec = self.electrode_data[elec_ind]
        drivefft = np.fft.rfft(drivevec)

        freqs = np.fft.rfftfreq(len(drivevec), d=1.0/self.fsamp)

        drivefilt, fund_ind, drive_freq = \
                    self.build_drive_filt(drivefft, freqs, nharmonics=nharmonics, \
                                          harms=harms, width=width, maxfreq=maxfreq)

        all_inds = np.arange(len(freqs)).astype(np.int)
        drive_ginds = []
        for i in range(drive_harms):
            if width != 0:
                drive_ginds += list( all_inds[np.abs(freqs - (i+1)*drive_freq) < 0.5*width] )
            else:
                drive_ginds.append( np.argmin(np.abs(freqs - (i+1)*drive_freq)) )

        # Apply filter by indexing with a boolean array
        bool_ginds = drivefilt > 0
        ginds = np.arange(len(drivefilt)).astype(np.int)[bool_ginds]

        outdic = {'ginds': ginds, 'fund_ind': fund_ind, 'drive_freq': drive_freq, \
                  'drive_ind': elec_ind, 'drive_ginds': drive_ginds}

        return outdic





    def get_datffts_and_errs(self, ginds, drive_freq, drive_ginds, noisebins=10, \
                             plot=False, diag=True, drive_ind=1, elec_drive=False, \
                             elec_ind=0, noiselim=(10,100)):   
        '''Applies a cantilever notch filter and returns the filtered data
           with an error estimate based on the neighboring bins of the PSD.

           INPUTS: ginds, boolean cantilever drive notch filter
                   drive_freq, cantilever drive freq

           OUTPUTS: datffts, ffts evalutated at ginds
                    diagdatffts, diagffts evaluated at ginds
                    daterrs, errors from surrounding bins
                    diagdaterrs, diag errors from surrounding bins
        ''' 


        freqs = np.fft.rfftfreq(len(self.pos_data[0]), d=1.0/self.fsamp)
        fund_ind = np.argmin(np.abs(freqs - drive_freq))
        bin_sp = freqs[1] - freqs[0]

        harm_freqs = freqs[ginds]

        if type(harm_freqs) == np.float64:
            harm_freqs = np.array([harm_freqs])
            just_one = True
        else:
            just_one = False

        datffts = np.zeros((3, len(ginds)), dtype=np.complex128)
        daterrs = np.zeros((3, len(ginds)*noisebins), dtype=np.complex128)
        if diag:
            diagdatffts = np.zeros((3, len(ginds)), dtype=np.complex128)
            diagdaterrs = np.zeros((3, len(ginds)*noisebins), dtype=np.complex128)

        noiseffts = np.zeros((3, len(ginds)), dtype=np.complex128)
        noise_inds = np.arange(len(freqs))[(freqs <= noiselim[1]) * (freqs >= noiselim[0])]
        #for ind in ginds:
        #    noise_inds = np.delete(noise_inds, ind)

        if elec_drive:
            drivefft_full = np.fft.rfft(self.electrode_data[elec_ind])
            meandrive = np.mean(self.electrode_data[elec_ind])
        else:
            drivefft_full = np.fft.rfft(self.cant_data[drive_ind])
            meandrive = np.mean(self.cant_data[drive_ind])
        driveffts = drivefft_full[ginds]
        driveffts_all = drivefft_full[drive_ginds]

        for resp in [0,1,2]:

            N = len(self.pos_data[resp])

            datfft = np.fft.rfft(self.pos_data[resp]*self.conv_facs[resp])
            # plt.loglog(freqs, np.abs(datfft)*fft_norm(self.nsamp, self.fsamp))
            # plt.show()
            # input()
            datffts[resp] += datfft[ginds]

            noise_dat = datfft[noise_inds]
            
            #noiseffts[resp] = np.mean(np.abs(noise_dat)) * \
            #                  np.exp(1.0j * np.mean(np.angle(noise_dat)))

            noiseffts[resp] = np.mean(noise_dat)

            ### OKAY UP TO HERE

            if diag:
                diagdatfft = np.fft.rfft(self.diag_pos_data[resp])
                diagdatffts[resp] += diagdatfft[ginds]

            err_ginds = []
            for freqind, freq in enumerate(harm_freqs):
                harm_ind = np.argmin(np.abs(freqs-freq))
                neg = False
                pos_ind = harm_ind + 2
                neg_ind = harm_ind - 2
                for i in range(noisebins):
                    if not neg:
                        neg = True
                        err_ginds.append(pos_ind)
                        pos_ind += 1
                    else:
                        neg = False
                        err_ginds.append(neg_ind)
                        neg_ind -= 1
            err_ginds.sort()

            daterrs[resp] += datfft[err_ginds]
            if diag:
                diagdaterrs[resp] += diagdatfft[err_ginds]

            if plot:
                normfac = np.sqrt(2.0 * bin_sp) * fft_norm(N, self.fsamp)

                #normfac = fft_norm(N, self.fsamp)
                
                avg_inds = np.arange(len(freqs))[(freqs > 10.0) * (freqs < 100.0)]
                new_avg_inds = []
                for avg_ind in avg_inds:
                    if avg_ind not in ginds:
                        new_avg_inds.append(avg_ind)

                print(np.mean(np.abs(datfft[avg_inds])*normfac))

                fig, axarr = plt.subplots(2,1,sharex=True,sharey=True,figsize=(10,8))
                axarr[0].loglog(freqs, np.abs(datfft)*normfac, alpha=0.4)
                axarr[1].loglog(freqs, np.abs(diagdatfft)*normfac, alpha=0.4)
                for gind in ginds:
                    b = []
                    for err_gind in err_ginds:
                        if np.abs(err_gind - gind) < noisebins:
                            b.append(err_gind)
                    axarr[0].loglog(freqs[b], np.abs(datfft[b])*normfac, \
                                color='C1')
                    axarr[1].loglog(freqs[b], np.abs(diagdatfft[b])*normfac, \
                                color='C1')
                axarr[0].loglog(freqs[ginds], np.abs(datfft[ginds])*normfac, \
                                '.', ms=10, color='C2')
                axarr[1].loglog(freqs[ginds], np.abs(diagdatfft[ginds])*normfac, \
                                '.', ms=10, color='C2')
                axarr[0].set_ylabel('Force [N]')
                axarr[1].set_ylabel('Diag Force [N]')
                axarr[1].set_xlabel('Frequency [Hz]')
                fig.tight_layout()

                fig2, axarr2 = plt.subplots(2,1,sharex=True,sharey=True,figsize=(10,8))

                axarr2[0].plot(freqs[ginds], datffts[resp].real * normfac, '.', \
                         label='real', ms=20)
                axarr2[0].plot(freqs[ginds], datffts[resp].imag * normfac, '.', \
                         label='imag', ms=20)
                axarr2[0].plot(freqs[err_ginds], daterrs[resp].real * normfac, '.', \
                         label='errs real', ms=5, alpha=0.6)
                axarr2[0].plot(freqs[err_ginds], daterrs[resp].imag * normfac, '.', \
                         label='errs imag', ms=5, alpha=0.6)

                axarr2[1].plot(freqs[ginds], diagdatffts[resp].real * normfac, '.', \
                         label='real', ms=20)
                axarr2[1].plot(freqs[ginds], diagdatffts[resp].imag * normfac, '.', \
                         label='imag', ms=20)
                axarr2[1].plot(freqs[err_ginds], diagdaterrs[resp].real * normfac, '.', \
                         label='errs real', ms=5, alpha=0.6)
                axarr2[1].plot(freqs[err_ginds], diagdaterrs[resp].imag * normfac, '.', \
                         label='errs imag', ms=5, alpha=0.6)

                axarr2[0].set_ylabel('Force [N]')
                axarr2[1].set_ylabel('Diag Force [N]')
                axarr2[1].set_xlabel('Frequency [Hz]')
                axarr2[0].legend(fontsize=10)
                fig2.tight_layout()

                plt.figure()
                plt.loglog(freqs, np.abs(drivefft_full)*normfac)
                plt.ylabel('Drive Amplitude [um or V]')
                plt.xlabel('Frequency [Hz]')
                plt.tight_layout()

                plt.show()

        normfac = np.sqrt(2.0 * bin_sp) * fft_norm(N, self.fsamp)

        datffts *= normfac
        driveffts *= normfac
        driveffts_all *= normfac
        daterrs *= normfac
        noiseffts *= normfac
        if diag:
            diagdatffts *= normfac
            diagdaterrs *= normfac

        if not diag:
            diagdatffts = np.zeros_like(datffts)
            diagdaterrs = np.zeros_like(daterrs)

        outdic = {'datffts': datffts, 'diagdatffts': diagdatffts, \
                  'daterrs': daterrs, 'diagdaterrs': diagdaterrs, \
                  'driveffts': driveffts, 'driveffts_all': driveffts_all, \
                  'noiseffts': noiseffts, 'meandrive': meandrive, \
                  'err_ginds': err_ginds}

        return outdic


    def detrend_poly(self, order=1, plot=False):
        '''Remove a polynomial of arbitrary order from data.

           INPUTS: order, order of the polynomial to subtract
                   plot, boolean whether to plot detrending result

           OUTPUTS: none, generates new class attribute.'''

        for resp in [0,1,2]:
            xarr = np.arange(len(self.pos_data[resp]))
            fit_model = np.polyfit(xarr, self.pos_data[resp], order)
            fit_eval = np.polyval(fit_model, xarr)

            if plot:
                fig, axarr = plt.subplots(2,1,sharex=True, \
                                gridspec_kw={'height_ratios': [1,1]})
                axarr[0].plot(xarr, self.pos_data[resp], color='k')
                axarr[0].plot(xarr, fit_eval, lw=2, color='r')
                axarr[1].plot(xarr, self.pos_data[resp] - fit_eval, color='k')
                fig.tight_layout()
                plt.show()

            self.pos_data[resp] = np.float64(self.pos_data[resp]) - fit_eval


        if len(self.other_data):
            ndim = self.other_data.ndim

            if ndim == 1:
                self.other_data = [self.other_data]

            for ax in range(ndim):

                dat = np.float64(self.other_data[ax])
                xarr = np.arange(len(dat))  

                fit_model = np.polyfit(xarr, dat, order)
                fit_eval = np.polyval(fit_model, xarr)

                diff = dat - fit_eval

                if plot:
                    fig, axarr = plt.subplots(2,1,sharex=True, \
                                    gridspec_kw={'height_ratios': [1,1]})
                    axarr[0].plot(xarr, dat, color='k')
                    axarr[0].plot(xarr, fit_eval, lw=2, color='r')
                    axarr[1].plot(xarr, diff, color='k')
                    fig.tight_layout()
                    plt.show()


                self.other_data[ax] = diff


    def high_pass_filter(self, order=1, fc=1.0):
        '''Apply a digital butterworth type filter to the data

           INPUTS: order, order of the butterworth filter
                   fc, cutoff frequency in Hertz

           OUTPUTS: none, generates new class attribute.'''

        Wn = 2.0 * fc / self.fsamp
        b, a = signal.butter(order, Wn, btype='highpass')

        for resp in [0,1,2]:
            self.pos_data[resp] = signal.filtfilt(b, a, self.pos_data[resp])


    def diagonalize(self, date='', interpolate=False, maxfreq=1000, \
                    step_cal_drive_freq=41.0, plot=False):
        '''Diagonalizes data, adding a new attribute to the DataFile object.

           INPUTS: date, date in form YYYYMMDD if you don't want to use
                         default TF from file date
                   interpolate, boolean specifying whether to use an 
                                interpolating function or fit to damped HO
                   maxfreq, max frequency above which data is top-hat filtered

           OUTPUTS: none, generates new class attribute.'''

        if self.new_trap:
            tf_path = '/data/new_trap_processed/calibrations/transfer_funcs/'
        else:
            tf_path = '/data/old_trap_processed/calibrations/transfer_funcs/'

        ext = configuration.extensions['trans_fun']
        if not len(date):
            tf_path +=  self.date
        else:
            tf_path += date

        if interpolate:
            tf_path += '_interp' + ext
        else:
            tf_path += ext

        #print tf_path

        # Load the transfer function. Note that this Hfunc maps
        # drive -> response, so we will need to invert
        try:
            Hfunc = pickle.load(open(tf_path, 'rb'))
        except Exception:
            print("Couldn't automatically find correct TF")
            traceback.print_exc()
            return

        # Generate FFT frequencies for given data
        N = len(self.pos_data[0])
        freqs = np.fft.rfftfreq(N, d=1.0/self.fsamp)

        # Compute TF at frequencies of interest. Appropriately inverts
        # so we can map response -> drive
        Harr = tf.make_tf_array(freqs, Hfunc)

        x_tf_res_freq = freqs[np.argmax(np.abs(Hfunc(0,0,freqs)))]
        y_tf_res_freq = freqs[np.argmax(np.abs(Hfunc(1,1,freqs)))]
        self.xy_tf_res_freqs = [x_tf_res_freq, y_tf_res_freq]

        if plot:
            tf.plot_tf_array(freqs, Harr)

        maxfreq_ind = np.argmin( np.abs(freqs - maxfreq) )
        Harr[maxfreq_ind+1:,:,:] = 0.0+0.0j

        f_ind = np.argmin( np.abs(freqs - step_cal_drive_freq) )
        mat = Harr[f_ind,:,:]
        conv_facs = [0, 0, 0]
        for i in [0,1,2]:
            conv_facs[i] = np.abs(mat[i,i])
        self.conv_facs = conv_facs

        # Compute the FFT, apply the TF and inverse FFT
        if self.new_trap:
            data = self.pos_data_3
        else:
            data = self.pos_data
        data_fft = np.fft.rfft(data)
        diag_fft = np.einsum('ikj,ki->ji', Harr, data_fft)


        if plot:
            norm = fft_norm(N, self.fsamp)
            fig, axarr = plt.subplots(3,1,sharex=True,sharey=True)
            for ax in [0,1,2]:
                axarr[ax].loglog(freqs, norm*np.abs(data_fft[ax])*conv_facs[ax])
                axarr[ax].loglog(freqs, norm*np.abs(diag_fft[ax]))
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



    def get_xy_resonance(self, elec_ind=0):
        freqs = np.fft.rfftfreq(self.nsamp, d=1.0/self.fsamp)
        freq_band = (freqs > 200) * (freqs < 600)
        
        upperx = freqs > self.xy_tf_res_freqs[0]
        lowerx = freqs < self.xy_tf_res_freqs[0]
        
        uppery = freqs > self.xy_tf_res_freqs[1]
        lowery = freqs < self.xy_tf_res_freqs[1]

        upfreq_ind = np.argmax(self.electrode_data[elec_ind] * freq_band * upperx)
        lowfreq_ind = np.argmax(self.electrode_data[elec_ind] * freq_band * lowerx)




    def get_force_v_pos(self, nbins=100, nharmonics=10, harms=[], width=0, \
                        sg_filter=False, sg_params=[3,1], verbose=True, \
                        cantilever_drive=True, elec_drive=False, \
                        fakedrive=False, fakefreq=50, fakeamp=80, fakephi=0, \
                        maxfreq=2500):
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

        if elec_drive:
            elec_ind = np.argmax(self.electrode_settings['driven'])
            drivevec = self.electrode_data[elec_ind]

        # Bin responses against the drive. If data has been diagonalized,
        # it bins the diagonal data as well
        dt = 1. / self.fsamp
        binned_data = [[0,0,0], [0,0,0], [0,0,0]]
        if len(self.diag_pos_data):
            diag_binned_data = [[0,0,0], [0,0,0], [0,0,0]]
        for resp in [0,1,2]:
            bins, binned_vec, binned_err = \
                            spatial_bin(drivevec, self.pos_data[resp], dt, \
                                        nbins = nbins, nharmonics = nharmonics, \
                                        harms = harms, width = width, \
                                        sg_filter = sg_filter, sg_params = sg_params, \
                                        verbose = verbose, maxfreq = maxfreq)
            binned_data[resp][0] = bins
            binned_data[resp][1] = binned_vec
            binned_data[resp][2] = binned_err
            
            if len(self.diag_pos_data):
                diag_bins, diag_binned_vec, diag_binned_err = \
                            spatial_bin(drivevec, self.diag_pos_data[resp], dt, \
                                        nbins = nbins, nharmonics = nharmonics, \
                                        harms = harms, width = width, \
                                        sg_filter = sg_filter, sg_params = sg_params, \
                                        verbose = verbose, maxfreq = maxfreq)

                diag_binned_data[resp][0] = diag_bins
                diag_binned_data[resp][1] = diag_binned_vec
                diag_binned_data[resp][2] = diag_binned_err

        self.binned_data = binned_data
        if len(self.diag_pos_data):
            self.diag_binned_data = diag_binned_data
        else:
            self.diag_binned_Data = ''
                

