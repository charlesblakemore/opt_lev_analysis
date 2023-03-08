import h5py, os, sys, re

import numpy as np

import matplotlib.pyplot as plt
import scipy

from bead_util_funcs import *
from bead_data_funcs import *


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

        



    def load(self, fname, plot_raw_dat=False, plot_sync=False, load_other=False, \
             skip_mon=False, load_all_pos=False, verbose=False, skip_fpga=False):

        '''Loads the data from file with fname into DataFile object. 
           Does not perform any calibrations.  
        ''' 

        fname = os.path.abspath(fname)

        dat, attribs = getdata(fname)

        if plot_raw_dat:
            for n in range(dat.shape[1]):
                plt.plot(dat[:,n], label=str(n))
            plt.legend()
            plt.show()
            input()

        self.fname = fname
        self.date = re.search(r"\d{8,}", fname)[0]
        #print fname

        # unix epoch time in ns (time.time() * 10**9)
        try:
            self.time = attribs["time"]
        except:
            self.time = attribs["Time"]


        if self.time == 0:
            self.FIX_TIME = True
        else:
            self.FIX_TIME = False


        try:
            self.fsamp = attribs["fsamp"]
        except:
            self.fsamp = attribs["Fsamp"]

            
        self.nsamp = len(dat[:,0])

        self.daqmx_time = np.linspace(0,self.nsamp-1,self.nsamp) * (1.0/self.fsamp) \
                               * (10**9) + self.time


        ### Data prior to this data didn't have the companion fpga data file
        ### and the xyz data was transposed.
        if int(self.date) < 20180601:
            skip_fpga = True
            skip_mon = False

            self.pos_data = np.transpose(dat[:, [0,1,2]])



        # If it's not an imgrid file, process all the fpga data
        if not skip_fpga:
            fpga_fname = fname[:-3] + '_fpga.h5'

            ### This date isn't final yet. Haven't looked over all the relevant data 
            ### folders to see exactly when we added the feedback signals into the
            ### output FIFOs, but it was relatively soon after the initial implementation
            ### of the FPGA data readout on 20180601
            if int(self.date) < 20180701:
                fpga_dat = get_fpga_data_2018(fpga_fname, \
                                              verbose=verbose, \
                                              timestamp=self.time)
            else:
                fpga_dat = get_fpga_data(fpga_fname, \
                                         verbose=verbose, \
                                         timestamp=self.time)

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


            if self.FIX_TIME:
                self.time = np.int64(fpga_dat['xyz_time'][0])

            self.sync_data = fpga_dat['sync']

            self.pos_data = fpga_dat['xyz']
            if load_all_pos:
                self.pos_data_2 = fpga_dat['xy_2']
            self.pos_time = fpga_dat['xyz_time']
            self.pos_fb = fpga_dat['fb']

            self.power = fpga_dat['power']
            self.power_fb = fpga_dat['power_fb']

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





