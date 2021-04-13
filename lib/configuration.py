#################################################################################
# Configuration file specifying all of the parameters used in the analysis. 
# Column labels for unpacking data attributes passed as lists are stores in 
# dictionaries with the label and the column(s) as the value(s)
###############################################################################

####Data column Labels
col_labels = {"other"     : [0, 1, 2, 3, 4, 5, 6, 7],
              "electrodes": [8, 9, 10, 11, 12, 13, 14, 15],
              "stage_pos" : [1, 2, 3] #[17, 18, 19] 
} 

####electrode column labels in  label: number pairs. cantilever approached from back. Left and right determined looking from side with bead dropper

num_electrodes = 7
elec_labels = {"cantilever":0, 
               "top":1, 
               "bottom":2, 
               "front":3,
               "back":4,
               "right":5,
               "left":6}

# Easy mapping from elec index to data column
elec_map = {1: 2,
            2: 2,
            3: 0,
            4: 0,
            5: 1,
            6: 1,}

electrodes = {0: "cantilever", 
              1: "top", 
              2: "bottom", 
              3: "front",
              4: "back",
              5: "right",
              6: "left"}

electrode_settings = {'driven'     : [0, 1, 2, 3, 4, 5, 6],
                     'amplitudes'  : [8, 9, 10, 11, 12, 13, 14],
                     'frequencies' : [16, 17, 18, 19, 20, 21, 22],
                     'dc_vals2'    : [24, 25, 26, 27, 28, 29, 30]}  

####ADC parameters to convert from bits to voltage when loading raw .h5 files
adc_params = {"adc_res":2**16,
              "adc_max_voltage":10.,
              "ignore_pts": 0}


####File extensions used at point throughout the analysis
extensions = {"data": ".h5",
              "image": ".npy",
              "stage_position": ".pos",
              "trans_fun": ".trans",
              } 

####Order of pressures coming from labview
pressure_inds = {"pirani":0,
             "cold_cathode":1,
             "baratron":2}

####Stage calibration and settings information
stage_inds = {"x DC": 0,
              "x driven": 3,
              "x amp": 4,
              "x freq": 5, 
              "y DC": 1,
              "y driven": 6,
              "y amp": 7,
              "y freq": 8,
              "z DC": 2,
              "z driven": 9,
              "z amp": 10,
              "z freq":11} 

# Taken straight from newport stage datasheet
stage_cal = 8.0 # um/V
stage_cal_func = lambda x: 8.0 * x
# From calibration taken by AFieguth, sometime late 2019
# volt_mon = [-0.007, 4.968, 9.91], pos_setting = [0, 250, 500]
stage_cal_new = 50.418 # um/V, with ~0.077 um offset
stage_cal_new_z = 10.0
stage_cal_new_func = lambda x: 50.418*x + 0.0766
stage_cal_new_z_func = lambda x: 10.0*x


#Stage keys that get calibrated by the stage_cal:
calibrate_stage_keys = ["x DC", "x amp", "y DC", "y amp", "z DC","z amp"] 
