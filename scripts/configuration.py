#################################################################################Configuration file specifying all of the parameters used in the analysis
###############################################################################

####Physical parameters in si units
p_param = {"bead_radius": 2.43e-6,
                "bead_rho": 2.2e3,
                "kb": 1.3806488e-23,
                "e_charge": 1.60217662e-19,
                "nucleon_mass": 1.6749e-27}


####Data column Labels
col_labels = {"bead_pos"  : [0, 1, 2],
                   "electrodes": [8, 9, 10, 11, 12, 13, 14],
                    "stage_pos" : [16, 17, 18]   
} 

####electrode column labels in  label: number pairs. cantilever approached from back. Left and right determined looking from side with bead dropper
elec_labels = {"cantilever":0, 
                    "top":1, 
                    "bottom":2, 
                    "front":3,
                    "back":4,
                    "right":5,
                    "left":6}

####ADC parameters to convert from bits to voltage when loading raw .h5 files
adc_params = {"adc_res":2**16,
              "adc_max_voltage":10.,
              "ignore_pts": 1000}


####File extensions used at point throughout the analysis
extensions = {"data": ".h5",
              "image": ".npy",
              "stage_position": ".pos",
              "trans_fun": ".trans",
              } 

####Order of pressures coming from labview
pressures = {"pirani":0,
             "cold_cathode":1,
             "baratron":2} 
