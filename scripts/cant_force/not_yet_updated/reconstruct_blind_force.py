import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import Tkinter
import tkFileDialog
import os, sys
from scipy.optimize import curve_fit
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize 

dirs = [66,67,68,69]

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )
maxfiles=5

def proc_dir(d):
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.diag_loader, maxfiles=maxfiles)

    dir_obj.load_H("./trans_funcs/Hout_20160630.p")
    dir_obj.load_step_cal('./calibrations/step_cal_20160701.p')
    
    dir_obj.thermal_cal_file_path = '/data/20160627/bead1/1_5mbar_nocool_withap.h5'
    dir_obj.thermal_calibration()
    #dir_obj.thermal_cal_fobj.plt_thermal_fit()

    #dir_obj.build_Hfuncs(fpeaks=[245, 255, 50], weight_peak=False, weight_above_thresh=True,\
    #                     plot_fits=False, weight_phase=True)

    dir_obj.diagonalize_files(simpleDCmat=False)
    dir_obj.get_conv_facs()
    #dir_obj.plot_H(cal=True)
    
    return dir_obj


dir_objs = map(proc_dir, dirs)



for obj in dir_objs:
    freqs = obj.fobjs[0].fft_freqs
    avg_fft = np.zeros(obj.fobjs[0].data_fft.shape, dtype=np.complex128)
    avg_diag_fft = np.zeros(obj.fobjs[0].diag_data_fft.shape, dtype=np.complex128)
    fft0 = np.fft.rfft(obj.fobjs[0].electrode_data)
    avg_drive_fft = np.zeros(fft0.shape, dtype=np.complex128)

    count = 0.
    for fobj in obj.fobjs:
        avg_fft += fobj.data_fft
        avg_diag_fft += fobj.diag_data_fft
        drive_fft = np.fft.rfft(fobj.electrode_data)
        avg_drive_fft += drive_fft
        count += 1.

    avg_fft = avg_fft / count
    avg_diag_fft = avg_diag_fft / count
    avg_drive_fft = avg_drive_fft / count
    
    plt.figure(1)
    for i in [0,1,2]:
        plt.subplot(3,1,i+1)
        plt.loglog(freqs, obj.conv_facs[i] * np.abs(avg_fft[i]))

        if i == 0:
            j = 5
        elif i == 1:
            j = 3
        elif i == 2:
            j = 1

        plt.loglog(freqs, np.abs(avg_diag_fft[i]))
        plt.loglog(freqs, obj.conv_facs[i] * np.abs(avg_drive_fft[j]))
    plt.show()



