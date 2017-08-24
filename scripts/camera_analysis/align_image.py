import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
import scipy
import glob
from scipy.optimize import curve_fit
    

data_dir1 = "/data/20170822/image_calibration/align_profiles_x_3"
data_dir2 = "/data/20170822/image_calibration/align_profiles_y"
out_dir = "/calibrations/20170814"


def get_stage_column(attribs, stage_cols = [17, 18, 19], attrib_inds = [3, 6, 9]):
    '''gets the first driven stage axis from data attribs'''
    stage_settings = attribs['stage_settings']
    driven = np.array(map(bool, stage_settings[attrib_inds]))
    return (np.array(stage_cols)[driven])[0]

def gauss_beam(r, mu, w, A):
    '''gaussian beam function for fitting'''
    return np.exp(-2.*(r-mu)**2/w**2)

def profile(fname, ends = 100, stage_cal = 8., h_ax = 19, data_column = 5):
    dat, attribs, f = bu.getdata(fname)
    dat = dat[ends:-ends, :]
    stage_column = get_stage_column(attribs)
    dat[:,stage_column]*=stage_cal
    f.close()


def save_cal(p_arr, path):
    #Makes path if it does not exist and saves parr to path/stage_position.npy
    if not os.path.exists(path):
        os.makedirs(path)
    outfile = os.path.join(path, 'stage_position')
    np.save(outfile, p_arr)


