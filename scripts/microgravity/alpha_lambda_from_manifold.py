import dill as pickle

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interp

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config




data_manifold_path = '/force_v_pos/20170903_force_v_pos_dic.p'

theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'







#########################################################

data_manifold = pickle.load( open(data_manifold_path, 'rb') )

Gdata = np.load(theory_data_dir + 'Gravdata.npy')
yukdata = np.load(theory_data_dir + 'yukdata.npy')
lambdas = np.load(theory_data_dir + 'lambdas.npy')
xpos = np.load(theory_data_dir + 'xpos.npy')
ypos = np.load(theory_data_dir + 'ypos.npy')
zpos = np.load(theory_data_dir + 'zpos.npy')

yuk_fx_func = np

def yuk_fx(x, y, z, alpha, yuklambda):
    
