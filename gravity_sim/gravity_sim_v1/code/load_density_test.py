#!/usr/bin/python

import numpy as np
import pickle as pickle
import scipy.interpolate as interpolate
import scipy, sys, time


rhopath = '/farmshare/user_data/cblakemo/gravity_sim/test_masses/attractor_v2/rho_arr.p'
rho, xx, yy, zz = pickle.load(open(rhopath, 'rb'))
print("Density Loaded.")
sys.stdout.flush()

print(np.mean(rho))