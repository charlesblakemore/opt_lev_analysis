import os, fnmatch, sys, time

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import calib_util as cu
import configuration as config

import time


dirname = '/data/20180609/sync_test/basic_sync'

elec_ind = 3
pos_ind = 0  # {0: x, 1: y, 2: z}


files = bu.find_all_fnames(dirname)
files = bu.sort_files_by_timestamp(files)



for filname in files:
    df = bu.DataFile()
    df.load(filname)
