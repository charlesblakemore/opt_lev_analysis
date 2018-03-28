import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import background_util as bgu
import configuration as config

import warnings
warnings.filterwarnings("ignore")

cbead = '/data/20180314/bead1'

#dir1 = cbead + '/grav_data/ydrive_1sep_1height_1V-1300Hz_shieldin_1V-cant'
#dir1 = cbead + '/grav_data/ydrive_1sep_1height_2V-2200Hz_shield_0mV-cant'

dir1 = cbead + '/grav_data/ydrive_6sep_1height_shield-2Vac-2200Hz_cant-0mV'


#dir1 = '/data/20180314/bead1/grav_data/xdrive_3height_5Vac-1198Hz'
#dir1 = '/data/20180314/bead1/grav_data/xdrive_1height_nofield_shieldin'

#dir1 = '/data/20180308/bead2/grav_data/onepos_long'

unwrap = True

ext_cant_drive = True
ext_cant_ind = 1

harms_to_track = [1,2,3]
#harms_to_track = [1,2,3,4,5,6,7,8,9,10]

harms_to_label = [1,2,3]

sub_cant_phase = True
plot_first_drive = False

ax0val = 80   # um
ax1val = None   # um
ax2val = None   # um

#ylim = (1e-21, 1e-14)
#ylim = (1e-7, 1e-1)
ylim = ()
arrow_fac = 5

lpf = 2500   # Hz

file_inds = (0, 100)

diag = False


###########################################################




allfiles = bu.find_all_fnames(dir1)

sep0background = bgu.Background(allfiles)
sep0background.select_by_position(ax0val=ax0val)
sep0background.analyze_background(file_inds=file_inds, ext_cant_drive=ext_cant_drive, \
                                  ext_cant_ind=ext_cant_ind)
sep0background.plot_background()


#freqs, harm_freqs, avg_asd, amps, phases, amp_errs, phase_errs, tint = \
#        analyze_background(allfiles, file_inds=file_inds, diag=diag, \
#                           data_axes=[0,1,2], other_axes=[], \
#                           unwrap=unwrap, harms_to_track=harms_to_track)

#plot_background(freqs, harm_freqs, avg_asd, amps, phases, amp_errs, phase_errs, tint, \
#                harms_to_label=harms_to_label, harms_to_plot=harms_to_track)
