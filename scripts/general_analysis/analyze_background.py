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

cbead = '/data/20180404/bead2'

#dir1 = cbead + '/grav_data/ydrive_1sep_1height_shield-0V_cant-0mV_NOBEAD'
#dir1 = cbead + '/grav_data/ydrive_1sep_1height_shield-0V_cant-0mV_minimalfb'

#dir1 = cbead + '/grav_data/tumbling/ydrive_1sep_1height_cant0mV_noshield'
dir1 = cbead + '/grav_data/ydrive_1sep_1height_cant-3Vac-1750Hz_noshield'
dir1 = cbead + '/grav_data/ydrive_1sep_1height_cant-3Vac-1750Hz_noshield_farback'
dir1 = cbead + '/grav_data/shield_out/ydrive_1sep_1height_cant-0mV-farback_shield-3Vac-1750Hz-initpos'

load = False #True
save = False

unwrap = True

ext_cant_drive = True
ext_cant_ind = 1

#harms_to_track = [1]
harms_to_track = [1,2,3]
#harms_to_track = [1,2,3,4,5,6,7,8,9,10]

harms_to_label = [1,2,3]#,21,35]
harms_to_label = range(10)

sub_cant_phase = True
plot_first_drive = False

ax0val = None   # um
ax1val = None   # um
ax2val = None   # um

maxthrow = 80
minsep = 15

#ylim = (1e-21, 1e-14)
#ylim = (1e-7, 1e-1)
ylim = ()
arrow_fac = 5

lpf = 2500   # Hz

file_inds = (0, 10000)

diag = False


###########################################################



allfiles, lengths = bu.find_all_fnames(dir1)

sep0background = bgu.Background(allfiles)
sep0background.load_axvecs(find_again=True)#False)

xposvec = np.array(sep0background.axvecs[0].keys())
xposvec.sort()
nxpos = len(xposvec)


backgrounds = {}
seps = maxthrow + minsep - xposvec
#for sepind, sep in enumerate(seps):
for xind, xpos in enumerate(xposvec):
    progstr = '%i / %i separation' % (xind+1, nxpos)
    suffix = 'xpos' + str(int(xpos)) #str(int(seps[xind]))
    sepXbackground = bgu.Background(allfiles)
    if load:
        path = sepXbackground.get_savepath(suffix=suffix)
        try:
            temp_obj = pickle.load(open(path, 'rb'))
            sepXbackground = temp_obj
        except:
            print "Couldn't load data..."
            load = False
    if not load:
        sepXbackground.select_by_position(ax0val=xpos)
        sepXbackground.analyze_background(file_inds=file_inds, \
                                          ext_cant_drive=ext_cant_drive, \
                                          ext_cant_ind=ext_cant_ind, \
                                          progstr=progstr, harms_to_track=harms_to_track)
        #sepXbackground.filter_background_vs_time(btype='lowpass', order=3, Tc=0.10)

        if save:
            sepXbackground.save(suffix=suffix)

    backgrounds[xpos] = sepXbackground

backgrounds[xposvec[0]].plot_background(harms_to_plot=harms_to_track, plot_temp=False, \
                                        harms_to_label=harms_to_label)


nfiles_per_sep = backgrounds[xposvec[0]].nfiles
for xind, xpos in enumerate(xposvec):
    backgrounds[xpos].filter_background_vs_time(btype='lowpass', order=3, Tc=30)
    amps = backgrounds[xpos].amps_lpf
    times = backgrounds[xpos].times
    lab = '%i' % xpos
    for ind in [0,1]:
        plt.figure(ind+1)
        if ind == 0:
            title = "Fundamental: $f_0$"
        else:
            title = "Harmonic: %i $f_0$" % (ind+1)
        plt.title(title)
        plt.plot(times, amps[1][ind], label=lab)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [N]')

plt.figure(1)
plt.legend(loc=0)
plt.figure(2)
plt.legend(loc=0)

plt.show()

