import sys, time, itertools

import dill as pickle

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opti
import scipy.linalg as linalg

import bead_util as bu
import grav_util as gu
import calib_util as cal
import transfer_func_util as tf
import configuration as config

import warnings
warnings.filterwarnings("ignore")


##################################################################
######################## Script Params ###########################

minsep = 15       # um
maxthrow = 80     # um
beadheight = 20   # um

#data_dir = '/data/20180314/bead1/grav_data/ydrive_6sep_1height_shield-2Vac-2200Hz_cant-0mV'
#data_dir = '/data/20180524/bead1/grav_data/many_sep_many_h'

#data_dir = '/data/20180613/bead1/grav_data/no_shield/X60-80um_Z20-30um'
data_dir = '/data/20180618/bead1/grav_data/shield/X60-80um_Z15-25um'

split = data_dir.split('/')
name = split[-1]
date = split[2]

save_alphadat = True
load_alphadat = False
alphadat_filname = '/processed_data/alphadat/' + date + '_' + name + '.alphadat'

save_fildat = True
load_fildat = False
fildat_filname = '/processed_data/fildat/' + date + '_' + name + '.fildat'



savepath = '/sensitivities/20180618_grav-shield_1.npy'
save = True
load = False
file_inds = (0, 2000)
max_file_per_pos = 10

theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'
#theory_data_dir = '/data/grav_sim_data/1um_spacing_x-0-p80_y-m250-p250_z-m20-p20/'

tfdate = ''
diag = False

confidence_level = 0.95

lamb_range = (1.7e-6, 1e-4)


userlims = [(5e-6, 50e-6), (-240e-6, 240e-6), (-10e-6, 10e-6)]
#userlims = [(5e-6, 20e-6), (-240e-6, 240e-6), (-5e-6, 0e-6)]
#userlims = []

tophatf = 300   # Hz, doesn't reconstruct data above this frequency
nharmonics = 10
harms = [2,3,4,5,6,7,8,9]

plotfilt = False
plot_just_current = False
figtitle = ''

ignoreX = False
ignoreY = False
ignoreZ = False

compute_min_alpha = False

noiseband = 10

##################################################################
################# Constraints to plot against ####################

alpha_plot_lims = (1000, 10**17)
lambda_plot_lims = (10**(-8), 10**(-3))


#limitdata_path = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/' + \
#                 'decca2_limit.txt'

limitdata_path = '/sensitivities/decca1_limits.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'


#limitdata_path2 = '/home/charles/opt_lev_analysis/gravity_sim/gravity_sim_v1/data/' + \
#                 'no_decca2_limit.txt'
limitdata_path2 = '/sensitivities/decca2_limits.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'



##################################################################
##################################################################
##################################################################






height = 25.0


if not plot_just_current:
    gfuncs, yukfuncs, lambdas, lims = gu.build_mod_grav_funcs(theory_data_dir)
    print "Loaded grav sim data"

    datafiles = bu.find_all_fnames(data_dir, ext=config.extensions['data'])

    datafiles = datafiles[file_inds[0]:file_inds[1]]
    print "Processing %i files..." % len(datafiles)

    if len(datafiles) == 0:
        print "Found no files in: ", data_dir
        quit()



    if not load_alphadat:
        if not load_fildat:
            fildat = gu.get_data_at_harms(datafiles, minsep=minsep, maxthrow=maxthrow, \
                                          beadheight=beadheight, plotfilt=plotfilt, \
                                          cantind=0, ax1='x', ax2='z', diag=diag, plottf=False, \
                                          nharmonics=nharmonics, harms=harms, \
                                          ext_cant_drive=True, ext_cant_ind=1, \
                                          max_file_per_pos=max_file_per_pos, userlims=userlims, \
                                          tfdate=tfdate, tophatf=tophatf, noiseband=noiseband)
            if save_fildat:
                gu.save_fildat(fildat_filname, fildat)
        else:
            fildat = gu.load_fildat(fildat_filname)


        alphadat = gu.find_alpha_vs_file(fildat, gfuncs, yukfuncs, lambdas, lims, \
                                         ignoreX=ignoreX, ignoreY=ignoreY, ignoreZ=ignoreZ, \
                                         plot_best_alpha=False, diag=diag)

        if save_alphadat:
            gu.save_alphadat(alphadat_filname, alphadat, lambdas, minsep, maxthrow, beadheight)

    else:
        stuff = gu.load_alphadat(alphadat_filname)

        alphadat = stuff['alphadat']
        lambdas = stuff['lambdas']
        minsep = stuff['minsep']
        maxthrow = stuff['maxthrow']
        beadheight = stuff['beadheight']
        

    
    fits, outdat, alphas_1, alphas_2, alphas_3 = gu.fit_alpha_vs_alldim(alphadat, lambdas, minsep=minsep, \
                                                                  maxthrow=maxthrow, beadheight=beadheight, \
                                                                  plot=False, scale_fac=1.0*10**9)
    

    #alphas_bf, alphas_95cl, fits = \
    #            fit_alpha_vs_sep_1height(alphadat, height, minsep=minsep, maxthrow=maxthrow, \
    #                                     beadheight=beadheight)









fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)
if diag:
    fig2, ax2 = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)

if not plot_just_current:
    ax.loglog(lambdas, alphas_1, linewidth=2, label='Constant From Planar Fit')
    ax.loglog(lambdas, np.abs(alphas_2), linewidth=2, label='Mean of De-Planed Data')
    ax.loglog(lambdas, alphas_3, linewidth=2, label='Std. Dev. of De-Planed Data')

ax.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
ax.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
ax.grid()

ax.set_xlim(lambda_plot_lims[0], lambda_plot_lims[1])
ax.set_ylim(alpha_plot_lims[0], alpha_plot_lims[1])

ax.set_xlabel('$\lambda$ [m]')
ax.set_ylabel('$\\alpha$')

ax.legend(numpoints=1, fontsize=9)

ax.set_title(figtitle)

plt.tight_layout()

if diag:
    ax2.loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
    ax2.loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')
    ax2.grid()

    ax2.set_xlim(lambda_plot_lims[0], lambda_plot_lims[1])
    ax2.set_ylim(alpha_plot_lims[0], alpha_plot_lims[1])

    ax2.set_xlabel('$\lambda$ [m]')
    ax2.set_ylabel('$\\alpha$')

    ax2.legend(numpoints=1, fontsize=9)

    ax2.set_title(figtitle)

    plt.tight_layout()

plt.show()


