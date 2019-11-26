import os, fnmatch, sys, time

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings('ignore')


#dirs = ['/data/20180122/bead4/dual_drive_testing/33120A_burleigh_chirp_4']
#dirs = ['/data/20180122/bead4/dual_drive_testing/3325A_trek_chirp_2']

dirs = ['/data/20180126/hvamp_tf/3325A_trek']

hvamp_key = 'trek' #'burleigh'
title = 'Trek HV Amp - 2'
#title = ''

recompute = True
save = False

computed_tf_path = '/rot_data/hvamp_tfs.p'



#########


if hvamp_key == 'trek':
    monitor_div = 200
    drive_axes = [4]
    resp_axes = [6]

if hvamp_key == 'burleigh':
    monitor_div = 100
    drive_axes = [5]
    resp_axes = [7]

try:
    computed_tf_dict = pickle.load(open(computed_tf_path, 'rb'))
except:
    computed_tf_dict = {}
    parts = computed_tf_path.split('/')
    parent_dir = '/'
    for ind, part in enumerate(parts):
        if ind == 0 or ind == len(parts) - 1:
            continue
        parent_dir += part
        parent_dir += '/'
        if not os.path.isdir(parent_dir):
            os.mkdir(parent_dir)



file_inds = (0, 10000)


###########################################################

def lowpass_filter(freq, amp, cutoff, order=1.0):
    return amp / (1.0 + 1.0j * (freq / cutoff)**order)


def build_hvamp_tf(files, hvamp_key='trek', drive_axes=[4], resp_axes=[6],\
                   monitor_div=200, file_inds=(0,10000), save='True', \
                   title=''):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then plots the amplitude spectral density of any number of data
       or cantilever/electrode drive signals

       INPUTS: hvamp_key, string specifying which hvamp (to store tf)
                          ex. 'burleigh', 'trek'
               files, list of files names to extract data
               drive_axes, list of other_data axes with drive
               resp_axes, list of other_data axes with HV monitor
               file_inds, indices for min and max file

       OUTPUTS: none, plots stuff
    '''



    files = [(os.stat(path), path) for path in files]
    files = [(stat.st_ctime, path) for stat, path in files]
    files.sort(key = lambda x: (x[0]))
    files = [obj[1] for obj in files]

    files = files[file_inds[0]:file_inds[1]]

    tffreqs = []
    tfvals = []

    old_per = 0
    print("Processing %i files..." % len(files))
    print("Percent complete: ")
    for fil_ind, fil in enumerate(files):
        
        # Display percent completion
        per = int(100. * float(fil_ind) / float(len(files)) )
        if per > old_per:
            print(old_per, end=' ')
            sys.stdout.flush()
            old_per = per

        if hvamp_key in computed_tf_dict and not recompute:
            tf = computed_tf_dict[hvamp_key]
            tffreqs = tf[0]
            tfvals = tf[1]
            break

        # Load data
        df = bu.DataFile()
        try:
            df.load(fil)
            df.load_other_data()
        except:
            continue
            
        df.detrend_poly()

        for axind, ax in enumerate(drive_axes):
            ax = ax - 3
            rax = resp_axes[axind] - 3

            normfac = len(df.other_data[ax]) * df.fsamp * 0.5

            freqs = np.fft.rfftfreq(len(df.other_data[ax]), d=1.0/df.fsamp)

            fft = np.fft.rfft(df.other_data[ax])
            rfft = np.fft.rfft(df.other_data[rax])

            maxind = np.argmax(np.abs(fft[1:])) + 1  # ignore dc bin

            tfval = rfft[maxind] * monitor_div / fft[maxind] 
            
            tfvals.append(tfval)
            tffreqs.append(freqs[maxind])

    tffreqs = np.array(tffreqs)
    tfvals = np.array(tfvals)

    sortinds = np.argsort(tffreqs)
    tffreqs = tffreqs[sortinds]
    tfvals = tfvals[sortinds]

    computed_tf_dict[hvamp_key] = (tffreqs, tfvals)

    fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True)

    magfit = lambda x,a,fc: np.abs(lowpass_filter(x,a,fc))

    guess = [np.abs(tfvals)[0], 1000]

    popt1, pcov1 = opti.curve_fit(magfit, tffreqs, np.abs(tfvals))

    phasefit = lambda x,a,fc: np.angle(lowpass_filter(x,a,fc))
    #phasefit = lambda x,a: np.angle(lowpass_filter(x,a,popt1[1]))

    popt2, pcov2 = opti.curve_fit(phasefit, tffreqs, np.angle(tfvals))

    ax[0].loglog(tffreqs, np.abs(tfvals), linewidth=2, label='data')
    ax[0].loglog(tffreqs, magfit(tffreqs, *popt1), '--', label='fit')
    ax[1].semilogx(tffreqs, np.angle(tfvals)*(180.0/np.pi), linewidth=2)
    ax[1].semilogx(tffreqs, phasefit(tffreqs, *popt1)*(180.0/np.pi), '--')

    ax[0].set_ylabel('TF Mag [abs]', fontsize=14)
    ax[1].set_xlabel('Frequency [Hz]', fontsize=14)
    ax[1].set_ylabel('TF Phase [deg]', fontsize=14)


    plt.setp(ax[0].get_xticklabels(), fontsize=14, visible = True)
    plt.setp(ax[0].get_yticklabels(), fontsize=14, visible = True)
    plt.setp(ax[1].get_xticklabels(), fontsize=14, visible = True)
    plt.setp(ax[1].get_yticklabels(), fontsize=14, visible = True)

    ax[0].yaxis.grid(which='major', color='k', linestyle='--', linewidth=0.5)
    ax[0].xaxis.grid(which='major', color='k', linestyle='--', linewidth=0.5)
    ax[1].yaxis.grid(which='major', color='k', linestyle='--', linewidth=0.5)
    ax[1].xaxis.grid(which='major', color='k', linestyle='--', linewidth=0.5)

    ax[0].legend()

    plt.tight_layout()

    if len(title):
        plt.subplots_adjust(top=0.92)
        plt.suptitle(title, fontsize=18)

    if save:
        pickle.dump(computed_tf_dict, open(computed_tf_path, 'wb'))

    print()
    print("AMP FIT: %0.1e gain, %0.1e cutoff" % (popt1[0], popt1[1]))
    print("PHASE FIT: %0.1e gain, %0.1e cutoff" % (popt2[0], popt2[1]))
    sys.stdout.flush()
    
    plt.show()



allfiles, lengths = bu.find_all_fnames(dirs)

build_hvamp_tf(allfiles, hvamp_key=hvamp_key, monitor_div=monitor_div, \
               drive_axes=drive_axes, resp_axes=resp_axes, save=save, 
               title=title)
