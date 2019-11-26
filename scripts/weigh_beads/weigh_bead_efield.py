import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti
import scipy.constants as constants

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config
import transfer_func_util as tf

plt.rcParams.update({'font.size': 14})

dirs = ['/data/20180927/bead1/weigh_bead_dc/ramp_top_negative_bottom_at_p100', \
        '/data/20180927/bead1/weigh_bead_dc/ramp_top_negative_bottom_at_p100_10_repeats'
       ]


dirs = ['/data/20180927/bead1/weigh_bead_20e_10v_bottom_constant', \
       ]

#
V2 = 100.0
amp_gain = 200 #????




#dirs = ['/data/20181119/bead1/mass_meas/neg_charge_2', \
#       ]

dirs = ['/data/20181119/bead1/mass_meas/pos_charge_1', \
       ]

pos = True



mon_fac = 200

maxfiles = 1000 # Many more than necessary
lpf = 2500   # Hz

file_inds = (0, 500)

userNFFT = 2**12
diag = False


fullNFFT = False

###########################################################


power_dat = np.loadtxt('/power_v_bits/20181119_init.txt', delimiter=',')
bits_to_power = interp.interp1d(power_dat[0], power_dat[1])


e_top_dat = np.loadtxt('/calibrations/e-top_1V_optical-axis.txt', delimiter=',')
e_top_func = interp.interp1d(e_top_dat[0], e_top_dat[1])

e_bot_dat = np.loadtxt('/calibrations/e-bot_1V_optical-axis.txt', delimiter=',')
e_bot_func = interp.interp1d(e_bot_dat[0], e_bot_dat[1])



def line(x, a, b):
    return a * x + b





def weigh_bead_efield(files, colormap='jet', sort='time', file_inds=(0,10000), \
                      pos=False):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then plots the amplitude spectral density of any number of data
       or cantilever/electrode drive signals

       INPUTS: files, list of files names to extract data
               data_axes, list of pos_data axes to plot
               cant_axes, list of cant_data axes to plot
               elec_axes, list of electrode_data axes to plot
               diag, boolean specifying whether to diagonalize

       OUTPUTS: none, plots stuff
    '''

    files = [(os.stat(path), path) for path in files]
    files = [(stat.st_ctime, path) for stat, path in files]
    files.sort(key = lambda x: (x[0]))
    files = [obj[1] for obj in files]

    files = files[file_inds[0]:file_inds[1]]
    #files = files[::10]

    date = files[0].split('/')[2]

    charge_file = '/calibrations/charges/' + date
    if pos:
        charge_file += '_recharge.charge'
    else:
        charge_file += '.charge'

    q_bead = np.load(charge_file)[0] * constants.elementary_charge
    print(q_bead / constants.elementary_charge)

    run_index = 0

    masses = []

    nfiles = len(files)
    print("Processing %i files..." % nfiles)

    eforce = []
    power = []

    
    for fil_ind, fil in enumerate(files):#files[56*(i):56*(i+1)]):

        bu.progress_bar(fil_ind, nfiles)

        # Load data
        df = bu.DataFile()
        try:
            df.load(fil, load_other=True)
        except:
            continue

        df.calibrate_stage_position()

        df.calibrate_phase()

        if fil_ind == 0:
            init_phi = np.mean(df.zcal)

        top_elec = mon_fac * np.mean(df.other_data[6])
        bot_elec = mon_fac * np.mean(df.other_data[7])

        # Synth plugged in negative so just adding instead of subtracting negative
        Vdiff = V2 + amp_gain * df.synth_settings[0]

        Vdiff = np.mean(df.electrode_data[2]) - np.mean(df.electrode_data[1])

        Vdiff = top_elec - bot_elec


        force = - (Vdiff / (4.0e-3)) * q_bead
        force2 = (top_elec * e_top_func(0.0) + bot_elec * e_bot_func(0.0)) * q_bead

        try:
            mean_fb = np.mean(df.pos_fb[2])
            mean_pow = bits_to_power(mean_fb)
        except:
            continue


        #eforce.append(force)
        eforce.append(force2)
        power.append(mean_pow)

    eforce = np.array(eforce)
    power = np.array(power)

    power = power / np.mean(power)

    inds = np.abs(eforce) < 2e-13
    eforce = eforce[inds]
    power = power[inds]

    popt, pcov = opti.curve_fit(line, eforce*1e13, power, \
                                absolute_sigma=False, maxfev=10000)
    test_vals = np.linspace(np.min(eforce*1e13), np.max(eforce*1e13), 100)


    fit = line(test_vals, *popt)

    lev_force = -popt[1] / (popt[0] * 1e13)

    mass = lev_force / (9.806)

    mass_err = np.sqrt( pcov[0,0] / popt[0]**2 + \
                        pcov[1,1] / popt[1]**2 + \
                        np.abs(pcov[0,1]) / np.abs(popt[0]*popt[1]) ) * mass


    #masses.append(mass)

    print(mass * 1e12)
    print(mass_err * 1e12)


    plt.figure()

    plt.plot(eforce, power, 'o')
    plt.xlabel('Elec. Force [N]', fontsize=14)
    plt.ylabel('Levitation Power [arb]', fontsize=14)

    plt.tight_layout()

    plt.plot(test_vals*1e-13, fit, lw=2, color='r', \
             label='Implied mass: %0.3f ng' % (mass*1e12))
    plt.legend()



    plt.show()



    #print np.mean(masses) * 1e12
    #print np.std(masses) * 1e12



allfiles, lengths = bu.find_all_fnames(dirs, sort_time=True)
weigh_bead_efield(allfiles, pos=pos)
