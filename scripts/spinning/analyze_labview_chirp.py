import os, fnmatch, sys

import dill as pickle

import scipy.interpolate as interp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config




## This script assumes there aren't that many chirps and thus opens
## them all, with subfuntions to plot various parts

dir1 = '/data/20171221/bead6/labview_chirps_1-1205Hz'
final_chirp = 1250

def make_file_objs(datadir, hpf=False, hpf_freq=1.0, \
                   detrend=False, diag=False):

    objs = []
    files = bu.find_all_fnames(datadir)

    for fil in files:
        df = bu.DataFile()
        df.load(fil)

        df.calibrate_stage_position()

        if hpf:
            df.high_pass_filter(fc=hpf_freq)
        if detrend:
            df.detrend_poly()
            
        objs.append(df)

    return objs


def plot_chirps(filobjs, files=[0,1], drive_elec=3, show=True):
    
    chirp_fig, chirp_ax = plt.subplots(len(files),2,figsize=(14,8))

    for fil_ind in files:
        df = filobjs[fil_ind]
        Nsamp = len(df.pos_data[0])

        t = np.linspace(0, (Nsamp-1)*(1.0/df.fsamp), Nsamp)

        fft = np.fft.rfft(df.electrode_data[drive_elec])
        freqs = np.fft.rfftfreq(Nsamp,d=t[1]-t[0])
        asd = np.abs(fft)

        #epsd, freqs = mlab.psd(df.electrode_data[drive_elec], \
        #                       Fs=df.fsamp, NFFT=Nsamp)

        chirp_ax[fil_ind,0].plot(t, df.electrode_data[drive_elec])
        chirp_ax[fil_ind,0].set_xlabel('Time [s]')
        chirp_ax[fil_ind,0].set_xlabel('Voltage [V]')

        chirp_ax[fil_ind,1].loglog(freqs, asd)
        chirp_ax[fil_ind,1].set_xlabel('Frequency [Hz]')
        chirp_ax[fil_ind,1].set_xlabel('ASD [arb]')

    plt.tight_layout()

    if show:
        plt.show()



def plot_response_to_chirp(filobjs, files=[0,1], drive_elec=3, \
                           chirp_length=60., show=True, late=False,
                           plot_final_chirp=False, final_chirp=100):

    early_fig, early_ax = plt.subplots(len(files),2,figsize=(14,8),\
                                       sharex='col', sharey='col')
        
    for fil_ind in files:
        df = filobjs[fil_ind]
        Nsamp = len(df.pos_data[0])

        t = np.linspace(0, (Nsamp-1)*(1.0/df.fsamp), Nsamp)
        if not late:
            tbool = t <= chirp_length
        elif late:
            tbool = t > chirp_length

        labels = {0: 'X', 1: 'Y', 2: 'Z'}
        dat = [[], [], []]
        for ind in [0,1,2]:

            fft = np.fft.rfft(df.pos_data[ind][tbool])
            freqs = np.fft.rfftfreq(int(np.sum(tbool)),d=t[1]-t[0])
            asd = np.abs(fft)

            #rpsd, freqs = mlab.psd(df.pos_data[ind][tbool], \
            #                       Fs=df.fsamp, NFFT=int(np.sum(tbool)))

            early_ax[fil_ind,0].plot(t[tbool], df.pos_data[ind][tbool], \
                                     label=labels[ind])
            early_ax[fil_ind,0].set_xlabel('Time [s]')
            early_ax[fil_ind,0].set_ylabel('Response [V]')

            early_ax[fil_ind,1].loglog(freqs, asd)
            early_ax[fil_ind,1].set_xlabel('Frequency [Hz]')
            early_ax[fil_ind,1].set_ylabel('ASD [arb]')

            if plot_final_chirp:
                early_ax[fil_ind,1].axvline(x=final_chirp, color='r')

    plt.tight_layout()
    if show:
        plt.show()










filobjs = make_file_objs(dir1)

plot_chirps(filobjs, show=False)
plot_response_to_chirp(filobjs, show=False, \
                       plot_final_chirp=True, final_chirp=final_chirp)
plot_response_to_chirp(filobjs, late=True, \
                       plot_final_chirp=True, final_chirp=final_chirp)
