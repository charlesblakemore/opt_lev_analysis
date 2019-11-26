import os, fnmatch, sys, time

import dill as pickle

import scipy.interpolate as interp
import scipy.optimize as opti

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

import peakdetect as pdet

import warnings
warnings.filterwarnings('ignore')


#dirs = ['/data/20180122/bead4/drive_at_rotfreq/test2_prefield', \
#        '/data/20180122/bead4/drive_at_rotfreq/test2_fieldon_getting_inphase', \
#        '/data/20180122/bead4/drive_at_rotfreq/test2_fieldon_getting_inphase2', \
#        '/data/20180122/bead4/drive_at_rotfreq/test2_fieldon_getting_inphase3', \
#        '/data/20180122/bead4/drive_at_rotfreq/test2_fieldon_chirping1', \
#        '/data/20180122/bead4/drive_at_rotfreq/test2_fieldoff_1', \
#        '/data/20180122/bead4/drive_at_rotfreq/test2_fieldoff_2']

#dirs = ['/data/20180122/bead4/drive_at_rotfreq/test3_prefield', \
#        '/data/20180122/bead4/drive_at_rotfreq/test3_fieldon_getting_inphase']

plot_lastn_hours = 120
plot_together = True

recompute = False #True

computed_freq_path = '/rot_data/20180122_bead/drive_at_rotfreq/test3.p'
try:
    computed_freq_dict = pickle.load(open(computed_freq_path, 'rb'))
except:
    computed_freq_dict = {}
    parts = computed_freq_path.split('/')
    parent_dir = '/'
    for ind, part in enumerate(parts):
        if ind == 0 or ind == len(parts) - 1:
            continue
        parent_dir += part
        parent_dir += '/'
        if not os.path.isdir(parent_dir):
            os.mkdir(parent_dir)

logtime = False

minfreq = 5000
maxfreq= 10000

# Peak finding params
lookahead = 5
delta_per = 0.02

# Percent of center frequency to include in band pass
# for feature frequency identification
percent_band = 0.05

# Drive threshold (smaller is more stringent)
drive_thresh = 5e-4 #1e-3

dirmarkers = {2: ('Field On', 'k', '-'), \
              3: ('Changing Freq', 'g', '--'), \
              4: ('Stop at 5400Hz', 'm', '--'), \
              6: ('Field Off', 'r', '-'), \
              7: ('Catch at 5250Hz', 'm', '-'), \
              8: ('', 'r', '-')}

dirmarkers = {3: ('Chirp Down', 'm', '--'), \
              4: ('Stop at 5000Hz', 'm', '-'), \
              5: ('Shit Happens', 'k', '--'), \
              7: ('Catch at 3450Hz', 'k', '-'), \
              8: ('Chirp Down', 'g', '--'), \
              9: ('Stop at 3000Hz', 'g', '-'), \
              10: ('Chirp Up', 'b', '--'), \
              11: ('Stop at 3250Hz', 'b', '-')}


dirmarkers = {}

field_on_at_beginning = False #True

step10 = False
invert_order = False

plot_peaks = False
plot_drive_peaks = False #True

file_inds = (0, 10000)

userNFFT = 2**12
diag = False

fullNFFT = True

###########################################################

def gauss(x, A, mu, sig, c):
    return A * np.exp(-(x-mu)**2 / (2 * sig**2)) + c



def fit_monochromatic_line(files, data_axes=[0,1], drive_axes=[6], diag=True, \
                           minfreq=2000, maxfreq=8000, pickfirst=True, \
                           colormap='jet', sort='time', file_inds=(0,10000), \
                           dirlengths=[]):
    '''Loops over a list of file names, loads each file, diagonalizes,
       then plots the amplitude spectral density of any number of data
       or cantilever/electrode drive signals

       INPUTS: files, list of files names to extract data
               data_axes, list of pos_data axes to plot
               diag, boolean specifying whether to diagonalize
               colormap, matplotlib colormap string for sort
               sort, sorting key word
               file_inds, indices for min and max file

       OUTPUTS: none, plots stuff
    '''



    files = [(os.stat(path), path) for path in files]
    files = [(stat.st_ctime, path) for stat, path in files]
    files.sort(key = lambda x: (x[0]))
    files = [obj[1] for obj in files]

    files = files[file_inds[0]:file_inds[1]]

    if step10:
        files = files[::10]
    if invert_order:
        files = files[::-1]

    times = []
    peak_pos = []
    drive_pos = []
    errs = []
    drive_errs = []

    colors = bu.get_color_map(len(files), cmap=colormap)

    bad_inds = []

    oldtime = 0
    old_per = 0
    print(files[-1])
    print("Processing %i files..." % len(files))
    print("Percent complete: ")
    for fil_ind, fil in enumerate(files):
        
        # Display percent completion
        per = int(100. * float(fil_ind) / float(len(files)) )
        if per > old_per:
            print(old_per, end=' ')
            sys.stdout.flush()
            old_per = per

        if fil in computed_freq_dict and not recompute:
            soln = computed_freq_dict[fil]
            times.append(soln[0])
            peak_pos.append(soln[1])
            errs.append(soln[2])
            drive_pos.append(soln[3])
            drive_errs.append(soln[4])
            old = soln[1]
            old_drive = soln[3]
            continue
        else:
            newsoln = [0, 0, 0, 0, 0]

        # Load data
        df = bu.DataFile()
        try:
            df.load(fil)
        except:
            continue

        if len(drive_axes) > 0:
            df.load_other_data()
            
        ctime = time.mktime(df.time.timetuple())
            
        times.append(ctime)

        newsoln[0] = ctime
    
        cpos = []
        errvals = []
        for axind, ax in enumerate(data_axes):
        
            #fac = df.conv_facs[ax]
            if fullNFFT:
                NFFT = len(df.pos_data[ax])
            else:
                NFFT = userNFFT

            psd, freqs = mlab.psd(df.pos_data[ax], Fs=df.fsamp, NFFT=NFFT)

            fitbool = (freqs > minfreq) * (freqs < maxfreq)

            maxval = np.max(psd[fitbool])
            delta = delta_per*maxval
            peaks = pdet.peakdetect(psd[fitbool], lookahead=lookahead, delta=delta)

            pos_peaks = peaks[0]
            neg_peaks = peaks[1]

            if plot_peaks:
                for peakind, pos_peak in enumerate(pos_peaks):
                    try:
                        neg_peak = neg_peaks[peakind]
                    except:
                        continue
                    plt.loglog(freqs[fitbool][pos_peak[0]], pos_peak[1], 'x', color='r')
                    plt.loglog(freqs[fitbool][neg_peak[0]], neg_peak[1], 'x', color='b')
                plt.loglog(freqs[fitbool], psd[fitbool])
                plt.show()

            np_pos_peaks = np.array(pos_peaks)

            try:
            
                if fil_ind == 0:
                    ucutoff = 100000
                    lcutoff = 0
                else:
                    ucutoff = (1.0 + percent_band) * old
                    lcutoff = (1.0 - percent_band) * old

                vals = []
                for peakind, peak in enumerate(pos_peaks):
                    newval = freqs[fitbool][peak[0]]
                    if newval > ucutoff:
                        continue
                    if newval < lcutoff:
                        continue

                    vals.append(newval) 

                cpos.append(np.mean(vals))
                for val in vals:
                    errvals.append(val)

            except:
                print('FAILED')
                continue


        drive_cpos = []
        drive_errvals = []
        for axind, ax in enumerate(drive_axes):
            ax = ax - 3

            if fullNFFT:
                NFFT = len(df.other_data[ax])
            else:
                NFFT = userNFFT

            psd, freqs = mlab.psd(df.other_data[ax], Fs=df.fsamp, NFFT=NFFT)

            fitbool = (freqs > minfreq) * (freqs < maxfreq)

            maxval = np.max(psd[fitbool])
            delta = delta_per*maxval
            peaks = pdet.peakdetect(psd[fitbool], lookahead=lookahead, delta=delta)

            pos_peaks = peaks[0]
            neg_peaks = peaks[1]

            np_pos_peaks = np.array(pos_peaks)

            if plot_drive_peaks:
                for peakind, pos_peak in enumerate(pos_peaks):
                    try:
                        neg_peak = neg_peaks[peakind]
                    except:
                        continue
                    plt.loglog(freqs[fitbool][pos_peak[0]], pos_peak[1], 'x', color='r')
                    plt.loglog(freqs[fitbool][neg_peak[0]], neg_peak[1], 'x', color='b')
                plt.loglog(freqs[fitbool], psd[fitbool])
                plt.show()

            try:
                maxind = np.argmax(np_pos_peaks[:,1])

                maxpeak = pos_peaks[maxind]
                vals = []
    
                if maxpeak[1] < np.mean(psd[fitbool]) * (1.0 / drive_thresh):
                    vals.append(np.nan)
                else:
                    vals.append(freqs[fitbool][maxpeak[0]])
    
                #for peakind, peak in enumerate(pos_peaks):
                #    if peak[0] < 1e-2:
                #        continue
                #    newval = freqs[fitbool][peak[0]]
                #    if newval > ucutoff:
                #        continue
                #    if newval < lcutoff:
                #        continue
                #
                #    vals.append(newval) 

                drive_cpos.append(np.mean(vals))
                for val in vals:
                    drive_errvals.append(val)

            except:
                print('FAILED DRIVE ANALYSIS')
                continue

        freqval = np.mean(cpos)
        errval = np.std(errvals)

        drive_freqval = np.mean(drive_cpos)
        drive_errval = np.std(drive_errvals)

        if len(cpos) < 0:
            bad_inds.append(fil_ind)
        else:
            peak_pos.append(freqval)
            drive_pos.append(drive_freqval)
            errs.append(errval)
            drive_errs.append(drive_errval)

            old = np.mean(cpos)
            old_drive = np.mean(drive_cpos)

            newsoln[1] = freqval
            newsoln[2] = errval

            newsoln[3] = drive_freqval
            newsoln[4] = drive_errval

        oldtime = ctime

        computed_freq_dict[fil] = newsoln

    times2 = np.delete(times, bad_inds)
    times2 = times2 - np.min(times)
    
    peak_pos = np.array(peak_pos)
    drive_pos = np.array(drive_pos)

    sortinds = np.argsort(times2)
    times2 = times2[sortinds]
    peak_pos = peak_pos[sortinds]
    drive_pos = drive_pos[sortinds]

    times2 = np.array(times2)
    peak_pos = np.array(peak_pos)
    drive_pos = np.array(drive_pos)

    bad_inds = np.array(bad_inds)

    max_hours = np.max( times2*(1.0/3600) )
    plot_ind = np.argmin(np.abs(times2*(1.0/3600) - (max_hours - plot_lastn_hours) ) )

    if not plot_together:
        fig, ax = plt.subplots(2,1,figsize=(10,10), sharex=True, sharey=True)
    elif plot_together:
        fig, ax = plt.subplots(1,1,figsize=(10,5), sharex=True, sharey=True)
        ax = [ax]
    ax[0].errorbar(times2[plot_ind:]*(1.0/3600), peak_pos[plot_ind:], yerr=errs[plot_ind:], fmt='o', \
                   color='C0', label='Bead Rotation')
    if plot_together:
        ax[0].errorbar(times2[plot_ind:]*(1.0/3600), drive_pos[plot_ind:], \
                       yerr=drive_errs[plot_ind:], fmt='o', alpha=0.15, \
                       color='C1', label='Drive')
    elif not plot_together:
        ax[1].errorbar(times2[plot_ind:]*(1.0/3600), drive_pos[plot_ind:], \
                       yerr=drive_errs[plot_ind:], fmt='o', color='C1', label='Drive')

    if logtime:
        ax[0].set_xscale("log")
        if not plot_together:
            ax[1].set_xscale("log")

    if not plot_together:
        ax[1].set_xlabel('Elapsed Time [hrs]', fontsize=14)
    elif plot_together:
        ax[0].set_xlabel('Elapsed Time [hrs]', fontsize=14)

    ax[0].set_ylabel('Rotation Frequency [Hz]', fontsize=14)
    if not plot_together:
        ax[1].set_ylabel('Rotation Frequency [Hz]', fontsize=14)

    plt.setp(ax[0].get_xticklabels(), fontsize=14, visible = True)
    plt.setp(ax[0].get_yticklabels(), fontsize=14, visible = True)
    if not plot_together:
        plt.setp(ax[1].get_xticklabels(), fontsize=14, visible = True)
        plt.setp(ax[1].get_yticklabels(), fontsize=14, visible = True)

    ax[0].yaxis.grid(which='major', color='k', linestyle='--', linewidth=0.5)
    ax[0].xaxis.grid(which='major', color='k', linestyle='--', linewidth=0.5)
    if not plot_together:
        ax[1].yaxis.grid(which='major', color='k', linestyle='--', linewidth=0.5)
        ax[1].xaxis.grid(which='major', color='k', linestyle='--', linewidth=0.5)

    label_keys = list(dirmarkers.keys())

    plot_first = max_hours <= plot_lastn_hours

    if field_on_at_beginning and plot_first:
        ax[0].axvline(x=times2[0], lw=2, label='Field On', color='r', ls='-')


    if len(dirlengths) != 0:
        oldlength = 0
        for dirind, length in enumerate(dirlengths):
            oldlength += length
            tlength = oldlength - np.sum(bad_inds < oldlength)

            if tlength < plot_ind:
                continue

            if dirind+2 in label_keys:
                ax[0].axvline(x=times2[tlength]*(1.0/3600), lw=2, label=dirmarkers[dirind+2][0], \
                           color=dirmarkers[dirind+2][1], ls=dirmarkers[dirind+2][2])


    ax[0].legend()

    plt.tight_layout()

    pickle.dump(computed_freq_dict, open(computed_freq_path, 'wb'))

    plt.show()



allfiles, lengths = bu.find_all_fnames(dirs)

fit_monochromatic_line(allfiles, minfreq=minfreq, maxfreq=maxfreq, \
                       file_inds=file_inds, diag=diag, dirlengths=lengths)
