import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

plt.rcParams.update({'font.size': 14})


### Define some parameters for all drives and all signals
fsamp = 5000.0
nsamp = 50000

fdrive = 97.0

### Set the number of realizations to test, where each one will
### have a pseudorandom component to the amplitude 
ndrive = 100
nsig = 100

### Set some numerical scales to test how the function behaves
driveamp_scale = np.pi * 1e-14
sigamp_scale = 7.0 * np.pi

### Scale (sigma) for AWGN, relative to amplitude
relative_drive_noise = 0.001  
relative_sig_noise = 0.1

### Include terms at the same frequency, but out-of-phase to test the
### robustness of this estimation method
include_out_of_phase = True
out_of_phase_scale = 0.25
limit_90deg = True
unique_out_of_phase_amp = False

### Hard-coded xlimits for histogram and plotting to generate 
### similar plots for different parameters
# half_width = 5e-4
half_width = 0.3
xlim = (1.0-half_width, 1.0+half_width)

print_each = False





########################################################################
########################################################################
########################################################################



def correlation(drive, response, fsamp, fdrive, filt = False, band_width = 1):
    '''Compute the full correlation between drive and response,
       correctly normalized for use in step-calibration.

       INPUTS:   drive, drive signal as a function of time
                 response, resposne signal as a function of time
                 fsamp, sampling frequency
                 fdrive, predetermined drive frequency
                 filt, boolean switch for bandpass filtering
                 band_width, bandwidth in [Hz] of filter

       OUTPUTS:  corr_full, full and correctly normalized correlation'''

    ### First subtract of mean of signals to avoid correlating dc
    drive = drive-np.mean(drive)
    response = response-np.mean(response)

    ### bandpass filter around drive frequency if desired.
    if filt:
        b, a = signal.butter(3, [2.*(fdrive-band_width/2.)/fsamp, \
                             2.*(fdrive+band_width/2.)/fsamp ], btype = 'bandpass')
        drive = signal.filtfilt(b, a, drive)
        response = signal.filtfilt(b, a, response)
    
    ### Compute the number of points and drive amplitude to normalize correlation
    lentrace = len(drive)
    drive_amp = np.sqrt(2)*np.std(drive)

    ### Define the correlation vector which will be populated later
    corr = np.zeros(int(fsamp/fdrive))

    ### Zero-pad the response
    response = np.append(response, np.zeros(int(fsamp / fdrive) - 1) )

    ### Build the correlation
    n_corr = len(drive)
    for i in range(len(corr)):
        ### Correct for loss of points at end
        correct_fac = 2.0*n_corr/(n_corr-i) ### x2 from empirical test
        corr[i] = np.sum(drive*response[i:i+n_corr])*correct_fac

    return corr * (1.0 / (lentrace * drive_amp))





def progress_bar(count, total, suffix='', bar_len=50, newline=True):
    '''Prints a progress bar and current completion percentage.
       This is useful when processing many files and ensuring
       a script is actually running and going through each file

           INPUTS: count, current counting index
                   total, total number of iterations to complete
                   suffix, option string to add to progress bar
                   bar_len, length of the progress bar in the console

           OUTPUTS: none
    '''
    
    if len(suffix):
        max_bar_len = 80 - len(suffix) - 17
        if bar_len > max_bar_len:
            bar_len = max_bar_len

    if count == total - 1:
        percents = 100.0
        bar = '#' * bar_len
    else:
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '#' * filled_len + '-' * (bar_len - filled_len)
    
    ## This next bit writes the current progress bar to stdout, changing
    ### the string slightly depending on the value of percents (1, 2 or 3 digits), 
    ### so the final length of the displayed string stays constant.
    if count == total - 1:
        sys.stdout.write('[%s] %s%s ... %s\r' % (bar, percents, '%', suffix))
    else:
        if percents < 10:
            sys.stdout.write('[%s]   %s%s ... %s\r' % (bar, percents, '%', suffix))
        else:
            sys.stdout.write('[%s]  %s%s ... %s\r' % (bar, percents, '%', suffix))

    sys.stdout.flush()
    
    if (count == total - 1) and newline:
        print()





tarr = np.arange(0, nsamp/fsamp, 1.0/fsamp)

### Generate a set of drive signals with different amplitudes to ensure the 
### correlation is properly normalized and independent of the drive amplitude
drives = []
for i in range(ndrive):
    ### Use a pseudorandom value for the amplitude
    driveamp = np.abs( driveamp_scale * (1.0 + np.random.randn()) )
    sine = driveamp * np.sin(2.0 * np.pi * fdrive * tarr)

    ### Sample some additive white gaussian noise
    noise = relative_drive_noise * driveamp * np.random.randn(nsamp)

    ### Construct the drive
    drives.append(sine+noise)



### Compute the correlation of nsig different signals against each of the
### drive signals defined abovue
ratios = []
mean_ratios = []
for i in range(nsig):
    ### Use a pseudorandom value for the amplitude
    sigamp = sigamp_scale * np.random.randn()
    sig = sigamp * np.sin(2.0 * np.pi * fdrive * tarr)

    ### Optionally include an out of phase component
    if include_out_of_phase:
        ### Generate a new value for the amplitude to avoid numerical
        ### artifacts and sample more of the possible fluctuations
        if unique_out_of_phase_amp:
            sigamp2 = sigamp_scale * np.random.randn()
        else:
            sigamp2 = sigamp

        ### Limit the phase to not be within 45deg of either the in-phase or the 
        ### inverted signal  
        if limit_90deg:
            phase = np.random.choice([np.random.uniform(np.pi/4.0, 3.0*np.pi/4.0), \
                                      np.random.uniform(5.0*np.pi/4.0, 7.0*np.pi/4.0)])
        else:
            phase = np.random.uniform(0, 2.0*np.pi)

        sig += out_of_phase_scale * sigamp2 * np.sin(2.0 * np.pi * fdrive * tarr + phase)

    ### Sample some additive white gaussian noise
    sig += relative_sig_noise * sigamp * np.random.rand(nsamp)

    ### Loop over each drive and compute the correlaion
    corrs = []
    maxcorrs = []
    for drive in drives:
        corr = correlation(drive, sig, fsamp, fdrive)

        ### Select the in-phase result (the zeroth index)
        ratios.append(corr[0]/sigamp)
        corrs.append(corr[0])

        ### Keep track of the max correlations just in case
        maxcorrs.append(np.max(corr))

    ### Compute some stuff
    mean = np.mean(corrs)
    std = np.std(corrs)

    mean_ratios.append(mean/sigamp)

    if print_each:
        print()
        print('####################################################')
        print()
        #print(corrs)
        print('Sigamp : {:+0.6f}'.format(sigamp))
        print()
        print('  Mean (inphase) : {:+0.6f}'.format(mean))
        print('           Sigma : {:+0.6f}'.format(std))
        print('           Ratio : {:+0.6f}'.format(mean / sigamp))
        print()
        print(' Mean (max corr) : {:+0.6f}'.format(np.mean(maxcorrs)))
        print('           Sigma : {:+0.6f}'.format(np.std(maxcorrs)))
        print('           Ratio : {:+0.6f}'.format(np.mean(maxcorrs) / sigamp))
    else:
        progress_bar(i, nsig)

### Generate some strings for annotation
title_str = '{:d} unique signals, each correlated to {:d} unique drives'\
                    .format(nsig, ndrive)
plot_str_1 = 'Drive amp scale : {:0.4g}\nRelative drive noise : {:0.4g}'\
                    .format(driveamp_scale, relative_drive_noise)
plot_str_2 = 'Sig amp scale : {:0.4g}\nRelative sig noise : {:0.4g}'\
                    .format(sigamp_scale, relative_sig_noise)

fig1, ax1 = plt.subplots(1, 1, figsize=(8,6))
vals, bins, _ = ax1.hist(ratios, bins=20, range=xlim)

ax1.set_xlabel('Ratio of in-phase correlation to known amplitude')
# ax1.set_ylabel('Count')

### Compute some alignment marks for text annotations
maxval = np.max(vals)
top = maxval #*0.995

deltax = xlim[1] - xlim[0]
left = xlim[0] - 0.005*deltax
right = xlim[1] + 0.005*deltax

### Populate the figure with important information
ax1.text(left, top, plot_str_1, ha='left', va='top', size=12, \
         bbox=dict(boxstyle='square', ec='k', fc='w', alpha=0.7))
ax1.text(right, top, plot_str_2, ha='right', va='top', size=12, \
         bbox=dict(boxstyle='square', ec='k', fc='w', alpha=0.7))

ax1.set_title(title_str, fontdict={'fontsize': 16})

fig1.tight_layout()
plt.show()
