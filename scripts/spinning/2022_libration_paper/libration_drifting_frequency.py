import os, sys, time, itertools, re, warnings, h5py
import numpy as np
import dill as pickle
from iminuit import Minuit, describe

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as optimize
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed
# ncore = 1
ncore = 24

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 14})





#################################
###                           ###
###   WHICH DATA TO ANALYZE   ###
###                           ###
#################################


processed_base = '/data/old_trap_processed/spinning/'

input_dict = {}


beadtype = 'bangs5'
# beadtype = 'german7'


def formatter20200727(measString, ind, trial):
    if ind == 1:
        return os.path.join(measString, f'trial_{trial:04d}')
    else:
        return os.path.join(measString + f'_{ind}', f'trial_{trial:04d}')


meas_base = 'bead1/spinning/dds_phase_impulse_'
input_dict['20200727'] = [ formatter20200727(meas_base + meas, ind, trial) \
              for meas in ['lower_dg'] \
              for ind in [1,2] for trial in [0] ]


files_to_plot = [0, 4, 16]

save_dir = '/home/cblakemore/plots/libration_paper_2022'
plot_name = 'fig5_libration_spectra_drift_example_v2.svg'

save = True
show = True



####################################
###                              ###
###   PRIOR KNOWLEDGE OF STUFF   ###
###                              ###
####################################

impulse_magnitude = np.pi / 2





#####################################
###                               ###
###   SIGNAL PROCESSING OPTIONS   ###
###                               ###
#####################################

files_for_std = 10
out_cut = 50

### Integration time in seconds for each contiguous file. Should probably
### have this automatically detected eventually
file_length = 10.0





############################
###                      ###
###   PLOTTING OPTIONS   ###
###                      ###
############################

### Adjust empirically as needed for individual datasets

envelope_color = bu.lighten_color('k', 0.2)
# envelope_ls = 'none'
# envelope_lw = 5
# envelope_marker = '.'
# envelope_ms = 7
envelope_ls = '-'
envelope_lw = 4
envelope_marker = ''
envelope_ms = 5

line_alpha = 0.9
plot_lw = 2.5
plot_zorder = 5

### for 20200727, 8Vpp data
plot_freq_width = 5.0
xticks = [1.277e3, 1.278e3, 1.279e3, 1.280e3, 1.281e3]

ylim = [3e-8, 5e-3]

###
lower = 1275
upper = 1281
spacing = 2

xticks = np.linspace(lower, upper, int((upper-lower)/spacing) + 1)
xticklabels = [f'{int(tick):d}' for tick in xticks]

legend_loc = 'upper left'









########################################################################
########################################################################
########################################################################


ncore = 1



def gauss(x, A, mu, sigma, c):
    return A * np.exp(-1.0 * (x - mu)**2 / (2.0 * sigma**2)) + c

def ngauss(x, A, mu, sigma, c, n):
    return A * np.exp(-1.0 * np.abs(x - mu)**n / (2.0 * sigma**n)) + c






def proc_spectra(spectra_file, fig, axarr, axind):

    ### Use the context manager to hold the hdf5 file and copy the data
    ### and measurement parameters from the file
    with h5py.File(spectra_file, 'r') as fobj:

        ### Unpack the hdf5 attributes into local memory
        attrs = {}
        for key in fobj.attrs.keys():
            attrs[key] = fobj.attrs[key]

        nsamp = fobj['all_time'].shape[1]

        ### All times will be relative to this timestamp
        t0 = attrs['file_times'][0]*1e-9

        lib_freqs = attrs['lib_freqs']

        ### Find the impulse file to offset the libration vector (but 
        ### not the libration AMPLITUDE vector)
        impulse_start_file = np.argmax(np.abs(attrs['impulse_vec']))
        if impulse_start_file < files_for_std:
            print()
            print(f'Early impulse found in: {ringdown_file}')
            print('    NOT COMPUTING THIS ONE')
            print()
            return ringdown_file, np.mean(attrs['phi_dgs']), \
                    np.mean(attrs['drive_amps']), [0.0, 0.0, 0.0, 0.0], \
                    [0.0, 0.0, 0.0, 0.0], 0.0, [0.0, 0.0]
            # exit()

        time_arr = np.copy(fobj['all_time'])
        lib_arr = np.copy(fobj['all_lib'])
        lib_amp_arr = np.copy(fobj['all_lib_amp'])

        for i in range(attrs['nfile']):
            ctime = attrs['file_times'][i]*1e-9 - t0
            time_arr[i,:] += ctime

            if i > impulse_start_file:
                lib_off = impulse_magnitude \
                                * attrs['impulse_vec'][impulse_start_file]
            else:
                lib_off = 0
            lib_arr[i,:] += lib_off

    ### Cropping indices to remove Gibbs phenomena
    cut_inds = [False for i in range(out_cut)] + \
               [True for i in range(nsamp - 2*out_cut)] + \
               [False for i in range(out_cut)]
    cut_inds_all = [ cut_inds ] * attrs['nfile']

    cut_inds = np.array(cut_inds)
    cut_inds_all = np.array(cut_inds_all)

    fsamp = 1.0 / (time_arr[0,1] - time_arr[0,0])

    dt = time_arr[0,1] - time_arr[0,0]
    freqs = np.fft.rfftfreq(nsamp, d=dt)
    fac = bu.fft_norm(nsamp, fsamp)

    colors = bu.get_colormap(len(files_to_plot), cmap='plasma')
    lib_freqs = []
    for ind, file in enumerate(files_to_plot):

        fft = np.fft.rfft(lib_arr[file,:])
        asd = np.abs(fft)
        lib_freq = freqs[np.argmax(asd)]

        lib_freqs.append(lib_freq)

        plot_label = ''

        axarr[axind].loglog(freqs, (fac*asd)**2, color=colors[ind], \
                            alpha=line_alpha, zorder=plot_zorder, \
                            lw=plot_lw, label=plot_label)
        # axarr[axind].axvline(lib_freqs[i], color=colors[i])
        # ax.semilogy(freqs-lib_freq, fac*asd, color=colors[i], alpha=plot_alpha, \
        #             zorder=plot_zorder, lw=plot_lw)

    long_freqs = np.fft.rfftfreq(int(nsamp*impulse_start_file), d=dt)
    long_arr = lib_arr[:impulse_start_file,:].flatten()
    long_fac = bu.fft_norm(nsamp*impulse_start_file, fsamp)

    all_label = f'{impulse_start_file*10:d}-s integration'
    axarr[axind].loglog(long_freqs, (long_fac*np.abs(np.fft.rfft(long_arr)))**2, \
                        color=envelope_color, ls=envelope_ls, ms=envelope_ms, \
                        lw=envelope_lw, marker=envelope_marker, \
                        zorder=4, label=all_label)

    return t0




##### Build up the file paths
spectra_file_paths = []
for date in input_dict.keys():
    for meas in input_dict[date]:
        new_filename = os.path.join(processed_base, date, meas+'.h5')
        if os.path.exists(new_filename):
            spectra_file_paths.append( new_filename )
            print(new_filename)


nfile = len(spectra_file_paths)
fig, axarr = plt.subplots(nfile,1, figsize=(6.5,2.75*nfile+1), \
                          sharex=True, sharey=True)
if nfile == 1:
    axarr = [axarr]


times = []
for file_ind, filename in enumerate(spectra_file_paths):
    tstart = proc_spectra(filename, fig, axarr, file_ind)
    times.append(tstart)

delay = (times[-1] - times[0]) / 3600.0


for file_ind in range(nfile):
    axarr[file_ind].set_ylim(*ylim)
    axarr[file_ind].grid(which='major', axis='y')
    # axarr[file_ind].set_ylabel('Libration ASD [rad/$\\sqrt{\\rm Hz}$]')

fig.supylabel('       Libration PSD [rad$^2$/Hz]', fontsize=14)

axarr[0].set_xlim(lower - 0.2*spacing, upper + 0.2*spacing)

axarr[0].set_xticks(xticks)
axarr[0].set_xticklabels(xticklabels)
axarr[0].set_xticks([], minor=True)
axarr[0].set_xticklabels([], minor=True)
axarr[0].legend(loc=legend_loc, fontsize=12, framealpha=1.0)
axarr[-1].set_xlabel('Frequency [Hz]')

# axarr[-1].text(1279.5, 10e-4, f'Same MS, same $E$-field,\n{delay:0.1f} hours later', \
#                ha='center', va='center', fontsize=14)
axarr[-1].text(1279.5, 3.16e-4, f'{delay:0.1f} hours later', \
               ha='center', va='center', fontsize=14)

fig.tight_layout()
fig.subplots_adjust(left=0.15)

if save:
    fig.savefig(os.path.join(save_dir, plot_name))

if show:
    plt.show()









