import sys, os, re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
import dill as pickle

import bead_util as bu

plt.rcParams.update({'font.size': 14})

arg = sys.argv[1]

base_result_path = '/home/cblakemore/tmp/'
filenames = [\
             '20200320_mod_grav_binning.p', \
             # '20200320_mod_grav_rand1_binning.p', \
             # '20200320_mod_grav_rand2_binning.p', \
             # '20200320_mod_grav_rand3_binning.p', \
             # '20200320_mod_grav_far_binning.p', \
             # '20200320_mod_grav_far_rand1_binning.p', \
             # '20200320_mod_grav_far_rand2_binning.p', \
             # '20200320_mod_grav_far_rand3_binning.p', \
             # 'signal_injection_stbin_{:d}_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin_{:d}_rand1_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin_{:d}_rand2_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin_{:d}_rand3_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin2_{:d}_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin2_{:d}_rand1_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin2_{:d}_rand2_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin2_{:d}_rand3_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin3_{:d}_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin3_{:d}_rand1_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin3_{:d}_rand2_binning.p'.format(int(arg)), \
             # 'signal_injection_stbin3_{:d}_rand3_binning.p'.format(int(arg)), \
            ]

plot_base = '/home/cblakemore/plots/20200320/mod_grav/'
# plot_name = '20200320_far_binning_results_naive.svg'


signal_level = 0
# signal_level = 4.0e8
# signal_level = 5.0e8
# signal_level = 5.5e8

plot_name = '20200320_binning_results_naive.svg'
linthresh = 4e7
ylim = (3.9e7, 1e9)

# plot_name = 'signal_injection_stbin2_{:d}_binning_results_all.svg'.format(int(arg))
# linthresh = 4e7
# ylim = (3.9e7, 2e9)

# plot_name = 'signal_injection_stbin2_{:d}_binning_results_all_zoom.svg'.format(int(arg))
# linthresh = 2e8
# ylim = (1.9e8, 1e9)

show = False
save = True

# linthresh = 1e8
# ylim = (9.5e7, 2e9)


# freqs_to_plot = [6, 12, 18, 36]
freqs_to_plot = [6, 12, 18, 21, 30, 33, 36, 39]

file_length = 10.0  # Length of an individual integration to scale the xaxis

########################################################################################
########################################################################################
########################################################################################



figname = os.path.join(plot_base, plot_name)

# linestyles = ['-']
linestyles = ['-', '--', ':',  '-.']

binning_result0 = pickle.load( open(os.path.join(base_result_path, filenames[0]), 'rb') )
dict_keys0 = list(binning_result0.keys())

freqs = binning_result0['freqs']
colors = bu.get_color_map(len(freqs), cmap='plasma')
colors = bu.get_color_map(len(freqs_to_plot), cmap='plasma')


fig, axarr = plt.subplots(2,1,sharex=True,figsize=(10,6))

axarr[0].xaxis.set_tick_params(which='both', labelbottom=False)

min_pos_lim = 0
min_neg_lim = 0

for fileind, filename in enumerate(filenames):

    binning_result = pickle.load( open(os.path.join(base_result_path, filename), 'rb') )
    dict_keys = list(binning_result.keys())

    binning_nums = []
    for key in dict_keys:
        if 'limit_by_harm' in key:
            binning_nums.append( int(re.findall('[0-9]+', key)[0]) )

    binning_nums.sort()
    binning_nums = np.array(binning_nums)

    bin_lengths = binning_nums * file_length

    limit_arr = np.zeros( (len(freqs), len(binning_nums), 2) )

    for ind, binning_num in enumerate(binning_nums):
        c_lim_arr = binning_result[str(binning_num)+'_limit_by_harm']
        limit_arr[:,ind,:] = c_lim_arr[2,0,:,:]

    i = 0
    for freqind, freq in enumerate(freqs):
        if int(freq) not in freqs_to_plot:
            continue
        color = colors[i]
        i += 1
        if fileind == 0:
            label = '{:d} Hz'.format(int(freq))
        else:
            label = ''

        # if int(freq) == 21:
        #     print(limit_arr[freqind,:,0])
        #     input()

        zero_pos_lim = limit_arr[freqind,:,0] == 0.0
        zero_neg_lim = limit_arr[freqind,:,1] == 0.0

        axarr[0].plot(bin_lengths, limit_arr[freqind,:,0] + zero_pos_lim*ylim[1]**2, \
                      color=color, label=label, ls=linestyles[fileind])
        axarr[1].plot(bin_lengths, limit_arr[freqind,:,1] - zero_neg_lim*ylim[1]**2, \
                      color=color, label=label, ls=linestyles[fileind])

if signal_level:
    if signal_level > 0:
        axis = 0
    elif signal_level < 0:
        axis = 0
    axarr[axis].axhline(signal_level, lw=2, color='k', ls='--', alpha=0.6)

ax0_ylim = axarr[0].get_ylim()
ax1_ylim = axarr[1].get_ylim()

axarr[0].set_xscale('log')
axarr[0].set_yscale('symlog', linthreshy=linthresh)
axarr[0].yaxis.set_minor_locator(bu.MinorSymLogLocator(linthresh))

axarr[1].set_xscale('log')
axarr[1].set_yscale('symlog', linthreshy=linthresh)
axarr[1].yaxis.set_minor_locator(bu.MinorSymLogLocator(linthresh))

axarr[0].set_ylim(ylim[0], ylim[1])
axarr[1].set_ylim(-ylim[1], -ylim[0])

axarr[0].grid(axis='y')
axarr[1].grid(axis='y')

axarr[1].legend(fontsize=12, ncol=4, loc='lower right')

axarr[0].set_ylabel('$\\alpha_{95}$ [abs]')
axarr[1].set_ylabel('$\\alpha_{95}$ [abs]')
axarr[1].set_xlabel('Length of an individual bin [s]')

fig.tight_layout()

if save:
    fig.savefig(figname)

if show:
    plt.show()

plt.close(fig)

