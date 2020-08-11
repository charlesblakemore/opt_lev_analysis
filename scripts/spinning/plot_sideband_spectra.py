import os, time, itertools, re, warnings
import numpy as np
import matplotlib.pyplot as plt

from obspy.signal.detrend import polynomial

import bead_util as bu
import peakdetect as pdet

import scipy.optimize as opti
import scipy.signal as signal

from tqdm import tqdm
from joblib import Parallel, delayed
# ncore = 1
ncore = 20

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 14})
np.random.seed(12345)

### Boolean flags for various sorts of plotting (used for debugging usually)
plot_demod = False

### Filter constants
fspin = 30000
wspin = 2.0*np.pi*fspin
bandwidth = 10000.0

# Should probably measure these monitor factors
tabor_mon_fac = 100
#tabor_mon_fac = 100 * (1.0 / 0.95)

output_band = (0, 5000)
average_spectra = True


### Which data to analyze
# dir_name = '/data/old_trap/20200727/bead1/spinning/sideband_test_1Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/sideband_test_5Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/sideband_test_7Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/amplitude_impulse_1Vpp-to-7Vpp'
dir_name = '/data/old_trap/20200727/bead1/spinning/wobble_slow'
# dir_name = '/data/old_trap/20200727/bead1/spinning/wobble_fast'
file_inds = (0, 1000)
file_step = 1

waterfall = True
waterfall_fac = 0.01


make_image_sequence = True
xlim = (0.5, 5000)
xlim2 = (300, 1500)
# xlim2 = (1350, 1475)
ylim = (3e-4, 1e1)
# ylim = (4e-4, 3e-1)
# basepath = '/home/cblakemore/plots/20200727/spinning/amplitude_impulse_1Vpp-to-7Vpp/'
basepath = '/home/cblakemore/plots/20200727/spinning/wobble_sequence_1/'
# basepath = '/home/cblakemore/plots/20200727/spinning/wobble_sequence_2/'


track_features = True
allow_new_features = True
arrowprops = {'width': 5, 'headwidth': 10, 'headlength': 10, 'shrink': 0.05}
feature_savepath = '/data/old_trap_processed/spinning/feature_tracking/20200727/wobble_slow.p'


########################################################################
########################################################################
########################################################################

date = re.search(r"\d{8,}", dir_name)[0]

files, _ = bu.find_all_fnames(dir_name, ext='.h5',sort_time=True)
files = files[file_inds[0]:file_inds[1]:file_step]

Ibead = bu.get_Ibead(date=date)


def sqrt(x, A, x0, b):
    return A * np.sqrt(x-x0) + b

def power_law(x, a, b, c, d):
    return a*(x-b)**c + d

def line(x, a, b):
    return a*x + b



fobj = bu.hsDat(files[0], load=False, load_attribs=True)

nsamp = fobj.nsamp
fsamp = fobj.fsamp

time_vec = np.arange(nsamp) * (1.0 / fsamp)
freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

out_inds = (freqs > output_band[0]) * (freqs < output_band[1])



def proc_file(file):

    fobj = bu.hsDat(file, load=True)

    vperp = fobj.dat[:,0]
    elec3 = fobj.dat[:,1]

    inds = np.abs(freqs - fspin) < 200.0
    elec3_asd = np.abs(np.fft.rfft(elec3))
    true_fspin = np.average(freqs[inds], weights=elec3_asd[inds])

    amp, phase_mod = bu.demod(vperp, true_fspin, fsamp, plot=plot_demod, \
                              filt=True, bandwidth=bandwidth,
                              tukey=True, tukey_alpha=5.0e-4, \
                              detrend=True, detrend_order=1, harmind=2.0)

    phase_mod_fft = np.fft.rfft(phase_mod)

    return phase_mod_fft[out_inds]



results = Parallel(n_jobs=ncore)(delayed(proc_file)(file) for file in tqdm(files))
results = np.array(results)


times = []
for file in files:
    fobj = bu.hsDat(file, load=False, load_attribs=True)
    times.append(fobj.time)
times = np.array(times) * 1e-9
times -= times[0]


fac = bu.fft_norm(nsamp, fsamp)


if track_features:

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Identify Frequencies of Interest')
    ax.loglog(freqs[out_inds], np.abs(results[0])*fac)
    fig.tight_layout()
    plt.show()

    init_str = input('Enter frequencies separated by comma: ')
    init_features = list(map(float, init_str.split(',')))
    # init_features = [24.5, 1386]

    xdat = freqs[out_inds]

    def find_peaks(fft):
        upper_ind = np.argmin(np.abs(freqs - 200.0))
        fft *= fac

        popt, pcov = opti.curve_fit(line, np.log(xdat[1:upper_ind]), \
                                    np.log(np.abs(fft[1:upper_ind])), \
                                    maxfev=10000, p0=[1.0, -1.0])

        if np.abs(popt[0]) > 0.5:
            fft *= 1.0 / (np.exp(popt[1]) * xdat**popt[0])

        peaks = bu.find_fft_peaks(xdat, fft, window=50, lower_delta_fac=2.0, delta_fac=4.0)

        peaks = np.array(peaks)
        if np.abs(popt[0]) > 0.5:
            peaks[:,1] *= (np.exp(popt[1]) * peaks[:,0]**popt[0])

        return peaks


    all_peaks = Parallel(n_jobs=ncore)(delayed(find_peaks)(dataset) for dataset in tqdm(results))

    feature_lists = []
    for init_feature in init_features:
        clist = []
        feature = [init_feature, 0.0]

        for i, peaks in enumerate(all_peaks):
            distances = np.abs(peaks[:,0] - feature[0])

            sorter = np.argsort(distances)
            close_enough = distances[sorter] < 10.0

            peaks_sorted = peaks[sorter,:]
            peaks_valid = peaks_sorted[close_enough,:]

            try:
                feature_ind = np.argmax(peaks_valid[:3,1])
                feature = peaks_valid[feature_ind]

                clist.append(feature)
            except:
                print(feature)
                print(distances)
                clist.append([0.0, 0.0])

        feature_lists.append(clist)

    bu.make_all_pardirs(feature_savepath)
    pickle.dump(open(feature_savepath, 'wb'), feature_lists)






if make_image_sequence:

    for i, fft in enumerate(results):

        figname = os.path.join(basepath, 'image_{:04d}.png'.format(i))
        if i == 0:
            bu.make_all_pardirs(figname)

        fig, axarr = plt.subplots(2, 1, figsize=(8,8))

        axarr[0].loglog(freqs[out_inds], np.abs(fft)*fac, \
                  label='{:d} s'.format(int(times[i])))
        axarr[1].loglog(freqs[out_inds], np.abs(fft)*fac, \
                  label='{:d} s'.format(int(times[i])))


        # axarr[1].set_title('Zoom on Libration Feature')

        axarr[0].set_xlabel('Frequency [Hz]')
        axarr[1].set_xlabel('Frequency [Hz]')
        axarr[0].set_ylabel('Sideband ASD [rad / $\\sqrt{\\rm Hz}$]')
        axarr[1].set_ylabel('Sideband ASD [rad / $\\sqrt{\\rm Hz}$]')

        axarr[0].set_xlim(*xlim)
        axarr[1].set_xlim(*xlim2)
        axarr[0].set_ylim(*ylim)
        axarr[1].set_ylim(*ylim)
        axarr[0].legend(loc='upper left')

        plt.tight_layout()

        if track_features:
            for j, _ in enumerate(init_features):
                feature = feature_lists[j][i]
                freq_ind = np.argmin(np.abs(freqs[out_inds] - feature[0]))
                asdval = np.max( np.abs(fft[freq_ind-10:freq_ind+10])*fac )

                text = '{:0.1f} Hz'.format(feature[0])
                xy = (feature[0], 1.2*asdval)
                xytext = (feature[0], ylim[1]*0.5)
                axarr[0].annotate(text, xy, xytext, arrowprops=arrowprops, \
                                  ha='center', va='center', fontsize=12)
                axarr[1].annotate(text, xy, xytext, arrowprops=arrowprops, \
                                  ha='center', va='center', fontsize=12)

        fig.savefig(figname)
        # plt.show()

        print('Saved: ', figname.split('/')[-1])

        plt.close(fig)


elif average_spectra:

    fig, axarr = plt.subplots(2,1,figsize=(10,8))

    title = dir_name.split('/')[-1]
    axarr[0].set_title(title)

    axarr[0].loglog(freqs[out_inds], np.mean(np.abs(results), axis=0)*fac)
    axarr[0].set_xlim(*xlim)
    axarr[0].set_ylim(*ylim)
    axarr[0].set_xlabel('Frequency [Hz]')
    axarr[0].set_ylabel('Sideband ASD [rad / $\\sqrt{\\rm Hz}$]')

    axarr[1].loglog(freqs[out_inds], np.mean(np.abs(results), axis=0)*fac)
    axarr[1].set_xlim(*xlim2)
    axarr[1].set_ylim(*ylim)
    axarr[1].set_xlabel('Frequency [Hz]')
    axarr[1].set_ylabel('Sideband ASD [rad / $\\sqrt{\\rm Hz}$]')

    plt.tight_layout()
    plt.show()


else:

    colors = bu.get_color_map(results.shape[0], cmap='plasma')

    for i, fft in enumerate(results):
        if waterfall:
            fac *= waterfall_fac

        plt.loglog(freqs[out_inds], np.abs(fft)*fac, color=colors[i])

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Sideband ASD [rad / $\\sqrt{\\rm Hz}$]')

    plt.tight_layout()
    plt.show()



