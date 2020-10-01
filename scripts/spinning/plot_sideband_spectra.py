import sys, os, time, itertools, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle

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



#############################
### Which data to analyze ###
#############################
# dir_name = '/data/old_trap/20200727/bead1/spinning/sideband_test_1Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/sideband_test_5Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/sideband_test_7Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/amplitude_impulse_1Vpp-to-7Vpp'
# dir_name = '/data/old_trap/20200727/bead1/spinning/wobble_slow_2'
# dir_name = '/data/old_trap/20200727/bead1/spinning/wobble_fast'
# file_inds = (0, 1000)
# file_step = 1

try:
    trial_ind = int(sys.argv[1])
except:
    trial_ind = 1


date = '20200727'
# date = '20200924'

bead = 'bead1'

base = '/data/old_trap/{:s}/{:s}/spinning/'.format(date, bead)


# meas = 'dds_phase_modulation_sweep/trial_0000'
# file_inds = (0, 200)

# meas = 'arb_phase_impulse_many_2/trial_{:04d}'.format(trial_ind)
# file_inds = (7, 46)

# meas = 'dds_phase_impulse_many/trial_{:04d}'.format(trial_ind)
# meas = 'dds_phase_impulse_lower_dg_3/trial_{:04d}'.format(trial_ind)
# meas = 'dds_phase_impulse_low_dg_3/trial_{:04d}'.format(trial_ind)
meas = 'dds_phase_impulse_mid_dg_3/trial_{:04d}'.format(trial_ind)
# meas = 'dds_phase_impulse_high_dg_3/trial_{:04d}'.format(trial_ind)
# file_inds = (12, 36)
file_inds = (0, 15)
file_step = 1


# meas = 'initial_test'
# meas = 'crosstalk_check'
# meas = 'dipole_meas/initial/trial_0000'
# meas = 'dds_phase_impulse_1Vpp_lower_dg/trial_0000'
# file_inds = (0, 100)
# file_step = 3

dir_name = os.path.join(base, meas)


### Filter constants
# fspin = 19000
fspin = 25000
wspin = 2.0*np.pi*fspin
bandwidth = 10000.0

notch_freqs = []
# notch_freqs = [42036.5, 44986.4]
notch_freqs = [49020.3]
notch_qs = []
# notch_qs = [5000.0, 10000.0]
notch_qs = [10000.0]

detrend = True
force_2pi_wrap = True

### Boolean flags for various sorts of plotting (used for debugging usually)
plot_demod = False

### Should probably measure these monitor factors
tabor_mon_fac = 100
#tabor_mon_fac = 100 * (1.0 / 0.95)




#########################
### Plotting behavior ###
#########################
show = False

output_band = (0, 5000)
drive_output_band = (0, 80000)

average_spectra = False

save_spectra = True
processed_base = '/data/old_trap_processed/spinning/{:s}/'.format(date)
spectra_save_path = os.path.join(processed_base, 'dds_feedback_spectra.p')

waterfall = False   # Doesn't do anything if average_spectra = True
waterfall_fac = 0.01
 
### Full spectra plot limits
xlim = (0.5, 5000)
ylim = (3e-4, 5e0)
# ylim = (4e-4, 3e-1)

### Libration zoom plot limits
# xlim2 = (200, 1400)
# xlim2 = (1100, 1500)
# zoom_xticks = [400.0, 450.0, 500.0, 550.0]
# zoom_xticks = [1100.0, 1300.0, 1500.0]
# xlim2 = (1000, 1350)
# zoom_xticks = [1000.0, 1150.0, 1300.0]

xlim2 = (1, 5000)
zoom_xticks = []
# zoom_xticks = [800, 900, 950]

### Limits for drive plot
drive_xlim = (0.5, 80000)
# drive_xlim2 = (15000, 50000)
drive_xlim2 = (17500, 55000)

drive_ylim = (1e-4, 5e4)




###############################
### Image sequence settings ###
###############################
make_image_sequence = False
make_drive_image_sequence = False
show_first = False
label_time = False
# figsize=(8,8)
figsize=(8,5)
drive_figsize=(8,5)
plot_base = '/home/cblakemore/plots/{:s}/spinning/'.format(date)
basepath = os.path.join(plot_base, meas)



#################################
### Feature tracking settings ###
#################################
plot_features = False
plot_drive_features = True
arrowprops = {'width': 5, 'headwidth': 10, 'headlength': 10, 'shrink': 0.05}
feature_base = '/data/old_trap_processed/spinning/feature_tracking/'
# phase_feature_savepath = os.path.join(feature_base, \
#                                 '20200727/arb_phase_impulse_many_2_{:04d}.p'.format(trial_ind))
# drive_feature_savepath = os.path.join(feature_base, \
#                                 '20200727/arb_phase_impulse_many_2_{:04d}_drive.p'.format(trial_ind))
phase_feature_savepath = os.path.join(feature_base, \
                                '{:s}/dds_phase_impulse_many_{:04d}.p'.format(date, trial_ind))
drive_feature_savepath = os.path.join(feature_base, \
                                '{:s}/dds_phase_impulse_many_{:04d}_drive.p'.format(date, trial_ind))




########################################################################
########################################################################
########################################################################

if plot_demod:
    ncore = 1

sideband_basepath = os.path.join(basepath, 'sideband_spectra')

if make_drive_image_sequence:
    drive_basepath = os.path.join(basepath, 'drive_spectra')


date = re.search(r"\d{8,}", dir_name)[0]

files, _ = bu.find_all_fnames(dir_name, ext='.h5', sort_time=True)
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
fac = bu.fft_norm(nsamp, fsamp)

time_vec = np.arange(nsamp) * (1.0 / fsamp)
full_freqs = np.fft.rfftfreq(nsamp, 1.0/fsamp)

out_inds = (full_freqs > output_band[0]) * (full_freqs < output_band[1])
drive_out_inds = (full_freqs > drive_output_band[0]) * (full_freqs < drive_output_band[1])

times = []
for file in files:
    fobj = bu.hsDat(file, load=False, load_attribs=True)
    times.append(fobj.time)
times = np.array(times) * 1e-9
times -= times[0]





def proc_file(file):

    fobj = bu.hsDat(file, load=True)

    vperp = fobj.dat[:,0]
    elec3 = fobj.dat[:,1]

    phi_dg = fobj.attribs['phi_dg']

    inds = np.abs(full_freqs - fspin) < 200.0

    elec3_fft = np.fft.rfft(elec3)
    true_fspin = full_freqs[np.argmax(np.abs(elec3_fft))]
    # true_fspin = np.average(full_freqs[inds], weights=np.abs(elec3_fft)[inds])

    try:
        amp, phase_mod = bu.demod(vperp, true_fspin, fsamp, plot=plot_demod, \
                                  filt=True, bandwidth=bandwidth, \
                                  notch_freqs=notch_freqs, notch_qs=notch_qs, \
                                  tukey=True, tukey_alpha=5.0e-4, \
                                  detrend=detrend, detrend_order=1, harmind=2.0, \
                                  force_2pi_wrap=force_2pi_wrap)
    except:
        phase_mod = 1e-3 * np.random.randn(len(vperp))

    phase_mod_fft = np.fft.rfft(phase_mod)[out_inds] * fac

    drive_fft = elec3_fft[drive_out_inds] * fac * tabor_mon_fac

    return (phase_mod_fft, drive_fft, phi_dg)






results = Parallel(n_jobs=ncore)(delayed(proc_file)(file) for file in tqdm(files))


phase_results = []
drive_results = []
phi_dgs = []
for phase_fft, drive_fft, phi_dg in results:
    phase_results.append(phase_fft)
    drive_results.append(drive_fft)
    phi_dgs.append(phi_dg)

phase_results = np.array(phase_results)
drive_results = np.array(drive_results)
meas_phi_dg = np.mean(phi_dgs)



if save_spectra:

    try:
        spectra_dict = pickle.load(open(spectra_save_path, 'rb'))
    except FileNotFoundError:
        spectra_dict = {}

    if meas_phi_dg not in list(spectra_dict.keys()):
        spectra_dict[meas_phi_dg] = {}
        spectra_dict[meas_phi_dg]['paths'] = []
        spectra_dict[meas_phi_dg]['freqs'] = []
        spectra_dict[meas_phi_dg]['data'] = []

    saved = False
    for pathind, path in enumerate(spectra_dict[meas_phi_dg]['paths']):
        if path == dir_name:
            saved = True
            print('Already saved this one... overwriting!')
            break

    if saved:
        spectra_dict[meas_phi_dg]['data'][pathind] = phase_results
    else:
        spectra_dict[meas_phi_dg]['paths'].append(dir_name)
        spectra_dict[meas_phi_dg]['freqs'].append(full_freqs[out_inds])
        spectra_dict[meas_phi_dg]['data'].append(phase_results)

    pickle.dump(spectra_dict, open(spectra_save_path, 'wb'))




if make_image_sequence:


    if plot_features:
        phase_feature_lists = pickle.load( open(phase_feature_savepath, 'rb') )

    if plot_drive_features:
        drive_feature_lists = pickle.load( open(drive_feature_savepath, 'rb') )

    for i, fft in enumerate(phase_results):

        figname = os.path.join(sideband_basepath, 'image_{:04d}.png'.format(i))
        if i == 0:
            bu.make_all_pardirs(figname)

        fig, axarr = plt.subplots(2, 1, figsize=figsize)

        axarr[0].loglog(full_freqs[out_inds], np.abs(fft), \
                  label='{:d} s'.format(int(times[i])))
        axarr[1].loglog(full_freqs[out_inds], np.abs(fft), \
                  label='{:d} s'.format(int(times[i])))


        # axarr[1].set_title('Zoom on Libration Feature')

        axarr[0].set_xlabel('Frequency [Hz]')
        axarr[1].set_xlabel('Frequency [Hz]')
        axarr[0].set_ylabel('Sideband ASD\n[rad / $\\sqrt{\\rm Hz}$]')
        axarr[1].set_ylabel('Sideband ASD\n[rad / $\\sqrt{\\rm Hz}$]')

        axarr[0].set_xlim(*xlim)
        axarr[1].set_xlim(*xlim2)
        axarr[0].set_ylim(*ylim)
        axarr[1].set_ylim(*ylim)
        axarr[0].legend(loc='upper left')

        if len(zoom_xticks):
            axarr[1].set_xticks(zoom_xticks)
            axarr[1].set_xticks([], minor=True)

        plt.tight_layout()

        if plot_features:
            for j in range(len(phase_feature_lists)):
                feature = phase_feature_lists[j][i]
                freq_ind = np.argmin(np.abs(full_freqs[out_inds] - feature[0]))
                if freq_ind > 10:
                    lower_ind = freq_ind - 10
                else:
                    lower_ind = 0

                try:
                    asdval = np.max( np.abs(fft[lower_ind:freq_ind+10]) )
                except:
                    asdval = feature[1]

                text = '{:0.1f} Hz'.format(feature[0])
                xy = (feature[0], 1.2*asdval)
                xytext = (feature[0], ylim[1]*0.5)
                axarr[0].annotate(text, xy, xytext, arrowprops=arrowprops, \
                                  ha='center', va='center', fontsize=12)
                axarr[1].annotate(text, xy, xytext, arrowprops=arrowprops, \
                                  ha='center', va='center', fontsize=12)

        fig.savefig(figname)

        if show_first and i == 0 and not make_drive_image_sequence:
            plt.show()

        print('Saved: ', figname.split('/')[-1])




        if make_drive_image_sequence:
            drive_fft = drive_results[i]

            drive_figname = os.path.join(drive_basepath, 'image_{:04d}.png'.format(i))
            if i == 0:
                bu.make_all_pardirs(drive_figname)

            drive_fig, drive_axarr = plt.subplots(2, 1, figsize=drive_figsize)

            drive_axarr[0].loglog(full_freqs[drive_out_inds], np.abs(drive_fft), \
                      label='{:d} s'.format(int(times[i])))
            drive_axarr[1].loglog(full_freqs[drive_out_inds], np.abs(drive_fft), \
                      label='{:d} s'.format(int(times[i])))


            # axarr[1].set_title('Zoom on Libration Feature')

            drive_axarr[0].set_xlabel('Frequency [Hz]')
            drive_axarr[1].set_xlabel('Frequency [Hz]')
            drive_axarr[0].set_ylabel('Drive ASD\n[V / $\\sqrt{\\rm Hz}$]')
            drive_axarr[1].set_ylabel('Drive ASD\n[V / $\\sqrt{\\rm Hz}$]')

            drive_axarr[0].set_xlim(*drive_xlim)
            drive_axarr[1].set_xlim(*drive_xlim2)
            drive_axarr[0].set_ylim(*drive_ylim)
            drive_axarr[1].set_ylim(*drive_ylim)
            if label_time:
                drive_axarr[0].legend(loc='upper left')

            plt.tight_layout()

            if plot_drive_features:
                for j in range(len(drive_feature_lists)):
                    drive_feature = drive_feature_lists[j][i]
                    freq_ind = np.argmin(np.abs(full_freqs[drive_out_inds] - drive_feature[0]))
                    if freq_ind > 10:
                        lower_ind = freq_ind - 10
                    else:
                        lower_ind = 0

                    try:
                        drive_asdval = np.max( np.abs(drive_fft[lower_ind:freq_ind+10]) )
                    except:
                        drive_asdval = drive_feature[1]

                    drive_text = '{:0.1f} Hz'.format(drive_feature[0])
                    drive_xy = (drive_feature[0], 1.2*drive_asdval)
                    drive_xytext = (drive_feature[0], drive_ylim[1]*0.2)
                    if j == 0:
                        drive_axarr[0].annotate(drive_text, drive_xy, drive_xytext, \
                                          arrowprops=arrowprops, \
                                          ha='center', va='center', fontsize=12)
                    drive_axarr[1].annotate(drive_text, drive_xy, drive_xytext, \
                                      arrowprops=arrowprops, \
                                      ha='center', va='center', fontsize=12)

            drive_fig.savefig(drive_figname)

            if show_first and i == 0:
                plt.show()

            print('Saved: ', drive_figname.split('/')[-1])

            plt.close(drive_fig)

        plt.close(fig)








elif average_spectra:

    avg_asd = np.sqrt( np.mean(np.abs(phase_results)**2, axis=0) )


    fig, axarr = plt.subplots(2,1,figsize=(10,8))

    title = dir_name.split('/')[-1]
    axarr[0].set_title(title)

    axarr[0].loglog(full_freqs[out_inds], avg_asd)
    axarr[0].set_xlim(*xlim)
    axarr[0].set_ylim(*ylim)
    axarr[0].set_xlabel('Frequency [Hz]')
    axarr[0].set_ylabel('Sideband ASD [rad / $\\sqrt{\\rm Hz}$]')

    axarr[1].loglog(full_freqs[out_inds], avg_asd)
    axarr[1].set_xlim(*xlim2)
    axarr[1].set_ylim(*ylim)
    axarr[1].set_xlabel('Frequency [Hz]')
    axarr[1].set_ylabel('Sideband ASD [rad / $\\sqrt{\\rm Hz}$]')

    if len(zoom_xticks):
        axarr[1].set_xticks(zoom_xticks)
        axarr[1].set_xticks([], minor=True)

    # axarr[1].set_xticklabels([1100.0, 1150.0, 1200.0, 1250.0])

    plt.tight_layout()

    if show:
        plt.show()




else:

    colors = bu.get_color_map(len(phase_results), cmap='plasma')

    plt.figure(figsize=(10,5))

    scale_fac = 1.0
    for i, fft in enumerate(phase_results):
        if waterfall:
            scale_fac *= waterfall_fac

        plt.loglog(full_freqs[out_inds], np.abs(fft)*scale_fac, color=colors[i])

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Sideband ASD [rad / $\\sqrt{\\rm Hz}$]')
    plt.xlim(*xlim2)
    # plt.xlim(1270, 1282)
    # plt.ylim(*ylim)

    plt.tight_layout()

    if show:
        plt.show()



