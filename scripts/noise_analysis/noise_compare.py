###Script to compare the x position ASD from different files and generate a plot
import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import dill as pickle
import configuration
from matplotlib.mlab import psd
import scipy.constants as constants
import scipy.signal as signal

#define parameters global to the noise analysis
f0 = 210. #Hz
dat_column = 0
z_dat_column = 2
NFFT = 2**12
cal_file ='/calibrations/step_cals/step_cal_20170903.p' 
fs = np.array([1, 2500])#array to support multiplication
beam_params = {"p1": 0.025, "p2":0.0011, "w1":0.0037, "w2": 0.003, "xi": 0.34, "d":0.025}
lamb = 1064e-9


#Put filenames into dict with plot label as key
bead_files= {"MS":"/data/20170903/bead1/turbombar_xyzcool_discharged.h5", "No MS": "/data/20170918/noise_tests/no_bead.h5", "DAQ": "/data/20170918/noise_tests/x_y_terminated.h5", "Electronics": "/data/20170918/noise_tests/all_beams_blocked.h5"}

#bead_files= {"MS":"/data/20170903/bead1/turbombar_xyzcool_discharged.h5", "No MS": "/data/20170918/noise_tests/no_bead.h5", "No MS2": "/data/20170903/no_bead/grav_data/manysep_h0-20um/turbombar_xyzcool_discharged_stage-X0um-Y40um-Z9um_Ydrive40umAC-13Hz_0.h5", "DAQ": "/data/20170918/noise_tests/x_y_terminated.h5", "Electronics": "/data/20170918/noise_tests/all_beams_blocked.h5"}

z_bead_files= {"MS":"/data/20170903/bead1/turbombar_xyzcool_discharged.h5", "Back-Refl": "/data/20171002/noise_tests/back_reflect_vacuum.h5", "Electronics": "/data/20171002/noise_tests/synth500kHz.h5", "DAQ": "/data/20171002/noise_tests/synth50kHz_intoFPGA.h5"}

plt_order= {"MS":0, "No MS": 1, "DAQ": 3, "Electronics": 2}
#plt_order= {"MS":0, "No MS": 1, "No MS2": 4, "DAQ": 3, "Electronics": 2}

z_plt_order= {"MS":0, "Back-Refl":1, "DAQ": 3, "Electronics": 2}


def spring_k(f0):
    '''computes spring constant from resonant frequency f0 and mass derived from parameters in configuration.p_param'''
    p_param = configuration.p_param
    m = (4./3.)*np.pi*p_param['bead_radius']**3*p_param['bead_rho']
    return m*(2.*np.pi*f0)**2



def cal_constant(f0, cal_file = cal_file):
    '''derives m/V calibration constant from a charge step calibratioon file and resonant frequency'''
    vpn = pickle.load(open(cal_file, 'rb'))[0][0]
    k = spring_k(f0)
    return 1./(vpn*k)


def plt_df(ax, df, cf, lab, ls='-', marker='.'):
    '''plots the x position ASD for DataFile with '''
    psd_dat, freqs = psd(cf*signal.detrend(df.pos_data[:, dat_column]), \
                         Fs = df.fsamp, NFFT = NFFT)
    ax.loglog(freqs, np.sqrt(psd_dat), label = lab, ls=ls, marker=marker, \
              markersize=0)

def plt_zdf(ax, df, cf, lab, ls='-', marker='.'):
    '''plots the x position ASD for DataFile with '''
    psd_dat, freqs = psd(cf*signal.detrend(df.pos_data[:, z_dat_column]), \
                         Fs = df.fsamp, NFFT = NFFT)
    ax.loglog(freqs, np.sqrt(psd_dat), label = lab, ls=ls, marker=marker, \
              markersize=0)


def plot_shot_noise(ax, beam_params, f0 = f0, label = "Shot Noise Limit"):
    '''Plots shot noise limit from eq (4) in heterodyne paper.'''
    bp = beam_params
    c = constants.c
    e = constants.e
    k = spring_k(f0)
    num = np.pi*e*bp["p2"]*\
        (bp["w1"]**2 + bp["w2"]**2)**3*\
        (bp["p1"] + bp["p2"])
    denom = 8*bp["xi"]*bp["p1"]*bp["w1"]**4*\
        k**2*bp['d']**2*c**2 

    Sxx = num/denom
    print np.sqrt(Sxx)
    ax.plot(fs, np.sqrt(Sxx)*np.ones_like(fs), label = label)

def plot_z_shot_noise(ax, beam_params, label = "Shot Noise Limit"):
    '''Plots shot noise limit from eq (5) in heterodyne paper.'''
    bp = beam_params
    c = constants.c
    e = constants.e
    prefac = (lamb / (4 * np.pi))**2
    num = 2 * e * (bp['p1'] + bp['p2'])
    denom = bp['xi'] * bp['p1'] * bp['p2']

    Sxx = prefac * num / denom
    print np.sqrt(Sxx)
    ax.plot(fs, np.sqrt(Sxx)*np.ones_like(fs), label = label)
    

f, axarr = plt.subplots(1,2,sharex=True,sharey=True)

#styles = ['None', 'None', 'None', 'None']
styles = ['-', '-', '-', '-']
#styles = ['-', '--', '-.', ':']
#markers = ['.', 'o', '^', 's']
markers = ['.', '.', '.', '.']

#define key to plot in desired order
key = lambda kk: plt_order[kk]
#get calibration constant.
cf = cal_constant(f0)

#plot noise spectra
keys = bead_files.keys()
keys.sort(key = key)
for ind, k in enumerate(keys):
    style = styles[ind]
    marker = markers[ind]
    df = bu.DataFile()
    df.load(bead_files[k])
    plt_df(axarr[0], df, cf, k, ls=style, marker=marker)

zkey = lambda kk: z_plt_order[kk]
zcf = 1.0 / (1.3e14)
zcf = zcf / spring_k(159)

zkeys = z_bead_files.keys()
zkeys.sort(key = zkey)
for ind, k in enumerate(zkeys):
    style = styles[ind]
    marker = markers[ind]
    zdf = bu.DataFile()
    zdf.load(z_bead_files[k])
    plt_zdf(axarr[1], zdf, zcf, k, ls=style, marker=marker)

plot_shot_noise(axarr[0], beam_params)
plot_z_shot_noise(axarr[1], beam_params)

#axarr[0].legend(loc=3)
#axarr[1].legend(loc=3)
axarr[0].set_xlim(fs[0], fs[1])
axarr[0].set_xlabel("Frequency [Hz]")
axarr[1].set_xlabel("Frequency [Hz]")
axarr[0].set_ylabel("$\sqrt{S_{xx}}$ [$m/\sqrt{Hz}$]") 
axarr[1].set_ylabel("$\sqrt{S_{zz}}$ [$m/\sqrt{Hz}$]") 
plt.show()

#print spring_k(50)
