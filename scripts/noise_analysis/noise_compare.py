###Script to compare the x position ASD from different files and generate a plot
import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import dill as pickle
import configuration
from matplotlib.mlab import psd
import scipy.constants as constants

#define parameters global to the noise analysis
f0 = 220. #Hz
dat_column = 0
NFFT = 2**14
cal_file ='/calibrations/step_cals/step_cal_20170903.p' 
fs = np.array([.05, 2500])#array to support multiplication
beam_params = {"p1": 0.025, "p2":0.0011, "w1":0.0037, "w2": 0.0025, "xi": 0.34, "d":0.025}


#Put filenames into dict with plot label as key
bead_files= {"MS":"/data/20170903/bead1/turbombar_xyzcool_discharged.h5", "No MS": "/data/20170918/noise_tests/no_bead.h5", "DAQ": "/data/20170918/noise_tests/x_y_terminated.h5", "Electronics": "/data/20170918/noise_tests/all_beams_blocked.h5"}

plt_order= {"MS":0, "No MS": 1, "DAQ": 3, "Electronics": 2}

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
    


def plt_df(df, cf, lab):
    '''plots the x position ASD for DataFile with '''
    psd_dat, freqs = psd(cf*df.pos_data[:, dat_column], Fs = df.fsamp, NFFT = NFFT)
    plt.loglog(freqs, np.sqrt(psd_dat), label = lab)


def plot_shot_noise(beam_params, f0 = f0, label = "shot noise limit"):
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
    plt.plot(fs, np.sqrt(Sxx)*np.ones_like(fs), label = label)
    
#define key to plot in desired order
key = lambda kk: plt_order[kk]
#get calibration constant.
cf = cal_constant(f0)

#plot noise spectra
keys = bead_files.keys()
keys.sort(key = key)
for k in keys:
    df = bu.DataFile()
    df.load(bead_files[k])
    plt_df(df, cf, k)

plot_shot_noise(beam_params)
plt.legend()
plt.xlim(fs[0], fs[1])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Spectral density [$m/\sqrt{Hz}$]") 
plt.show()
