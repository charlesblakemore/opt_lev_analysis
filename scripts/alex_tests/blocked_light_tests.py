import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import os
import glob
import matplotlib.mlab as ml
import sys
sys.path.append("../microgravity")
from scipy.optimize import minimize_scalar as ms
import alex_limit_v2 as al2
from scipy.optimize import curve_fit
import matplotlib
reload(al2)
#################################################################################testing code
dat_dir = "/data/20180404/bead2/scatt_light_tests_20180419/pinhole_lens_tube_initial_freq_sweep2"


files = bu.sort_files_by_timestamp(bu.find_all_fnames(dat_dir))


def proc_dir(files, T = 10., G = 15., tuning = .14):
    T = 10.
    gf = al2.GravFile()

    gf.load(files[0])
    amps = np.zeros((len(files), 3, gf.num_harmonics))
    delta_f = np.zeros(len(files))
    phis = np.zeros((len(files), 3, gf.num_harmonics))
    sig_amps = np.zeros((len(files), 3, gf.num_harmonics))
    sig_phis = np.zeros((len(files), 3, gf.num_harmonics))
    ps = np.zeros(len(files))
    n = len(files)
    for i, f in enumerate(files[:-1]):
        bu.progress_bar(i, n)
        gf_temp = al2.GravFile()
        gf_temp.load(f)
        gf_temp.estimate_sig(use_diag = True)
        N = len(gf_temp.freq_vector)
        amps[i, :, :] = gf_temp.amps_at_harms/N
        phis[i, :, :] = gf_temp.phis_at_harms
        delta_f[i] = np.mean(gf_temp.electrode_data[3, :])*G*tuning
        sig_amps[i, :, :] = gf_temp.noise/(N*np.sqrt(2))
        sig_phis[i, :, :] = gf_temp.sigma_phis_at_harms
        ps[i] = float(gf_temp.pressures['pirani'])

    def line(x, m, b):
        return m*x + b

    fnum = np.arange(len(files))
    popt, pcov = curve_fit(line, fnum, ps)
    pfit = line(fnum, *popt)
    tarr = T*np.arange(len(files))

    return {"amps [N]": amps, "sig_amps [N]":sig_amps, "phis [rad]": phis, \
            "sig_phis [rad]": sig_phis, "p [Torr]": pfit, "t [s]": tarr,\
            "delta_f [GHz]": delta_f}  

    
def plot(data_dict, plot_harmonics = 3, xkey = "delta_f [GHz]", \
         ykey = "amps [N]", sigkey = "sig_amps [N]", direction = 0,\
         num_harmonics = 10):
    for i in range(plot_harmonics):
        plt.errorbar(data_dict[xkey][:-1], data_dict[ykey][:-1, direction, i],\
                     data_dict[sigkey][:-1, direction, i],\
                     fmt = 'o-', label = "harmonic " + str(i))
    plt.xlabel(xkey)
    plt.ylabel(ykey)
    plt.legend()
    

def plot_spectrum(data_dict, plot_harmonics = 3, direction = 0,\
         num_harmonics = 10, NFFT = 1000, lab = ""):
    for i in range(plot_harmonics):
        GHz = data_dict["delta_f [GHz]"]
        amps = data_dict["amps [N]"][:-1, direction, i]
        fsamp = 1./(GHz[1]-GHz[0])
        psd, freqs = matplotlib.mlab.psd(amps, Fs = fsamp, NFFT = NFFT,\
                detrend = matplotlib.mlab.detrend_linear)
        plt.plot(freqs*0.299, psd, label = "harmonic " + str(i) + lab)
    plt.xlabel("length scale")
    plt.ylabel("PSD N^2/s")
    plt.legend()

