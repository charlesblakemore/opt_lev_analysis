import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import os
import glob
import matplotlib.mlab as ml
import sys
sys.path.append("microgravity")
from scipy.optimize import minimize_scalar as ms
import alex_limit_v2 as al2
from scipy.optimize import curve_fit
reload(al2)
#################################################################################testing code
dat_dir = "/data/20180404/laser_freq_sweep_overnight"
files = bu.sort_files_by_timestamp(bu.find_all_fnames(dat_dir))[1000:2000]
T = 10.
gf = al2.GravFile()

gf.load(files[0])
amps = np.zeros((len(files), 3, gf.num_harmonics))
vdet = np.zeros(len(files))
phis = np.zeros((len(files), 3, gf.num_harmonics))
sig_amps = np.zeros((len(files), 3, gf.num_harmonics))
sig_phis = np.zeros((len(files), 3, gf.num_harmonics))
ps = np.zeros(len(files))
n = len(files)
for i, f in enumerate(files[:-1]):
    bu.progress_bar(i, n)
    gf_temp = al2.GravFile()
    gf_temp.load(f)
    gf_temp.estimate_sig(use_diag = False)
    amps[i, :, :] = gf_temp.amps_at_harms
    phis[i, :, :] = gf_temp.phis_at_harms
    vdet[i] = np.mean(gf_temp.electrode_data[3, :])
    sig_amps[i, :, :] = gf_temp.noise/np.sqrt(2)
    sig_phis[i, :, :] = gf_temp.sigma_phis_at_harms
    ps[i] = float(gf_temp.pressures['pirani'])
    N = len(gf_temp.freq_vector)

def line(x, m, b):
    return m*x + b


fnum = np.arange(len(files))

popt, pcov = curve_fit(line, fnum, ps)

pfit = line(fnum, *popt)
tarr = T*np.arange(len(files))

for i in range(gf_temp.num_harmonics-5):
    plt.errorbar(tarr, amps[:, 0, i]/N, sig_amps[:, 0, i]/N, fmt = 'o-', label = "harmonic " + str(i))
plt.xlabel("time[10s]")
plt.ylabel("amplitude [N]")
plt.legend()
plt.show()


for i in range(gf_temp.num_harmonics-5):
    plt.errorbar(tarr, phis[:, 0, i], sig_phis[:, 0, i], fmt = 'o-', label = "harmonic " + str(i))
plt.xlabel("time[10s]")
plt.ylabel("phase [rad]")
plt.legend()
plt.show()

plt.hist(phis[:, 0, 2], bins = 'auto')
plt.show()
