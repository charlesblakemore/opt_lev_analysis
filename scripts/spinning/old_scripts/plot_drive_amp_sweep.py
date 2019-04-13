import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import bead_util as bu
import copy
from scipy.optimize import curve_fit

path0 = "/data/20180927/bead1/spinning/amp_ramp_20181014_locked_good"
path1 = "/data/20180927/bead1/spinning/amp_ramp_20181014_unlocked_good"
dfreq = 1210.68
bw = 0.25
Ns = 250000
Fs = 5000.
k = 1e-13*(370*2.*np.pi)**2
axis = 0
freqs = np.fft.rfftfreq(Ns, 1./Fs)
f_bool = np.abs(freqs-dfreq)<bw/2.
f_ind = np.argmin(np.abs(freqs - dfreq))
nwind = np.sum(f_bool)
files0 = bu.find_all_fnames(path0)
files1 = bu.find_all_fnames(path1)

def get_sigma(fft, nn = 36):
    rolls = np.arange(-nn/2, nn/2)
    ffts = []
    for r in rolls[rolls != 0]:
        ffts.append(np.sum(fft[np.roll(f_bool, r*nwind)]))
    return np.mean(np.abs(ffts))/np.sqrt(2)

def get_amp_phase(arr, make_plot = False):
    fft = np.fft.rfft(ss.detrend(arr))*2./Ns
    fft_line = copy.deepcopy(fft)
    fft_line[np.logical_not(f_bool)] = 0.
    line_ft = np.sum(fft_line)
    if make_plot:
        plt.loglog(freqs, np.abs(fft))
        plt.loglog(freqs[f_bool], np.abs(fft[f_bool]))
        plt.loglog(freqs[f_ind], np.abs(fft[f_ind]), 'o')
        plt.show()
    return np.abs(line_ft), np.angle(line_ft), get_sigma(fft) 

adl = []
pdl = []
adul = []
pdul = []


arl = []
prl = []
sal = []
arul = []
prul = []
saul = []


df = bu.DataFile()
df.load(files0[0])
df.diagonalize()
cf = df.conv_facs[axis]
for i, f in enumerate(files0):
    df.load(f)
    df.load_other_data()
    ad, pd, sad = get_amp_phase(df.other_data[2])
    ar, pr, sar = get_amp_phase(df.pos_data[axis]*cf/k)
    adl.append(ad)
    pdl.append(pd)
    arl.append(ar)
    prl.append(pr)
    sal.append(sar)


for i, f in enumerate(files1):
    df.load(f)
    df.load_other_data()
    ad, pd, sad = get_amp_phase(df.other_data[2])
    ar, pr, sar = get_amp_phase(df.pos_data[axis]*cf/k)
    adul.append(ad)
    pdul.append(pd)
    arul.append(ar)
    prul.append(pr)
    saul.append(sar)

def ffun(e, k, q0):
    return np.arcsin(k/e) + q0

e_field_cal = 200./0.004
phil = np.array(prl)-np.array(pdl)
phil[phil<0.]+=2.*np.pi
phiul = np.array(prul)-np.array(pdul)
phiul[phiul<0.]+=2.*np.pi
plt.errorbar(np.array(adl)*e_field_cal, phil, np.array(sal)/np.array(arl), fmt = 'o', label = "first sweep")
plt.errorbar(np.array(adul)*e_field_cal, phiul, np.array(saul)/np.array(arul), fmt = 'o', label = "second sweep")
plt.xlabel("Drive Electric Field Amplitude [V/m]")
plt.ylabel("Relative Response Phase [rad]")
plt.legend()
plt.show()
