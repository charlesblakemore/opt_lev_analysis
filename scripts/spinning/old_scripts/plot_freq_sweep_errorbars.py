import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import bead_util as bu
import copy
from scipy.optimize import curve_fit
import re

path0 = "/data/20181025/bead1/spinning/sudden_turn_on_600Hz_good"


#path1 = "/data/20180927/bead1/spinning/amp_ramp_20181014_unlocked_good"
bw = 0.5
Ns = 250000
Fs = 5000.
k = 1e-13*(370*2.*np.pi)**2
axis = 0
freqs = np.fft.rfftfreq(Ns, 1./Fs)
nwind = int(np.floor(bw/(freqs[1]-freqs[0])))
files0 = bu.find_all_fnames(path0)
#files1 = bu.find_all_fnames(path1)

def get_dfreq(fname, darr, min_ind = 10):
    fft = np.fft.rfft(darr)
    fft[np.arange(len(fft))<min_ind] = 0
    #ig = float(re.findall('\d+Hz', fname)[0][:-2])
    #ind_ig = np.argmin(np.abs(freqs-ig))
    #fbool = np.abs(np.arange(len(fft))-ind_ig)>nwind*s_fac
    #fft[fbool] = 0.
    return freqs[np.argmax(np.abs(fft))]


def get_sigma(fft, dfreq, nn = 36):
    fb = np.abs(freqs-dfreq)<bw/2.
    rolls = np.arange(-nn/2, nn/2)
    ffts = []
    for r in rolls[rolls != 0]:
        ffts.append(np.sum(fft[np.roll(fb, r*nwind)]))
    return np.mean(np.abs(ffts))/np.sqrt(2)

def get_amp_phase(arr, dfreq, make_plot = False):
    fft = np.fft.rfft(ss.detrend(arr))*2./Ns
    fft_line = copy.deepcopy(fft)
    f_bool = np.abs(dfreq-freqs)<nwind
    fft_line[np.logical_not(f_bool)] = 0.
    line_ft = np.sum(fft_line)
    if make_plot:
        f_ind = np.argmin(np.abs(freqs-dfreq))
        #plt.loglog(freqs, np.abs(fft))
        plt.loglog(freqs[f_bool], np.abs(fft[f_bool]))
        plt.loglog(freqs[f_ind], np.abs(fft[f_ind]), 'o')
        #plt.show()
    return np.abs(line_ft), np.angle(line_ft), get_sigma(fft, dfreq) 

def plot_fft_blocks(arr, bs = 25000):
    freqplt = np.fft.rfftfreq(bs, d = 1./Fs)
    n = int(np.floor(len(arr)/bs))
    bls = np.arange(len(arr))<bs
    for i in range(n):
        sel = np.roll(bls, bs*i)
        plt.loglog(freqplt, np.abs(np.fft.rfft(arr[sel])), label = str(bs*i/Fs)+"s")


nfiles = len(files0)
dfreqs = np.zeros(nfiles)
adl = np.zeros(nfiles)
pdl = np.zeros(nfiles)
arl = np.zeros(nfiles)
prl = np.zeros(nfiles)
sal = np.zeros(nfiles)
ts = np.zeros(nfiles)

df = bu.DataFile()
df.load(files0[0])
df.diagonalize()
cf = 1.#df.conv_facs[axis]
for i, f in enumerate(files0):
    df.load(f)
    df.load_other_data()
    dfreq = get_dfreq(f, df.other_data[2])
    ad, pd, sad = get_amp_phase(df.other_data[2], dfreq, make_plot = True)
    ar, pr, sar = get_amp_phase(df.pos_data[axis]*cf/k, dfreq)
    dfreqs[i] = dfreq
    adl[i] = ad
    pdl[i] = pd
    arl[i] = ar
    prl[i] = pr
    sal[i] = sar
    ts[i] = df.time
plt.show()

ts = (ts-min(ts))/1E9 #convert to seconds
bt1 = ts<1900.
bt2 = (ts>2100.)*(ts<3774.)

plt.loglog(dfreqs[bt1], arl[bt1], 'o', label = "locked")
plt.loglog(dfreqs[bt2], arl[bt2], 'o', label = "un locked")
plt.show()
