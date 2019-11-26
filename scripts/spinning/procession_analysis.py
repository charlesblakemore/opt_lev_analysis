import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re
import matplotlib

vpa = 1e5
apw = 0.1
t = 0.5
p0 = 0.001
g = 500
cal = 0.66
wpv = 1./(t*vpa*apw*g)


paths = [ "/data/20181204/bead1/high_speed_digitizer/procession/zhat_myhat_50k_8vpp_1",\
        "/data/20181204/bead1/high_speed_digitizer/procession/zhat_myhat_50k_7vpp_2",\
        "/data/20181204/bead1/high_speed_digitizer/procession/zhat_myhat_50k_6vpp_0",\
        "/data/20181204/bead1/high_speed_digitizer/procession/zhat_myhat_50k_5vpp_2",\
        "/data/20181204/bead1/high_speed_digitizer/procession/zhat_myhat_50k_4vpp_1",\
        "/data/20181204/bead1/high_speed_digitizer/procession/zhat_myhat_50k_3vpp_0"]


sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3]) 

def get_files(path):
    files = glob.glob(path + "/*.h5")
    files.sort(key = sfun)
    return files

files = list(map(get_files, paths))




def line(x, m, b):
    return m*x + b

def dec2(arr, fac):
    return ss.decimate(ss.decimate(arr, fac), fac)

def sqrt_fun(x, a):
    return a*np.sqrt(x)

def proc_files(files_i):
    obj0 = hsDat(files_i[0])
    Ns = obj0.attribs['nsamp']
    Fs = obj0.attribs['fsamp']
    freqs = np.fft.rfftfreq(Ns, d = 1./Fs)
    fft = np.abs(np.fft.rfft(obj0.dat[:, 0]))   
    for f in files_i[1:]:
        obji = hsDat(files_i[i])
        fft += np.abs(np.fft.rfft(obji.dat[:, 0]))
    fft/= len(files)*len(fft)
    return freqs, fft
    



guesses = np.array([3.5, 3.063, 2.58, 2.19, 1.70, 1.27])
bw = 0.2
plt_inds = [0, 2, 4]
plt_indi = 2
lines = np.zeros_like(guesses)
matplotlib.rcParams.update({'font.size':14})
ff, axarr = plt.subplots(len(plt_inds) + 2, 1, figsize = (6,7.5), dpi = 100, sharex = True,  gridspec_kw = {"height_ratios":[10, 10, 10, 1, 10]})
Es = np.array([8., 7., 6., 5., 4., 3.])*cal*50./0.004
text_loc = (1, 50)
labs = ["66.0 kV/m", "57.8 kV/m", "49.5 kV/m", "41.3 kV/m", "33.0 kV/m", "24.8 kV/m"]
for i, f in enumerate(files[::-1]):
    freqs, fft = proc_files(files[i])
    bf = np.abs(freqs-guesses[i])<bw
    OO = freqs[bf][np.argmax(fft[bf])]
    lines[i] = OO
    ind_OO = np.argmin(freqs - OO)
    #ax.axvline(x = lines[i], color = 'k', linestyle = '--')
    if i in plt_inds:
    
        axarr[plt_indi].plot([OO*2.*np.pi], [np.max(fft)*1e6*wpv/p0], "D", markersize = 10, color = "C3")
        axarr[plt_indi].plot(freqs*2.*np.pi, fft*1e6*wpv/p0, label = labs[i], color = "C0")
        axarr[plt_indi].set_yticks([0, 25, 50, 75])
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        axarr[plt_indi].set_xlim([0, 24])
        axarr[plt_indi].set_ylim([0, 80])
        axarr[plt_indi].text(text_loc[0], text_loc[1], labs[i], fontsize = 12)
        #axarr[plt_indi].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0))
        if plt_indi == 1:
            axarr[plt_indi].set_ylabel(r"$P_{\bot}/P_{0}$ [ppm]")
        plt_indi -=1

lines*=2.*np.pi
axarr[-2].axis("off")
ws_plt = np.linspace(0, np.max(lines))
popt, pcov = curve_fit(line, lines, Es)
axarr[-1].plot(lines[plt_inds], Es[plt_inds]/1000, "D", markersize = 10, color = "C3")
axarr[-1].plot(lines,Es/1000,  'o', color = "C2")
axarr[-1].plot(ws_plt, line(ws_plt, *popt)/1000, 'r', label = r"$2.95 \pm 0.06$ (kV/m)/(rad/s)")
axarr[-1].legend(loc = 4, fontsize = 12)
axarr[-3].set_xlabel(r"$\omega$ [rad/s]")
axarr[-1].set_xlabel(r"$\Omega$ [rad/s]")
axarr[-1].set_ylabel(r"$E$ [kV/m]")
axarr[-1].set_ylim([0, 75])
axarr[-3].xaxis.labelpad = 10
plt.subplots_adjust(top = 0.96, bottom = 0.1, left = 0.18, right = 0.92, hspace = 0.3)
plt.show()

ff.savefig("/home/arider/plots/20181221/precession_vs_E.png", dpi = 200)
ff.savefig("/home/arider/plots/20181219/precession_vs_E.png", dpi = 200)
