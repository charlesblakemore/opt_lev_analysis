import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import curve_fit
from amp_ramp_3 import flattop
from amp_ramp_3 import hp_filt
from amp_ramp_3 import bp_filt
from ring_down_analysis_v3 import track_frequency
from plot_phase_vs_pressure_many_gases import build_full_pressure
import bead_util_funcs as buf
import bead_util as bu
import hs_digitizer as hd
import os 

matplotlib.rcParams['agg.path.chunksize'] = 10000

save = True

overwrite = True

#fils = ['/data/old_trap/20191105/bead4/phase_mod/change_dg/neg_2_0_to_0_7/']
fils = ['/data/old_trap/20191105/bead4/phase_mod/change_press/change_press_2/']
out_paths = ['/home/dmartin/Desktop/analyzedData/20191105/changing_pm_freq_7_8/']


files, zero = bu.find_all_fnames(fils[0])

files = files[0:]

tabor_fac = 100.
spinning_freq = 25e3
pm_bandwidth = 200
drive_pm_freq = 330


def lorentzian(x, A, x0, g, B):
    return A * (1./ (1 + ((x - x0)/g)**2)) + B

def gauss(x, A, mean, std):
    return  A * np.exp(-(x-mean)**2/(2.*std**2))

def sqrt(x, a, b):
    return np.sqrt(b**2. - a*x**2.)

def sine(x, A, f, c):
    return A * np.sin(2.*np.pi*f*x + c)
def line(x,a,b):
    return a * x + b

def extract_libration_freq(obj):
    Ns = obj.attribs['nsamp']
    Fs = obj.attribs['fsamp']
    freqs = np.fft.rfftfreq(Ns,1./Fs)

    spin_sig = obj.dat[:,0]
    
    fft = np.fft.rfft(spin_sig)

    #plt.loglog(freqs,np.abs(fft))
    #plt.show()

    z = ss.hilbert(spin_sig)
    phase = ss.detrend(np.unwrap(np.angle(z)))
  
    fft_phase = np.fft.rfft(phase)

    #plt.loglog(freqs,np.abs(fft_phase))
    #plt.show()

    phase_filt = bp_filt(phase, 300, Ns, Fs,300)

    fft_phase = np.fft.rfft(phase_filt)
   
    #plt.loglog(freqs, np.abs(fft_phase))
    #plt.show()

    #plt.plot(phase_filt)
    #plt.show()
   
    z_phase = ss.hilbert(phase_filt)
    
    inst_freq = np.diff(np.unwrap(np.angle(z_phase)))/(2*np.pi) * Fs
    
    samples = np.arange(Ns-1)
    
    cut = (samples > 700) & (samples < Ns-1000)   
    inst_freq = inst_freq[cut]
    
    inst_freq_mean = np.mean(inst_freq)
    print(inst_freq_mean)
    

    #plt.plot(inst_freq)
    #plt.show()
    
    #plt.plot(phase_filt, label=r'$\phi$')
    #plt.plot(pm_amp, label=r'Envelope of $\phi$')
    #plt.ylabel(r'Amplitude')
    #plt.xlabel('Sample Number')
    #plt.legend()
    #plt.show()

    #plt.loglog(freqs,np.abs(fft_phase))
    #plt.show()

    libration_freq = inst_freq_mean

    return libration_freq
    
pressures = np.zeros((len(files), 3))
libration_freqs = np.zeros(len(files))

for i in range(len(files)):
    obj = hd.hsDat(files[i])
    
    pressures[i,:] = obj.attribs["pressures"]
    libration_freqs[i] = extract_libration_freq(obj)  
   
    
func, press = build_full_pressure(pressures, plot=True)
    
data = np.array([press,libration_freqs])

popt, pcov = curve_fit(sqrt, press, libration_freqs)
print(popt)

plt.plot(press, sqrt(press, *popt))
plt.plot(press, libration_freqs)
plt.show()

print(data)
if raw_input('save?') == '1':
    np.save('/home/dmartin/Desktop/analyzedData/20191105/phase_mod/change_press/change_press_2/change_press_2.npy', data)
