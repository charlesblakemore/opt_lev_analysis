import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re
import matplotlib
import bead_util_funcs as buf
import fnmatch

save = False

#path = "/daq2/20190514/bead1/spinning/wobble/manual_data"
#out_path = "/home/dmartin/analyzedData/20190514/amp_ramp/"

#out_base_fname = "amp_ramp_50k"
#files = glob.glob(path + "/*.h5")

path = "/daq2/20190626/bead1/spinning/wobble/wobble_many_slow"
files, zero = buf.find_all_fnames(path)

print(fnmatch.filter(files, '*0004*.h5'))

sub_files = fnmatch.filter(files,'*0004*')

obj = hsDat(sub_files[0])

freqs = np.fft.rfftfreq(obj.attribs['nsamp'], 1./obj.attribs['fsamp'])

fft = np.fft.rfft(obj.dat[:,0])
plt.plot(freqs,np.abs(fft))
plt.show()

'''
fc = 1e5
bw = 1e3
init_file = 0
final_file = len(files)
n_file = final_file-init_file
ns = 1

obj0 = hsDat(files[init_file])
t0 = obj0.attribs['time']
Ns = obj0.attribs['nsamp']
Fs = obj0.attribs['fsamp']
freqs = np.fft.rfftfreq(Ns, d = 1./Fs)
tarr0 = np.linspace(0, Ns/Fs, Ns)
freq_bool = np.abs(freqs-fc)>bw
d_amps = np.zeros(n_file)
f_wobs = np.zeros(n_file)
f_wob = 427.39
bwa = 10.
sbw = 0.5
'''



def line(x, m, b):
    return m*x + b

def dec2(arr, fac):
    return ss.decimate(ss.decimate(arr, fac), fac)

'''

matplotlib.rcParams.update({'font.size':12})
f, ax = plt.subplots(dpi = 200)

d_amps = np.empty(len(files))

for i, f in enumerate(files[init_file:final_file:ns]):
    print(f)
    try:
        
        obj = hsDat(f)
        print(obj.attribs)
        regex = re.compile('(?<=ta/)\d(\_\d*)?') #find drive amplitude from file name with underscore included or not
        match = regex.search(f)
        drive = match.group(0)
	
        if 1 == match.group(0).find('_'): #if the drive amp has an underscore, replace by a period
             drive = drive.replace('_','.')
        d_amps[i] = float(drive)
        #d_amps[i] = obj.attribs["network amp"]
        
        fft = np.fft.rfft(obj.dat[:, 0])
        fft[freq_bool] = 0.

        plt.plot(fft)

        a_sig = ss.hilbert(np.fft.irfft(fft))
        phase = ss.detrend(np.unwrap(np.angle(a_sig)))
        fft_phase = np.fft.rfft(phase)
        
        b_init = np.abs(freqs-f_wob)<bwa
        f_wobi = freqs[b_init][np.argmax(np.abs(fft_phase[b_init]))]
        b_small = np.abs(freqs-f_wobi)<sbw
        f_wob = np.average(freqs[b_small], weights = np.abs(fft_phase[b_small])**2)
        f_wobs[i] = f_wob

        plt.plot(freqs,np.abs(fft))
        plt.yscale('log')
        #plt.plot(np.abs(fft_phase))
		#plt.loglog(freqs, np.abs(fft_phase))
        #plt.xlim([0, bw])
        plt.show()

    
    except IOError:
        print("bad file")


def sqrt_fun(x, poi, toi):
    return np.sqrt(x*poi + toi)

if save:
    np.save(out_path + out_base_fname + "amps", d_amps)
    np.save(out_path + out_base_fname + "wobble_freq", f_wobs)
'''
