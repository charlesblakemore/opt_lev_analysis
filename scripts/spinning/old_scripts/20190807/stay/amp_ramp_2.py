import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
import re
import matplotlib
from peakdetect import peakdetect as pd

save = False 

#path = "/data/20181205/bead1/high_speed_digitizer/amp_ramp/50k_zhat"
#out_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/ampramp_data_0/"

#path = "/daq2/20190514/bead1/spinning/pramp3/N2/wobble/pre-meas1_repeat"
#out_path = "/home/dmartin/analyzedData/20190514/pramp3/N2/wobble/repeat/"
path = "/daq2/20190626/bead1/spinning/wobble/wobble_many/wobble_0000/"


out_base_fname = "amp_ramp_50k_"
files = glob.glob(path + "/*.h5")

fc = 1e5
bw = 1e3
init_file = 0
final_file = len(files)
n_file = final_file-init_file
ns = 1
#sfun = lambda fname: int(re.findall('\d+.h5', fname)[0][:-3]) 
#files.sort(key = sfun) files = np.array(files)
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


def line(x, m, b):
    return m*x + b

def dec2(arr, fac):
    return ss.decimate(ss.decimate(arr, fac), fac)

#Random flattop window I found (need to include ref). Apparently good for estimation of amplitudes
def flatop(M):
    a0 = 0.21557895
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368
	
    n = np.arange(0,M)
    return a0 - a1*np.cos(2*np.pi*n/(M-1)) + \
           a2*np.cos(4*np.pi*n/(M-1)) - \
           a3*np.cos(6*np.pi*n/(M-1)) + \
           a4*np.cos(8*np.pi*n/(M-1))

def find_spikes(threshold):
	max_ind = []
	max_ind_freq = []

	for i in range(len(fft_phase)):
		#
		diff_left = np.abs(fft_phase[i]-fft_phase[i-4])
		diff_right = np.abs(fft_phase[i]-fft_phase[(i+4)%len(fft_phase)])
		
		if diff_left > threshold and diff_right > threshold:
			#add way to check for multiple spikes
			max_ind.append(i)
			max_ind_freq.append(freqs[i])
			
			print('spike: {}'.format(freqs[i]))
	
		
	return max_ind, max_ind_freq	

def find_phasemod_freq(obj,amp_thresh,lookahead=55,delta=3400):
	fft = np.fft.rfft(obj.dat[:,0])	
	
	fft[freq_bool] = 0
	
	#Take Hilbert tranform of signal to generate the complex part
	hilb = ss.hilbert(np.fft.irfft(fft))
	#Extract the phase which contains information about the phase modulation of the microsphere
	phase = ss.detrend(np.unwrap(np.angle(hilb)))
	fft_phase = np.fft.rfft(phase)
	
	#Max_peaks is really the only useful thing here. This just finds the peaks in the FFT at certain frequencies.
	max_peaks, min_peaks = pd(np.abs(fft_phase),freqs,lookahead,delta)
	

	#plt.plot(freqs,np.abs(fft_phase))	
	for i in range(len(max_peaks)):
		#plt.scatter(max_peaks[i][0],max_peaks[i][1])
		
		#The peak must be greater than 60 Hz (there is sometimes a 53 Hz signal) and the amplitude must be above some threshold
		if max_peaks[i][0] > 60 and max_peaks[i][1] > amp_thresh:
			freq = max_peaks[i][0]
		else:
			freq = 0
	plt.show()
	return freq

def find_drive_amp(obj,bandwidth,plot=False):
	
	psd, freqs = matplotlib.mlab.psd(obj.dat[:,1], Ns, Fs, window=flatop(len(obj.dat[:,1])))
	
	mask = ((freqs-fc/2)<bandwidth/2) & ((freqs-fc/2)>-bandwidth/2)
	
	amp = 2*np.sum(np.sqrt(psd)[mask])
	
	if plot:
		plt.plot(freqs,2*np.sqrt(np.abs(psd)))
		plt.yscale('log')
		plt.show()
	return amp


#psd, freqs = matplotlib.mlab.psd(obj0.dat[:,1], Ns, Fs, window=flatop(len(obj0.dat[:,1])))
##psd2, freqs2 = matplotlib.mlab.psd(obj0.dat[:,1],Ns,Fs,window=matplotlib.mlab.window_none)
#
amp = np.zeros(n_file)
phase_freq = np.zeros(n_file)
#
#plt.plot(freqs,2*np.abs(psd))
#plt.yscale('log')
#plt.show()
#
#amp_d = find_drive_amp(obj0,10)
#print(amp_d)

print('starting')
for i, f in enumerate(files[init_file:final_file:ns]):
	print(i)
	
	obj = hsDat(f)
	amp[i] = find_drive_amp(obj,10)
	phase_freq[i] = find_phasemod_freq(obj, 6000)
	

print(50.*0.66*(1/4e-3)*amp)
if save:
	np.save(out_path + out_base_fname + "damps", amp)
	np.save(out_path + out_base_fname + "wob_freqs", phase_freq) 
