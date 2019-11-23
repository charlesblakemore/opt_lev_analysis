import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.signal as ss
import re
import matplotlib
from hs_digitizer import * #import before bead_util_funcs
import bead_util_funcs as buf
import fnmatch
import os 

from scipy.optimize import curve_fit
from peakdetect import peakdetect as pd


save = False 

#path = "/data/20181205/bead1/high_speed_digitizer/amp_ramp/50k_zhat"
#out_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/ampramp_data_0/"
#path = "/daq2/20190514/bead1/spinning/pramp3/N2/wobble/pre-meas1_repeat"
#out_path = "/home/dmartin/analyzedData/20190514/pramp3/N2/wobble/repeat/"

#path = "/daq2/20190626/bead1/spinning/pramp/Kr/wobble_1/wobble_0000"

path = "/daq2/20190805/bead1/spinning/wobble/reset_dipole_1/after_reset"
#path = "/daq2/20190626/bead1/spinning/wobble/wobble_many/wobble_0000"
out_path = "/home/dmartin/analyzedData/20190626/pramp/"
out_base_fname = "wobble_many_wobble_0000"

def sqrt(x,a,b):
	return a*np.sqrt(x)

def line(x, m, b):
    return m*x + b

def dec2(arr, fac):
    return ss.decimate(ss.decimate(arr, fac), fac)

#Random flattop window I found (need to include ref). Apparently good for estimation of amplitudes
def flattop(M):
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
	hilb = ss.hilbert(np.fft.irfft(fft)*flattop(len(obj.dat[:,0])))
	#Extract the phase which contains information about the phase modulation of the microsphere
	phase = ss.detrend(np.unwrap(np.angle(hilb)))
	fft_phase = np.fft.rfft(phase)
	
	#Max_peaks is really the only useful thing here. This just finds the peaks in the FFT at certain frequencies.
	max_peaks, min_peaks = pd(np.abs(fft_phase),freqs,lookahead,delta)
	

	#plt.plot(freqs,np.abs(fft_phase))	
	for i in range(len(max_peaks)):
		#plt.scatter(max_peaks[i][0],max_peaks[i][1])
		
		#The peak must be greater than 60 Hz (there is sometimes a 53 Hz signal) and the amplitude must be above some threshold
		if max_peaks[i][0] > 80 and max_peaks[i][1] > amp_thresh:
			freq = max_peaks[i][0]
		else:
			freq = 0
	#plt.show()

	
		
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

def find_drive_amp_filt(obj):

	Ns = obj.attribs['nsamp']
	Fs = obj.attribs['fsamp']
	freqs = np.fft.rfftfreq(Ns,1/Fs)
	
	low_freq = (drive_freq - 1000)/freqs[-1]
	high_freq = (drive_freq + 1000)/freqs[-1]

	b, a = ss.butter(2,[low_freq,high_freq],btype='bandpass')
	
	sig = ss.filtfilt(b,a,obj.dat[:,1])
	
	win = flattop(len(sig))
	
	psd_filt, freqs = matplotlib.mlab.psd(sig,Ns,Fs,window=win)
	
	damp = 2 * np.sum(np.sqrt(psd_filt))
	
	return damp

	
def bp_filt(signal,frequency,Ns,Fs,bandwidth):
	freqs = np.fft.rfftfreq(Ns,1/Fs)
	
	low_freq = (frequency - bandwidth/2)/freqs[-1]
	high_freq = (frequency + bandwidth/2)/freqs[-1]
	
	if low_freq < 0.:
		low_freq = 1e-20	
	
	b, a = ss.butter(2,[low_freq,high_freq],btype='bandpass')
	
	sig = ss.filtfilt(b,a,signal)

	return sig	
	

if __name__ == "__main__":
	fc = 1e5 #2 times spinning frequency
	bw = 1e3 #Bandwidth for freq_bool
	drive_freq = 50e3
	
	files, zero = buf.find_all_fnames(path) 
		
	n_file = len(files)
	amp = np.zeros(n_file)
	phase_freq = np.zeros(n_file)
	
	obj0 = hsDat(files[0])
	Ns = obj0.attribs['nsamp']
	Fs = obj0.attribs['fsamp']
	freqs = np.fft.rfftfreq(Ns, 1/Fs)
	
	freq_bool = np.abs(freqs-fc)>bw
	#d_amps = np.zeros(n_file)
	#f_wobs = np.zeros(n_file)

	
	for i, f in enumerate(files):
		print(i)

		obj = hsDat(f)
		
		#amp[i] = find_drive_amp(obj,10)
		amp[i] = find_drive_amp_filt(obj)
		phase_freq[i] = find_phasemod_freq(obj, 6000) #Hz

	data_arr = np.array([amp,0,phase_freq,0])

	E = amp * 0.66 * 100 * 0.5 * (1./4.e-3)
	freq = phase_freq * 2 * np.pi
	popt, pcov = curve_fit(sqrt, E, freq)
	 
	E_arr = np.linspace(0, E[np.argmax(E)],10000)

	print popt

	plt.plot(E_arr, sqrt(E_arr,*popt))
	plt.scatter(E,freq)
	plt.show()	
	
	if save:
		np.save(out_path + out_base_fname + "data_arr", data_arr)
		#np.save(out_path + out_base_fname + "damps", amp)
		#np.save(out_path + out_base_fname + "wob_freqs", phase_freq)

