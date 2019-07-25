import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from hs_digitizer import *
from scipy.optimize import curve_fit
import matplotlib
import re

amp_fac = 50.*0.66 #Tabor monitor:1/100 and 2*Vpk
electrode_sep = 0.004
num_volt = 1 #Number of times a file is taken at a given electrode Vpp

rho = 1.55e3
m = 88.5e-15
 
moment_of_inertia =2.020002487e-25 
#in_path = "/home/arider/opt_lev_analysis/scripts/spinning/processed_data/20181204/ampramp_data_0/"
in_base_fname = "amp_ramp_50k"
in_path = ['/home/dmartin/analyzedData/20190514/pramp3/N2/wobble/nextday/','/home/dmartin/analyzedData/20190514/amp_ramp/june12/','/home/dmartin/analyzedData/20190514/pramp3/N2/wobble/repeat/']

def load_data(fil):
	amps = np.load(fil+in_base_fname + "_damps.npy")
	amps*= amp_fac
	amps/= electrode_sep
	
	
	freqs = np.load(fil+in_base_fname + "_wob_freqs.npy")
	
	freqs *= 2.*np.pi
	
	exc = freqs != 0
	freqs = freqs[exc]
	amps = amps[exc]
	
	return amps, freqs	


def sqrt_fun(x, poi, toi, c):
	return np.sqrt(x*poi)   

def slice_array(ind,array,width):
	'''Takes an array and splits it into array with some width beginning at ind*width'''

	return array[ind*width:(1+ind)*width]

def find_dipole_moment(amps_,freqs_):
	'''Computes average amplitudes and freqs for sliced arrays'''
	avg_amps = np.ones([len(amps)/num_volt])
	avg_freqs = np.ones([len(freqs)/num_volt])
	
	std_amps = np.ones([len(amps)/num_volt])
	std_freqs = np.ones([len(freqs)/num_volt])


	for i in range(len(amps_)/num_volt):
		sliced_amps = slice_array(i,amps_,num_volt)
	 	sliced_freqs = slice_array(i,freqs_,num_volt)
		#sliced_avg_amps = slice_array(i,avg_amps,num_volt) 	
		#sliced_avg_freqs = slice_array(i,avg_freqs,num_volt) 	
		
		exc_zero = sliced_freqs != 0
		sliced_freqs = sliced_freqs[exc_zero]
		avg_amps[i] = np.sum(sliced_amps)/len(sliced_amps)	
		avg_freqs[i] = np.sum(sliced_freqs)/len(sliced_freqs)
		
		std_amps[i] = np.std(sliced_amps)
		std_amps[i] /= np.sqrt(len(sliced_amps)) #standard error
		
		std_freqs[i] = np.std(sliced_freqs)
		std_freqs[i] /= np.sqrt(len(sliced_freqs)) #standard error 
		#sliced_avg_amps[:] = avg_field
		#sliced_avg_freqs[:] = avg_freq
		#avg_amps[i*num_volt:num_volt + i*num_volt] = avg
	
	
	avg_amps = avg_amps/1000
		
	#print(avg_amps)
	#print(avg_freqs)
	#print(std_freqs)
	#print(std_amps)
	#popt, pcov = curve_fit(sqrt_fun, avg_amps, avg_freqs)
	popt, pcov = curve_fit(sqrt_fun, avg_amps, avg_freqs)
	print(popt)
	
	dipole_moment = (popt[0]/1000) * (moment_of_inertia) * (1/1.602e-19) * (1e6)
	print(dipole_moment)


	plt.plot(np.sort(avg_amps), sqrt_fun(np.sort(avg_amps), *popt),'r',label=r'{} $e\cdot\mu m$'.format(round(dipole_moment,1)))
	#plt.errorbar(avg_amps,avg_freqs,xerr=std_amps,yerr=std_freqs,ecolor='k', fmt='o')
	plt.scatter(avg_amps,avg_freqs)




for i in range(len(in_path)):
	amps, freqs = load_data(in_path[i])
	
	avg_amps = np.ones([len(amps)/num_volt])
	avg_freqs = np.ones([len(freqs)/num_volt])
	
	std_amps = np.ones([len(amps)/num_volt])
	std_freqs = np.ones([len(freqs)/num_volt])
	
	find_dipole_moment(amps,freqs)

plt.legend()

plt.show()

