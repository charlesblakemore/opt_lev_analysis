import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


save = False 
#path_list = ['/home/dmartin/analyzedData/20190626/ringdown/50kHz_ringdown_18_hilbert_deriv_quantities_0.npz',\
#			'/home/dmartin/analyzedData/20190626/ringdown/50kHz_ringdown_deriv_quantities.npz']

#path_list = ['/home/dmartin/analyzedData/20190626/ringdown/0_8_sec_freq_50kHz_ringdown_18_hilbert_deriv_quantities.npz','/home/dmartin/analyzedData/20190626/ringdown/0_8_sec_freq_50kHz_ringdown2_30_hilbert_deriv_quantities.npz']#,'/home/dmartin/analyzedData/20190626/ringdown/50kHz_ringdown_18_hilbert_deriv_quantities_0.npz']

path_list = ['/home/dmartin/analyzedData/20190626/ringdown/after_pramp/after_pramp_19_hilbert_deriv_quantities_1.npz']
path_list = ['/home/dmartin/Desktop/analyzedData/20200130/spinning/series_3/base_press/ringdown/50kHz_2/50kHz_2_0.npz']

out_path = '/home/dmartin/analyzedData/20190626/ringdown/after_pramp/'

def gauss(x,a,b,c):
	return a*np.exp(-(x-b)**2/(2*c))

def exp(x, a, b):
	return a*np.exp(-x *(1/b))

L = len(path_list)
fig, ax = plt.subplots(L,1,sharex=True,sharey=True)
#fig, ax = plt.subplots(L,1)

data = []
for j in range(len(path_list)):
	data = np.load(path_list[j])
	
	
	#Some of the data files were saved with gauss_fit_params when the mean of Gaussian fits to the chirp feature was used.	
	#spin_freqs = data['gauss_fit_params'][:,1]
	time = data['time']
	spin_freqs = 0.5*data['spin_freq']
	spin_freq_errs = data['spin_freq_err']

	
	spin_freqs *= 2*np.pi
	
	diff = 100000
	max_time = 6000.
	mask = time < max_time
	i = 0

	
	#The data is cut using a mask, then a curve fit is performed on the cut data. A difference
	while diff > 20000:
		print(i)
		i+=1
		max_time = max_time - 1 
		mask = time < max_time 
			
		popt, pcov = curve_fit(exp,time[mask],spin_freqs[mask],p0=[100e3,100],\
							  sigma=spin_freq_errs[mask])
		
		diff = np.sum(spin_freqs[mask]-exp(time[mask],*popt))
		total = np.sum(exp(time[mask],*popt))#np.sum(spin_freqs[mask])
		print(diff/total,total)
		#except TypeError as e:
		#	print(e)
		#	break 
	
	init_spin_freq = popt[0]
	init_spin_freq_err = np.sqrt(np.diag(pcov))[0]	
	
	tau = popt[1]
	tau_err = np.sqrt(np.diag(pcov))[1]
	
	if L > 1:
		ax[j].scatter(time,(spin_freqs)*(1/(2*np.pi*50e3)),alpha=0.5)
		
		ax[j].plot(time,(exp(time,*popt))*(1/(2*np.pi*50e3)),'r',label=\
				   r'$\tau = {} \pm {}$ s'.format(round(tau,2),round(tau_err,2)))
		
		ax[j].plot(time[mask],(exp(time[mask],*popt))*(1/(2*np.pi*50e3)),\
				   'tab:orange',label='$\omega_{MS}$'+' = {} krad/s at time cutoff = {} s'\
					.format(round((1e-3)*spin_freqs[mask][-1],4),round(max_time,4)))
		
		ax[j].scatter(time[mask][-1], spin_freqs[mask][-1] * (1/(2*np.pi*50e3)))	
		
		ax[j].set_ylabel('$\omega_{MS}$ $[\omega_{drive}$ = $100\pi$ krad/s]')	
		
		ax[j].legend()
	else:	
		
		ax.scatter(time, (spin_freqs)*(1/(2*np.pi*50e3)), alpha=0.5)
		
		ax.plot(time, (exp(time,*popt))*(1/(2*np.pi*50e3)),'r',label=\
				r'$\tau = {} \pm {}$ s'.format(round(tau,0),round(tau_err,0)))
		
		ax.plot(time[mask],(exp(time[mask],*popt))*(1/(2*np.pi*50e3)),'tab:orange',\
				label='$\omega_{MS}$'+' = {} krad/s at time cutoff = {} s'\
				.format(round((1e-3)*spin_freqs[mask][-1],4),round(max_time,4)))
		
		ax.scatter(time[mask][-1],spin_freqs[mask][-1] * (1/(2*np.pi*50e3)))	
		
		ax.set_ylabel('$\omega_{MS}$ $[\omega_{drive}$ = $100\pi$ krad/s]')	
		
		ax.legend()
		
#plt.axis([-50,6100,10,1100])
 
	print(path_list[j].split('/')[-2])
	
	if save:
		np.savez(out_path + path_list[j].split('/')[-2], time = time, spin_freqs = spin_freqs,\
				 init_spin_freq = init_spin_freq, init_spin_freq_err = init_spin_freq_err, \
				 tau = tau, tau_err = tau_err)
 
plt.yscale('log')
plt.xlabel('Time [s]')




#if save:
	#fig.savefig(out_path + ' ,dpi=200)

plt.show()
