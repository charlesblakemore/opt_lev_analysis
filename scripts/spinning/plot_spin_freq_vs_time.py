import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


save = False 
#path_list = ['/home/dmartin/analyzedData/20190626/ringdown/50kHz_ringdown_18_hilbert_deriv_quantities_0.npz',\
#			'/home/dmartin/analyzedData/20190626/ringdown/50kHz_ringdown_deriv_quantities.npz']

#path_list = ['/home/dmartin/analyzedData/20190626/ringdown/0_8_sec_freq_50kHz_ringdown_18_hilbert_deriv_quantities.npz','/home/dmartin/analyzedData/20190626/ringdown/0_8_sec_freq_50kHz_ringdown2_30_hilbert_deriv_quantities.npz']#,'/home/dmartin/analyzedData/20190626/ringdown/50kHz_ringdown_18_hilbert_deriv_quantities_0.npz']

path_list = ['/home/dmartin/analyzedData/20190626/ringdown/after_pramp/after_pramp_19_hilbert_deriv_quantities_1.npz']
path_list = ['/home/dmartin/Desktop/analyzedData/20200130/spinning/series_3/base_press/ringdown/50kHz_2/50kHz_2_0.npz']
path_list = ['/home/dmartin/Desktop/analyzedData/20200330/gbead3/spinning/ringdown/110kHz_1.npz']
path_list = ['/home/dmartin/Desktop/analyzedData/20200330/gbead3/spinning/ringdown/high_press/high_press_5/110kHz_8Vpp_xy_ringdown_1/110kHz_8Vpp_xy_ringdown_1.npz']
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
    
    #mask = time > 50
    mask = time > 0
    time = time[mask]

    
    spin_freqs = data['spin_freq'][mask]
    spin_freq_errs = data['spin_freq_err'][mask]

    
    #spin_freqs *= 2*np.pi
    
    diff = 100000
    max_time = 6000.
    mask = time < max_time
    i = 0

    start_freq = spin_freqs[0]
    #The data is cut using a mask, then a curve fit is performed on the cut data. A difference
    while diff > 20000:
    	print(i)
    	i+=1
    	max_time = max_time - 1 
    	mask = time < max_time 
    		
    	popt, pcov = curve_fit(exp,time[mask],spin_freqs[mask],p0=[106e3,10],\
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
    	ax[j].scatter(time,(spin_freqs)*(1/(start_freq)),alpha=0.5)
    	
    	ax[j].plot(time,(exp(time,*popt))*(1/(start_freq)),'r',label=\
    			   r'$\tau = {} \pm {}$ s'.format(round(tau,2),round(tau_err,2)))
    	
    	ax[j].plot(time[mask],(exp(time[mask],*popt))*(1/(start_freq)),\
    			   'tab:orange',label='$\omega_{MS}$'+' = {} Hz at time cutoff = {} s'\
    				.format(round((1e-3)*spin_freqs[mask][-1],4),round(max_time,4)))
    	
    	ax[j].scatter(time[mask][-1], spin_freqs[mask][-1] )	
    	
    	ax[j].set_ylabel('Spin frequency [Hz]')	
    	
    	ax[j].legend()
    else:	
    	
    	ax.scatter(time, spin_freqs, alpha=0.5)
    	
    	ax.plot(time, exp(time,*popt),'r',label=\
    			r'$\tau = {} $ s'.format(round(tau,0)))
    	
    	ax.plot(time[mask],exp(time[mask],*popt),'tab:orange',\
    			label='$f_{MS}$'+' = {} [kHz] at time cutoff = {} s'\
    			.format(round((1e-3)*spin_freqs[mask][-1],4),round(max_time,4)))
    	
    	ax.scatter(time[mask][-1],spin_freqs[mask][-1])	
    	
    	ax.set_ylabel('spin frequency [Hz]')	
    	
    	ax.legend()
    	
#axis([-50,6100,10,1100])
 
    print(path_list[j].split('/')[-2])
    
    

    if save:
    	np.savez(out_path + path_list[j].split('/')[-2], time = time, spin_freqs = spin_freqs,\
    			 init_spin_freq = init_spin_freq, init_spin_freq_err = init_spin_freq_err, \
    			 tau = tau, tau_err = tau_err)
     
#plt.yscale('log')
plt.xlabel('Time [s]')

print(start_freq)


#if save:
	#fig.savefig(out_path + ' ,dpi=200)

plt.show()
