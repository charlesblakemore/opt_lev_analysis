import hs_digitizer as hd
import h5py
import bead_util_funcs as buf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import bead_util as bu
from peakdetect import peakdetect 
from scipy import signal 
from scipy.optimize import curve_fit
from amp_ramp_3 import find_phasemod_freq 

save_data = True 

#path_list = ["/daq2/20190626/bead1/spinning/ringdown/50kHz_ringdown/",\
#			"/daq2/20190626/bead1/spinning/ringdown/50kHz_ringdown2/"]

path_list = ["/daq2/20190626/bead1/spinning/wobble/after_pramp_series/perp_z_axis/wobble_perp_z_axis_spindown2/"]
path_list = ['/data/old_trap/20200130/bead1/spinning/series_3/base_press/ringdown/50kHz_2/']
path_list = ['/data/old_trap/20200322/gbead1/spinning/ringdown/110kHz_1']
path_list = [#'/data/old_trap/20200330/gbead3/spinning/ringdown/110kHz_1/',\
             '/data/old_trap/20200330/gbead3/spinning/ringdown/50kHz_1/']
path_list = ['/data/old_trap/20200330/gbead3/spinning/ringdown/high_press/high_press_6/110kHz_8Vpp_xy_ringdown_2/']

out_path = "/home/dmartin/analyzedData/20190626/ringdown/after_pramp/"
out_path = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_3/base_press/ringdown/50kHz_2/'
out_path = '/home/dmartin/Desktop/analyzedData/20200330/gbead3/spinning/ringdown/'
out_path = '/home/dmartin/Desktop/analyzedData' + '{}'.format(path_list[0].split('old_trap')[1])

#Somehow allows me to plot a 50e3 sample signal. Need to understand it
matplotlib.rcParams['agg.path.chunksize'] = 10000

spin_freq = 110e3
crossp_freq = 2*spin_freq
crossp_freq = 92e3
filter_bw = 300 


plot_gauss =  False
plot_inst_phase = False
plot_inst_freq = False

def gauss(x,a,b,c):
	return a*np.exp(-(x-b)**2/(2*c)) 

def quadratic(x,a,b,c):
	return a*x**2 + b*x + c

def line(x,a,b):
	return a*x + b

def track_frequency(df=None,freqs=None,curr_spin_freq=0,amp=1,gauss_width=50,plot=False, fft=None , wind = 2050., max_amp=6.4, notch_filt=True):
	'''Uses a Gaussian fit to follow the chirp feature as it progress to lower frequency.'''
	
	gauss_fit_params = []
	spin_freqs = []
	time = []
	file_ind = []
        Fs = df.attribs['fsamp'] 

    	#For whatever reason the curve fit will not properly track the 	  #chirp feature if given previous amp and width values.
        p_init = [amp,curr_spin_freq,gauss_width]
	
        if fft is None:
            if notch_filt:
                f = 79.053e3#2*spin_freq
                Q = 100
                b, a = signal.iirnotch(2.*f/Fs, Q)
                freq, h = signal.freqz(b, a, worN=len(df.dat[:,0]))
                sig_filt = signal.lfilter(b, a, df.dat[:,0])

                fft = np.fft.rfft(sig_filt)
            else:
                fft = np.fft.rfft(df.dat[:,0])

	mask = (freqs > curr_spin_freq-wind) & (freqs < curr_spin_freq+wind)
	
	try:	
                while max_amp > np.amax(np.abs(fft[mask])):
                    curr_spin_freq -= 800#1500
                    mask = (freqs > curr_spin_freq-wind) & (freqs < curr_spin_freq+wind)
                    p_init = [amp,curr_spin_freq,gauss_width]
                    while (np.abs(fft[mask])[0] > max_amp or np.abs(fft[mask])[1] > max_amp) and curr_spin_freq > 100:
                        curr_spin_freq -= 100
                        mask = (freqs > curr_spin_freq-wind) & (freqs < curr_spin_freq+wind)
                        p_init = [amp,curr_spin_freq,gauss_width]

                        if len(fft[mask]) < 2:
                            break

                #plt.loglog(freqs,np.abs(fft))
                #plt.show()

                #plt.loglog(freqs[mask],np.abs(fft)[mask])
                #plt.show()
                print(p_init)
                popt, pcov = curve_fit(gauss,freqs[mask],np.abs(fft[mask]),p0=p_init)
	        
                print(popt)
		new_spin_freq = popt[1]
		
		if plot:
                        mask = (freqs > new_spin_freq-wind) & (freqs < new_spin_freq+wind)
			plt.plot(freqs[mask],gauss(freqs[mask],*popt),label='center {}'.format(popt[1]))
			plt.plot(freqs[mask],np.abs(fft)[mask])
			plt.title('track freq')
			plt.show()
		
		#p_init[1] = new_spin_freq
	
		#gauss_fit_params.append(popt)	
		#spin_freqs.append(new_spin_freq)
		#time.append((df.attribs['time']-t0)*1e-9)
	
                max_amp = np.amax(np.abs(fft[mask]))
	except RuntimeError as e:
		#print(e)
		#plt.plot(freqs[mask],gauss(freqs[mask],*popt))
		#plt.plot(freqs[mask],np.abs(fft)[mask])
		#plt.show()
		new_spin_freq = curr_spin_freq
		popt = [0,0,0] 
                print('fit failed')
                max_amp = max_amp
		pass

	return new_spin_freq, popt, max_amp

def drive_state_ind(files):
	'''Finds the file index at which the drive switches off'''
	df0 = hd.hsDat(files[0])
	Ns = df0.attribs['nsamp']
	Fs = df0.attribs['fsamp']
	
	freqs = np.fft.rfftfreq(Ns, 1/Fs)
	off_ind = 0
	
	init_drive = np.fft.rfft(df0.dat[:,1])
	
	drive_mask = (freqs > spin_freq - 400) & (freqs < spin_freq + 400) 
	drive_max_ind = np.argmax(init_drive[drive_mask])


	for i, curr_file in enumerate(files):
	#Finds when the drive is turned off
		curr_df = hd.hsDat(curr_file)
		
		drive = np.fft.rfft(curr_df.dat[:,1])
		
		if np.abs(drive[drive_mask][drive_max_ind]) < 1e-4*np.abs(init_drive[drive_mask][drive_max_ind]):
			print(i,'drive off')
			off_ind = i
			break
		

	return off_ind

def find_spin_freq(files,meas_name):
	'''Finds spin frequency using the mean of a Gaussian fit to the chirp feature associated with the microsphere's decreasing frequency. Since the chirp varies over such a small frequency the average can approx represent the frequency the sphere is spinning during the data was taken.'''

	df0 = hd.hsDat(files[0])
	Ns = df0.attribs['nsamp']
	Fs = df0.attribs['fsamp']
	
	freqs = np.fft.rfftfreq(Ns, 1/Fs)
	off_ind = 0
	
	init_drive = np.fft.rfft(df0.dat[:,1])
	
	drive_mask = (freqs > spin_freq * 0.5 - 400) & (freqs < 0.5 * spin_freq + 400) 
	drive_max_ind = np.argmax(init_drive[drive_mask])


	off_ind = drive_state_ind(files)
	
	sub_files = files[off_ind:]
	
	mask =  (freqs > spin_freq-400) & (freqs < spin_freq+400)
	gauss_fit = track_frequency(sub_files)
	
	gauss_fit_params = []
	spin_freqs = []
	time = []
	file_ind = []
	p_init = [1,spin_freq,50]

	t0 = df0.attribs['time']	
	for i,f in enumerate(sub_files):
		
		df = hd.hsDat(f)
		fft = np.fft.rfft(df.dat[:,0])
	


		try:	
			popt, pcov = curve_fit(gauss,freqs[mask],np.abs(fft[mask]),p0=p_init)
		
			new_spin_freq = popt[1]
			
			mask = (freqs > new_spin_freq-300) & (freqs < new_spin_freq+300)
			#plt.plot(freqs[mask],gauss(freqs[mask],*popt),label='center {}'.format(popt[1]))
			plt.loglog(freqs[mask],np.abs(fft)[mask])
			plt.show()
			
			p_init[1] = new_spin_freq
		
			gauss_fit_params.append(popt)	
			spin_freqs.append(new_spin_freq)
			file_ind.append(i)
			time.append((df.attribs['time']-t0)*1e-9)

		except RuntimeError as e:
			print(e)
			plt.plot(freqs[mask],gauss(freqs[mask],*popt))
			plt.plot(freqs[mask],np.abs(fft)[mask])
			plt.show()
			break
	
		print(new_spin_freq, "index {}".format(i))
	if save:
		print('saving')
		np.savez(out_path + "{}_deriv_quantities.npz".format(meas_name),\
				file_ind=file_ind,time=time,gauss_fit_params=gauss_fit_params, off_ind=off_ind)		

def find_spin_freqs_hilbert(files,meas_name,curr_spin_freq, plot=False, plot_phase=False, plot_freq=False, save=False):
	''' Finds frequency of the spinning microsphere by analyzing the chirp in the spectrum and fitting a quadratic to the instantaneous phase. From this, we know the instantaneous frequency of the sphere.'''
	freq = []
	freq_err = []
	time = [] 
	gauss_params = [25,curr_spin_freq,700]
	#curr_spin_freq = spin_freq	

	df = hd.hsDat(files[0])
	t0 = df.attribs['time']	
	Ns = df.attribs['nsamp']
	Fs = df.attribs['fsamp']
	freqs = np.fft.rfftfreq(Ns, 1/Fs)
	
	tarr = np.arange(Ns)/Fs
		
	max_amp = 6.4
	for i, f in enumerate(files):	
		print(i, f)
		df = hd.hsDat(f)
	
		fft = np.fft.rfft(df.dat[:,0])
		curr_spin_freq, gauss_params, max_amp = track_frequency(df,freqs,curr_spin_freq,gauss_params[0],gauss_params[2],plot, max_amp=max_amp)
	
                #Divided by freq[-1] since the filter low and high freq
		#should be normalized to np.pi. Check signal butter docs	
		low_freq = (curr_spin_freq-filter_bw*0.5)/freqs[-1] 
		high_freq = (curr_spin_freq+filter_bw*0.5)/freqs[-1]
		
                if low_freq < 0:
                    break
                #Used 2nd order filter simply because it was "stronger" 
		#in canceling out frequencies
	        
                print(low_freq,high_freq)
		b, a = signal.butter(2,[low_freq,high_freq],btype='bandpass')
		
		y = signal.filtfilt(b,a,df.dat[:,0])
		w, h = signal.freqz(b,a)
		
		
		fft_filt = np.fft.rfft(y * np.hanning(len(y)))
		
		mask = (freqs > low_freq) & (freqs < high_freq)	
		#plt.loglog(freqs,np.abs(fft_filt))
		#plt.xlabel('Frequency [Hz]')
		#plt.ylabel('Amplitude [Arb.]')
		#plt.show()

		irfft = np.fft.irfft(fft_filt)
		z = signal.hilbert(irfft)
		
		inst_phase = np.unwrap(np.angle(z))
		inst_freq = np.gradient(inst_phase)/(2. * np.pi) * Fs
		
		popt, pcov = curve_fit(quadratic,tarr,inst_phase, \
							   p0=[-1,2*np.pi*curr_spin_freq,1])
		
		f_slope = popt[0]/(2*np.pi)
		f_slope_err = np.sqrt(np.diag(pcov))[0]

		f_init = popt[1]/(2*np.pi)
		f_init_err = np.sqrt(np.diag(pcov))[1]
		

                t_half = tarr[-1]/2.

		#print(f_init)	
		f = f_slope * t_half + f_init #Take the freqeuncy at the center of the integration window as an estimate of
                                              #the current MS rotation freqeuncy
		
                f_err = np.sqrt((t_half*f_slope_err)**2 + (f_init_err)**2)

		print(f)	
	
		count = 0	
                
                if plot_phase:
                    plt.plot(tarr,inst_phase)
		    plt.plot(tarr,quadratic(tarr,*popt))
		    plt.title('Inst. Phase')
		    plt.ylabel(r'$\phi(t)$')
		    plt.xlabel('Time [s]')	
		    plt.show()
                
                if plot_freq:
		    plt.plot(tarr,line(tarr,popt[0],popt[1])/(2*np.pi))
		    plt.title('Inst. Frequency')
		    plt.ylabel('Frequency [Hz]')
		    plt.xlabel('Time [s]')
		    plt.show()
		
		time.append((df.attribs['time']-t0) * 1e-9)
		freq.append(f)
		freq_err.append(f_err)
		#print(freq_err)
		#freq.append(np.mean(inst_freq))	
		#print(np.mean(inst_freq),popt[1]/(2*np.pi))	
				
	if save:
		save_path = out_path + "{}".format(meas_name)
		
		base_path = save_path

		j = 0	
		while os.path.exists(save_path + ".npz"):
			save_path = base_path + '_{}'.format(j)
			j += 1

		print('saving')

		np.savez(save_path + ".npz",time=time,spin_freq=freq,spin_freq_err=freq_err,\
				 gauss_params=gauss_params)

def plot_data(f,freq,gauss_fit_params):
	df = hd.hsDat(f)
	Ns = df.attribs['nsamp']
	Fs = df.attribs['fsamp']
	
	freqs = np.fft.rfftfreq(Ns,1/Fs)
	fft = np.fft.rfft(df.dat[:,0])
	
	mask = (freqs > freq-300) & (freqs < freq+300)	
	
	plt.plot(freqs[mask],gauss(freqs[mask],*gauss_fit_params))	
	plt.plot(freqs[mask],np.abs(fft)[mask])
	plt.show()
	
	
if __name__ == "__main__":
    #if os.path.isdir(out_path) == False:
    #    print("out_path doesn't exist. Creating...")
    #    os.mkdir(out_path)

    if save_data:
        bu.make_all_pardirs(out_path)
    
    for i in range(len(path_list)):
    	files, zero= buf.find_all_fnames(path_list[i])
    	off_ind = drive_state_ind(files)
    	
    	name = path_list[i].split('/')[-2]
        meas_name = name 

        find_spin_freqs_hilbert(files[off_ind:], meas_name,crossp_freq, plot=plot_gauss, plot_freq=plot_inst_freq, plot_phase=plot_inst_phase, save=save_data)	

	
