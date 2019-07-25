import hs_digitizer as hd
import h5py
import bead_util_funcs as buf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.optimize import curve_fit

save = False 
#if raw_input('Save data?') == 1:
#	save = True

#else:
#	save = False
  
path_list = ["/daq2/20190626/bead1/spinning/ringdown/50kHz_ringdown/",\
			"/daq2/20190626/bead1/spinning/ringdown/50kHz_ringdown2/"]
#path_list = ["/data/20181204/bead1/high_speed_digitizer/spindown/base_pressure_150kstart_1/"]
out_path = "/home/dmartin/analyzedData/20190626/ringdown/"

matplotlib.rcParams['agg.path.chunksize'] = 10000

if os.path.isdir(out_path) == False:
	print("out_path doesn't exist. Creating...")
	os.mkdir(out_path)
	
spin_freq = 100e3

def gauss(x,a,b,c):
	return a*np.exp(-(x-b)**2/(2*c))

def track_frequency(f):
	
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
		time.append((df.attribs['time']-t0)*1e-9)
	
	except RuntimeError as e:
		print(e)
		plt.plot(freqs[mask],gauss(freqs[mask],*popt))
		plt.plot(freqs[mask],np.abs(fft)[mask])
		plt.show()
		break

	return gauss_fit_params

def find_spin_freq(files,meas_name):

	df0 = hd.hsDat(files[0])
	Ns = df0.attribs['nsamp']
	Fs = df0.attribs['fsamp']
	
	freqs = np.fft.rfftfreq(Ns, 1/Fs)
	off_ind = 0
	
	init_drive = np.fft.rfft(df0.dat[:,1])
	
	drive_mask = (freqs > spin_freq * 0.5 - 400) & (freqs < 0.5 * spin_freq + 400) 
	drive_max_ind = np.argmax(init_drive[drive_mask])
	
	for i, curr_file in enumerate(files):
	#Finds when the drive is turned off
		curr_df = hd.hsDat(curr_file)
		
		drive = np.fft.rfft(curr_df.dat[:,1])
		
		if np.abs(drive[drive_mask][drive_max_ind]) < 1e-4*np.abs(init_drive[drive_mask][drive_max_ind]):
			print(i,'drive off')
			off_ind = i
			#np.save(out_path + "{}_drive_off_ind".format(meas_name),[off_ind])
			break
	
	sub_files = files[off_ind:]
	
	mask =  (freqs > spin_freq-400) & (freqs < spin_freq+400)
	
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
			#plt.loglog(freqs[mask],np.abs(fft)[mask])
			#plt.show()
			
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
		#np.save(out_path + "{}_spin_freqs.npy".format(meas_name),spin_freqs) #saves in Hz
		#np.save(out_path + "{}_time.npy".format(meas_name),time)
		#np.save(out_path + "{}_file_ind.npy".format(meas_name),file_ind)	
		np.savez(out_path + "{}_deriv_quantities.npz".format(meas_name),\
				file_ind=file_ind,time=time,gauss_fit_params=gauss_fit_params,drive_off_ind=off_ind)		

def find_spin_freq_hilbert(files,meas_name)
	''' Finds frequency of the spinning microsphere by analyzing the chirp in the spectrum and taking the average of the instantaneous freq. Since the chirp varies of such a small frequency the average can approx represent the frequency the sphere is spinning during the data was taken.'''
	

	low_freq = (99e3/freqs[-1]) 
	high_freq = (101e3/freqs[-1])
	
	b, a= signal.butter(2,[low_freq,high_freq],btype='bandpass')
	
	y = signal.filtfilt(b,a,df.dat[:,0])
	w, h = signal.freqz(b,a)
	
	
	fft_filt = np.fft.rfft(y)
	
	#plt.loglog(freqs[mask],np.abs(fft_filt)[mask])
	#plt.show()
	
	z = signal.hilbert(np.fft.irfft(fft_filt))
	
	inst_phase = np.unwrap(np.angle(z))
	inst_freq = np.diff(inst_phase)/(2. * np.pi) * Fs
	
		
	#plt.plot(inst_freq)
	#plt.show()
	if save:
		print('saving')
		np.savez(out_path + "{}_hilbert_deriv_quantities.npz".format(meas_name),\
				file_ind=file_ind,time=time,gauss_fit_params=gauss_fit_params)

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
	
	

for i in range(len(path_list)):
	files, zero= buf.find_all_fnames(path_list[i])

	name = path_list[i].split('/')[-2]

	'''	
	off_ind = np.load(out_path + "drive_off_ind_{}.npy".format(name))
	
	files = files[off_ind[-1]:]
	
	for j in range(len(files)):
		j = j + 1600
		data = np.load(out_path + "{}_deriv_quantities.npz".format(name))
		gauss_fit = data['gauss_fit_params']
		spin_freq = gauss_fit[:,1]

		plot_data(files[j],spin_freq[j],gauss_fit[j])
	'''
	find_spin_freq(files,name)

'''
x = np.arange(len(spin_freqs))	
print(x)
plt.scatter(x,spin_freqs)
plt.show()
'''



