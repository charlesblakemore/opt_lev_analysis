import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import curve_fit
from amp_ramp_3 import flattop
from amp_ramp_3 import bp_filt
from ring_down_analysis_v3 import track_frequency
import bead_util_funcs as buf
import bead_util as bu
import hs_digitizer as hd
import os 

matplotlib.rcParams['agg.path.chunksize'] = 10000

save = True
wobble = False

overwrite = True 


fils = ['/data/old_trap/20191105/bead4/phase_mod/changing_phase_mod_freq_4_fine/']

#fils = ['/daq2/20190805/bead1/spinning/wobble/reset_dipole_1/', '/daq2/20190805/bead1/spinning/wobble/reset_dipole_2/',\
#		'/daq2/20190805/bead1/spinning/wobble/reset_dipole_3/']
#out_paths = ['/home/dmartin/analyzedData/20190805/wobble/reset_dipole_1_redo/', \
#			 '/home/dmartin/analyzedData/20190805/wobble/reset_dipole_2_redo/', \
#			 '/home/dmartin/analyzedData/20190805/wobble/reset_dipole_3_redo/']

skip_files = ['none']

start_path = 0
end_path = 0

start_file = 0 
end_file = 0



out_paths = ['/home/dmartin/Desktop/analyzedData/20191105/']

#Uncomment for single file input and ouput and remove for multi file loop at bottom of the script which spits out multiple outputs	
#path = '/daq2/20190805/bead1/spinning/wobble/reset_dipole_4/'
#out_path = '/home/dmartin/analyzedData/20190805/wobble/reset_dipole_4_redo/'
#path = '/daq2/20190626/bead1/spinning/wobble/wobble_many_slow/wobble_0000'

tabor_fac = 100.
spinning_freq = 25e3
pm_bandwidth = 550
drive_pm_freq = 300

plot = False
gauss_fit = False
lorentzian_fit = True
libration = True
dipole = False

mask_on = False



def lorentzian(x, A, x0, g, B):
	return A * (1./ (1 + ((x - x0)/g)**2)) + B

def gauss(x, A, mean, std):
	return  A * np.exp(-(x-mean)**2/(2.*std**2))

def sqrt(x, a, b):
	return a * np.sqrt(x)
def sine(x, A, f, c):
        return A * np.sin(2.*np.pi*f*x + c)

def find_efield_amp(obj):
	Ns = obj.attribs['nsamp']
	Fs = obj.attribs['fsamp']
	
	drive = obj.dat[:,1]
	filt_sig = bp_filt(drive,spinning_freq,Ns,Fs,100)

	#window = flattop(len(filt_sig))	
	fft = np.fft.rfft(filt_sig)
	freqs = np.fft.rfftfreq(Ns,1/Fs)

	zeros = np.zeros(Ns)
	voltage = np.array([zeros,zeros,zeros,filt_sig, zeros, zeros, zeros, zeros])
	ef = bu.trap_efield(tabor_fac*voltage)

	window = flattop(len(ef[0]))
	z = ss.hilbert(ef[0])
	#plt.plot(np.abs(z))
	#plt.show()
	
		
	fft_z = np.fft.fft(np.abs(z))
	fft_z_freq = np.fft.fftfreq(Ns, 1/Fs)
	#plt.plot(np.abs(z))
	#plt.plot(fft_z_freq,np.abs(fft_z)/len(fft_z))
	#plt.show()

	efield_guess = max(np.abs(fft_z)/len(fft_z)) 	
	init_params = [efield_guess, 0, 1]

	popt, pcov = curve_fit(gauss,fft_z_freq, 2*np.abs(fft_z)/len(fft_z), p0 = init_params)
	x_axis = np.linspace(np.amin(fft_z_freq),np.amax(fft_z_freq),len(fft_z_freq))

	#The amplitude of the electric field is in the 0 frequency bin because it should be constant during the measurement.
	#plt.plot(fft_z_freq, 2*np.abs(fft_z)/len(fft_z))
	#plt.plot(x_axis, gauss(x_axis,*popt))
	#plt.show()	
	
	#Should try to use proper windowing and figure out the window correction factor to get proper estimate of amp.	
	efield = popt[0]
	efield_err = np.sqrt(np.diag(pcov))[0]

	return np.array([efield, efield_err])

def find_spin_freq(obj):
	Ns = obj.attribs['nsamp']
	Fs = obj.attribs['fsamp']

	spin_sig = obj.dat[:,0]
	spin_sig_filt = bp_filt(spin_sig,2*spinning_freq, Ns, Fs, 1000)

	fft_filt = np.fft.rfft(spin_sig_filt)
	fft_filt_freqs = np.fft.rfftfreq(Ns, 1./Fs)

	#plt.loglog(fft_filt_freqs, np.abs(fft_filt))
	#plt.show()	
        
        if plot:
            plt.loglog(fft_filt_freqs,np.abs(fft_filt))
            plt.show()

	window = flattop(len(spin_sig))
	z = ss.hilbert(spin_sig_filt)
	
	phase = ss.detrend(np.unwrap(np.angle(z)))	

	f = 70.
	Q = 3
	b, a = ss.iirnotch(2.*f/Fs, Q)
	freq, h = ss.freqz(b, a, worN=len(phase))
	phase = ss.lfilter(b, a, phase)
	
#	for i in range(5):
#		f = 75. * (i+1)
#		Q = 3 
#		b, a = ss.iirnotch(2.*f/Fs, Q)
#		freq, h = ss.freqz(b, a, worN=len(phase))	
#		phase = ss.lfilter(b, a, phase)

	window = np.hanning(len(phase))
	phase *= window
	
	fft_z = np.fft.rfft(phase)
	fft_freqs = np.fft.rfftfreq(Ns, 1./Fs)

        if plot:
            plt.loglog(fft_freqs, np.abs(fft_z))
            plt.show()

	freq_ind_guess = np.argmax(np.abs(fft_z))

	if gauss_fit:
		init_params = [np.abs(fft_z[freq_ind_guess]), fft_freqs[freq_ind_guess],1]

		try:
			popt, pcov = curve_fit(gauss, fft_freqs, np.abs(fft_z), p0=init_params)
		except RuntimeError:
			popt = [0,0,0]
			pcov = [[0,0],[0,0]]
			pass
		
		x_axis = np.linspace(np.amin(fft_freqs),np.amax(fft_freqs), 3. * len(fft_freqs))
		
		spin_freq = popt[1]
		spin_freq_err =  popt[2]#np.sqrt(np.diag(pcov))[1], can possibly add these errors in quadrature
	
	elif lorentzian_fit:
		init_params = [np.abs(fft_z[freq_ind_guess]), fft_freqs[freq_ind_guess], 1, 0]
		
		try: 
			popt, pcov = curve_fit(lorentzian, fft_freqs, np.abs(fft_z), p0=init_params)
			
		except RuntimeError:
			popt = [0,0,0]
			pcov = [[0,0],[0,0]]
			pass 
		
		x_axis = np.linspace(np.amin(fft_freqs),np.amax(fft_freqs), 3. * len(fft_freqs))
		
		spin_freq = popt[1]
		spin_freq_err = popt[2]#np.sqrt(np.diag(pcov))[1]

	#plt.plot(fft_freqs, np.abs(fft_z))
	#plt.plot(x_axis,lorentzian(x_axis,*popt))
	#plt.plot((1/(2.*np.pi)) * freq * Fs, np.abs(h))
	#plt.axis([0,600,-1000,60e3])
	#plt.show()

	return np.array([spin_freq, spin_freq_err])

def forced_libration(obj, prev_pm_freq):
    Ns = obj.attribs['nsamp']
    Fs = obj.attribs['fsamp']
    freqs = np.fft.rfftfreq(Ns,1./Fs)

    t = np.arange(0,Ns/Fs,1./Fs)
    
    drive_sig = obj.dat[:,1]

    fft_drive = np.fft.rfft(drive_sig)

    z_drive = ss.hilbert(drive_sig)
    phase_drive = ss.detrend(np.unwrap(np.angle(z_drive)))

    phase_drive_filt = bp_filt(phase_drive, drive_pm_freq, Ns, Fs, pm_bandwidth)
    
    fft_phase_drive = np.fft.rfft(phase_drive_filt)

    freq_ind_max = np.argmax(np.abs(fft_phase_drive))
    freq_guess = freqs[freq_ind_max]

    p0 = [0, freq_guess, 0]
    
    popt, pcov = curve_fit(sine, t, phase_drive_filt, p0)

    E_pm_freq = popt[1]
   
    #plt.plot(sine(t,*popt))
    #plt.plot(phase_drive_filt)
    #plt.show()
    

    #plt.loglog(freqs, np.abs(fft_phase_drive))
    #plt.show()

    spin_sig = obj.dat[:,0]
    
    fft = np.fft.rfft(spin_sig)

    z = ss.hilbert(spin_sig)
    phase = ss.detrend(np.unwrap(np.angle(z)))
  
    phase_filt = bp_filt(phase,E_pm_freq, Ns, Fs, 10)

    #phase_filt = flattop(len(phase_filt)) * phase_filt
        
    z_phase_filt = ss.hilbert(phase_filt)
    
    
    pm_amp = np.abs(z_phase_filt)
    
    fft_phase = np.fft.rfft(phase_filt)
    
    fft = np.fft.rfft(phase)
    
    #plt.plot(phase)
    #plt.show()
    #
    plt.plot(phase_filt, label=r'$\phi$')
    plt.plot(pm_amp, label=r'Envelope of $\phi$')
    plt.ylabel(r'Amplitude')
    plt.xlabel('Sample Number')
    plt.legend()
    plt.show()

    #plt.loglog(freqs,np.abs(fft_phase))
    #plt.show()

    pm_amp_avg = np.mean(pm_amp)

    return np.array([E_pm_freq, pm_amp_avg])
    
def libration_chirp(obj):
    Ns = obj.attribs['nsamp']
    Fs = obj.attribs['fsamp']
   
    mask = -38000
    
    spin_sig = obj.dat[:,0]
    spin_sig_filt = bp_filt(spin_sig,2*spinning_freq, Ns, Fs, 9000)
    
    #plt.plot(spin_sig_filt)
    #plt.show()

    fft_filt = np.fft.rfft(spin_sig_filt)
    fft_filt_freqs = np.fft.rfftfreq(Ns, 1./Fs)
    plt.loglog(fft_filt_freqs,np.abs(fft_filt))
    plt.show()

    z = ss.hilbert(spin_sig_filt)

    phase = np.unwrap(np.angle(z))
    inst_freq_ = np.diff(phase)/(2*np.pi) *Fs

    plt.plot(inst_freq_)
    plt.show()

    fft_freq = np.fft.rfft(inst_freq_)

    plt.loglog(fft_filt_freqs[1:],np.abs(fft_freq))
    plt.show()

    phase = ss.detrend(phase)
    
    if mask_on:
        phase = phase[mask:]

    fft_phase = np.fft.rfft(phase)
    
    Ns = len(phase)

    fft_filt_freqs = np.fft.rfftfreq(Ns,1./Fs) 
    plt.loglog(fft_filt_freqs, np.abs(fft_phase))
    plt.show()

    phase_sig_filt = bp_filt(phase,300.,Ns,Fs,1000)
    fft_filt_phase = np.fft.rfft(phase_sig_filt)
    
    #track_frequency(freqs=fft_filt_freqs, curr_spin_freq=80,plot=True,fft=fft_filt_phase, wind=50)    
    
    
    plt.loglog(fft_filt_freqs,np.abs(fft_filt_phase))
    plt.show()

    

    ###Extract instantaneous frequency of sideband###
    z_phase_sig_filt = ss.hilbert(phase_sig_filt)

    z_phase_sig = ss.hilbert(phase)
    phase_psg = np.unwrap(np.angle(z_phase_sig))
    
    #phase_psg = np.unwrap(np.angle(z_phase_sig_filt))
    inst_freq = np.diff(phase_psg)/(2. * np.pi) *Fs
    
    plt.plot(ss.detrend(phase_psg))
    #plt.plot(inst_freq[:])
    plt.show()
    #################################################
    
    #plt.plot(phase_sig_filt)
    #plt.show()

efield_amps = []
efield_amp_errs = []
spin_freqs = []	
spin_freq_errs = []
pm_freq = 0
for k, path in enumerate(fils):
	paths = []
	

        for root, dirnames, filenames in os.walk(path):
		if dirnames:
			for i, dirname in enumerate(dirnames):
				if 'junk' in dirname:
					continue
				if dirname in skip_files:
					continue
				paths.append(os.path.join(root,dirname))
				
			break #break after first level, don't need to go any further
                else:
                    print(path)
                    paths.append(path)
                    break
	
        #paths.append(path)
        #print(paths)
        #raw_input()
        out_path = out_paths[k]
	if save:
		buf.make_all_pardirs(out_path)

		
	if end_path == 0:
		paths = paths[start_path:]
	else:
		paths = paths[start_path:end_path]
	
	for j, path in enumerate(paths):
                meas_name = path.split('/')[-1]
		save_path = out_path + '{}'.format(meas_name)
                       
		if not overwrite and os.path.exists(save_path + '.npy'):		
			continue
	
		files, zero = bu.find_all_fnames(path)

                if end_file == 0:
                    files = files[start_file:]
                else:
                    files = files[start_files:end_files]

		efield_amps = []
		efield_amp_errs = []
		spin_freqs = []	
		spin_freq_errs = []
		times = []
                E_pm_freqs = []
                pm_amp_avgs = []


		for i in range(len(files)):
                        buf.progress_bar(i, len(files))
			data = hd.hsDat(files[i])
			
			time = data.attribs['time']
			
                        if dipole:
			    efield_arr = find_efield_amp(data)
			    spin_freqs_arr = find_spin_freq(data)	
		        

			    efield_amps.append(efield_arr[0])
			    efield_amp_errs.append(efield_arr[1])
			    spin_freqs.append(spin_freqs_arr[0])
			    spin_freq_errs.append(spin_freqs_arr[1])
			    times.append(time)	

                        if libration:
                            arr = forced_libration(data,pm_freq)
                            
                            pm_freq = arr[0]
                            
                            E_pm_freqs.append(arr[0])
                            pm_amp_avgs.append(arr[1])
                            #libration_chirp(data)
                
                print(E_pm_freqs,pm_amp_avgs)

                plt.scatter(E_pm_freqs,pm_amp_avgs)
                plt.show()

                if save and wobble:
			np.save(save_path, np.array([efield_amps,efield_amp_errs,\
										spin_freqs,spin_freq_errs,times]))

                if save and libration:
                    print(save_path)
                    np.save('/home/dmartin/Desktop/analyzedData/20191105/phase_mod/forced_libration/no_window/forced_libration_4_fine_no_window.npy', np.array([E_pm_freqs,pm_amp_avgs]))

                

#popt, pcov = curve_fit(sqrt, efield_amps, spin_freqs, sigma=spin_freq_errs)
#x_axis = np.linspace(efield_amps[0], efield_amps[-1], len(spin_freqs*2))
#
#plt.plot(efield_amps, spin_freqs)
#plt.plot(x_axis, sqrt(x_axis,*popt))
#plt.show()
#	
#meas_name = path.split('/')[-2]
#
#if save:
#	np.save(out_path + '_{}'.format(meas_name), np.array([efield_amps,efield_amp_errs,spin_freqs,spin_freq_errs]))
