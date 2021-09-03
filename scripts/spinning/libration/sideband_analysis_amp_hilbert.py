import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import curve_fit
from amp_ramp_3 import flattop
from amp_ramp_3 import bp_filt
from amp_ramp_3 import hp_filt
from ring_down_analysis_v3 import track_frequency
import bead_util_funcs as buf
import bead_util as bu
import hs_digitizer as hd
import os
from joblib import Parallel, delayed

matplotlib.rcParams['agg.path.chunksize'] = 10000

save = True
wobble = True
parallel = True
n_jobs = 25
overwrite = True

#fils = ['/data/old_trap/20191105/bead4/phase_mod/changing_phase_mod_freq/changing_phase_mod_freq_10/']
#fils = ['/data/old_trap/20191105/bead4/phase_mod/changing_phase_mod_freq/changing_phase_mod_freq_9/']
#fils = ['/data/old_trap/20200322/gbead1/spinning/wobble/wobble_init/wobble_0000/']
#fils = ['/data/old_trap/20200307/gbead1/spinning/wobble/wobble_0/wobble_0002/']
#fils = ['/data/old_trap/20200322/gbead1/spinning/wobble/25kHz_xz_1/wobble_0000/']
#fils = ['/data/old_trap/20200322/gbead1/spinning/wobble/50kHz_yz_1/wobble_0000/']
#fils = ['/data/old_trap/20200322/gbead1/spinning/wobble/wobble_init/wobble_0000/']
#fils = ['/data/old_trap/20200330/gbead3/spinning/wobble/low_amp/25kHz_xy_1/wobble_0001/']
fils = ['/data/old_trap/20190626/bead1/spinning/wobble/after_pramp_series/reset_dipole_2/']
fils = ['/data/old_trap/20200924/bead1/spinning/dipole_meas/initial/trial_0000/']
fils = ['/data/old_trap/20210530/bead1/spinning/wobble/trial/']

#fils = ['/daq2/20190805/bead1/spinning/wobble/reset_dipole_1/', '/daq2/20190805/bead1/spinning/wobble/reset_dipole_2/',\
#		'/daq2/20190805/bead1/spinning/wobble/reset_dipole_3/']
#out_paths = ['/home/dmartin/analyzedData/20190805/wobble/reset_dipole_1_redo/', \
#			 '/home/dmartin/analyzedData/20190805/wobble/reset_dipole_2_redo/', \
#			 '/home/dmartin/analyzedData/20190805/wobble/reset_dipole_3_redo/']

out_paths = ['/home/dmartin/Desktop/analyzedData/20210530/spinning/dipole_meas/trial/']

skip_files = ['none']

start_path = 0
end_path = 0

start_file = 0
end_file = 0


#Uncomment for single file input and ouput and remove for multi file loop at bottom of the script which spits out multiple outputs	
#path = '/daq2/20190805/bead1/spinning/wobble/reset_dipole_4/'
#out_path = '/home/dmartin/analyzedData/20190805/wobble/reset_dipole_4_redo/'
#path = '/daq2/20190626/bead1/spinning/wobble/wobble_many_slow/wobble_0000'

mass = 79e-15
radius = 2.17e-6

tabor_fac = 100.
spinning_freq = 50e3
pm_bandwidth = 200
drive_pm_freq = 100

low = 15
high = 15

plot = True
if plot:
    n_jobs = 1

gauss_fit = True
lorentzian_fit = False
libration = False
dipole = True

mask_on = False



efield_amps = []
efield_amp_errs = []
spin_freqs = []
spin_freq_errs = []
times = []
E_pm_freqs = []
pm_amp_avgs = []


def lorentzian(x, A, x0, g, B):
	return A * (1./ np.sqrt((1 + ((x - x0)/g)**2)))

def gauss(x, A, mean, std, b):
	return  A * np.exp(-(x-mean)**2/(2.*std**2)) + b

def sqrt(x, a, b):
	return a * np.sqrt(x) 
def sine(x, A, f, c):
        return A * np.sin(2.*np.pi*f*x + c)
def line(x, a):
        return a*x
def find_efield_amp_parallel(f):
        obj = hd.hsDat(f)

	Ns = obj.attribs['nsamp']
	Fs = obj.attribs['fsamp']
	
	drive = obj.dat[:,1]
	filt_sig = bp_filt(drive,spinning_freq,Ns,Fs,100)

	#window = flattop(len(filt_sig))	
	fft = np.fft.rfft(filt_sig)
	freqs = np.fft.rfftfreq(Ns,1/Fs)
       
        if plot:
            fft = np.fft.rfft(drive)
            plt.plot(freqs, np.abs(fft))
            plt.show()


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
	init_params = [efield_guess, 0, 1, 0]

        #plt.semilogy(fft_z_freq, 2*np.abs(fft_z)/len(fft_z))
        #plt.show()

	#popt, pcov = curve_fit(gauss,fft_z_freq, 2*np.abs(fft_z)/len(fft_z), p0 = init_params)
	#x_axis = np.linspace(np.amin(fft_z_freq),np.amax(fft_z_freq),len(fft_z_freq))

	#The amplitude of the electric field is in the 0 frequency bin because it should be constant during the measurement.
	#plt.plot(fft_z_freq, 2*np.abs(fft_z)/len(fft_z))
	#plt.plot(x_axis, gauss(x_axis,*popt))
	#plt.show()	
	
	#Should try to use proper windowing and figure out the window correction factor to get proper estimate of amp.	
	#efield = popt[0]
	#efield_err = np.sqrt(np.diag(pcov))[0]

        efield = 2*np.abs(z)[len(z)/2]
        efield_err = np.std(np.abs(z))
        print(efield)
        #efield_amps.append(efield)
        #efield_amp_errs.append(efield_err)

	return np.array([efield, efield_err])


def find_efield_amp(obj, spinning_freq=spinning_freq):
	Ns = obj.attribs['nsamp']
	Fs = obj.attribs['fsamp']
	
	drive = obj.dat[:,1]
	filt_sig = bp_filt(drive,spinning_freq,Ns,Fs,100)

        fft = np.fft.rfft(drive)
        freqs = np.fft.rfftfreq(Ns,1./Fs)

        #plt.loglog(freqs, np.abs(fft))
        #plt.show()
        #plt.plot(drive)
        #plt.show()

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

	#efield_guess = max(np.abs(fft_z)/len(fft_z)) 	
	#init_params = [efield_guess, 0, 1, 0]

	#popt, pcov = curve_fit(gauss,fft_z_freq, 2*np.abs(fft_z)/len(fft_z), p0 = init_params)
	#x_axis = np.linspace(np.amin(fft_z_freq),np.amax(fft_z_freq),len(fft_z_freq))

	#The amplitude of the electric field is in the 0 frequency bin because it should be constant during the measurement.
	#plt.plot(fft_z_freq, 2*np.abs(fft_z)/len(fft_z))
	#plt.plot(x_axis, gauss(x_axis,*popt))
        #plt.show()	

	#Should try to use proper windowing and figure out the window correction factor to get proper estimate of amp.	

        efield = np.mean(2*np.abs(z))
        efield_err = np.std(2*np.abs(z))
        print(efield, efield_err)

        print(efield)
	return np.array([efield, efield_err])

def find_spin_freq_parallel(f):
        obj = hd.hsDat(f) 
                
	Ns = obj.attribs['nsamp']
	Fs = obj.attribs['fsamp']

	spin_sig = obj.dat[:,0]
	spin_sig_filt = bp_filt(spin_sig,2*spinning_freq, Ns, Fs, 5000)

	fft_filt = np.fft.rfft(spin_sig_filt)
	fft_filt_freqs = np.fft.rfftfreq(Ns, 1./Fs)

        fft = np.fft.rfft(spin_sig)

	#plt.loglog(fft_filt_freqs, np.abs(fft_filt))
	#plt.show()	
        
        if plot:
            plt.loglog(fft_filt_freqs, np.abs(fft))
            plt.loglog(fft_filt_freqs,np.abs(fft_filt))

            plt.show()

	window = flattop(len(spin_sig))
	z = ss.hilbert(spin_sig_filt)
	
	phase = ss.detrend(np.unwrap(np.angle(z)))	

	#f = 90.
	#Q = 3
	#b, a = ss.iirnotch(2.*f/Fs, Q)
	#freq, h = ss.freqz(b, a, worN=len(phase))
	#phase = ss.lfilter(b, a, phase)

        #f = 23.
        #Q = 3
        #b, a = ss.iirnotch(2.*f/Fs, Q)
        #freq, h = ss.freqz(b, a, worN=len(phase))
        #phase = ss.lfilter(b, a, phase)
	#for i in range(5):
	#	f = 24. * (i+1)
	#	Q = 3 
	#	b, a = ss.iirnotch(2.*f/Fs, Q)
	#	freq, h = ss.freqz(b, a, worN=len(phase))	
	#	phase = ss.lfilter(b, a, phase)

	#window = np.hanning(len(phase))
	#phase *= window
	
	fft_z = np.fft.rfft(phase)
	fft_freqs = np.fft.rfftfreq(Ns, 1./Fs)
        mask = fft_freqs > 200
        #mask = fft_freqs > 0

        fft_z = fft_z[mask]
        fft_freqs = fft_freqs[mask]

        freq_ind_guess = np.argmax(np.abs(fft_z))


        if plot:
            plt.loglog(fft_freqs,np.abs(fft_z))
            plt.scatter(fft_freqs[freq_ind_guess],np.abs(fft_z)[freq_ind_guess])
            plt.show()

        if plot:
            plt.loglog(fft_freqs, np.abs(fft_z))
            plt.show()
        
        mask_around_peak = (fft_freqs > fft_freqs[freq_ind_guess]-low) & (fft_freqs < fft_freqs[freq_ind_guess]+high)

        if gauss_fit:
		init_params = [np.abs(fft_z[freq_ind_guess]), fft_freqs[freq_ind_guess],1, 0]

                print(init_params)
		try:
			popt, pcov = curve_fit(gauss, fft_freqs[mask_around_peak], np.abs(fft_z)[mask_around_peak], p0=init_params)
		except RuntimeError:
			popt = [0,0,0,0]
			pcov = [[0,0],[0,0]]
			print('error')
                        pass
		
		x_axis = np.linspace(np.amin(fft_freqs),np.amax(fft_freqs), 3. * len(fft_freqs))
		
		spin_freq = popt[1]
		spin_freq_err =  popt[2]#np.sqrt(np.diag(pcov))[1], can possibly add these errors in quadrature
    	
                print(spin_freq)
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

        if plot:
            plt.loglog(fft_freqs, np.abs(fft_z))
	    plt.loglog(x_axis,gauss(x_axis,*popt))
	    #plt.plot((1/(2.*np.pi)) * freq * Fs, np.abs(h))
	    #plt.axis([0,600,-1000,60e3])
	    plt.show()

        print('end')
	return spin_freq, spin_freq_err#np.array([spin_freq, spin_freq_err])

def find_spin_freq_parallel_new(f):
        obj = hd.hsDat(f) 
                
	Ns = obj.attribs['nsamp']
	Fs = obj.attribs['fsamp']

	spin_sig = obj.dat[:,0]
	spin_sig_filt = bp_filt(spin_sig,2*spinning_freq, Ns, Fs, 5000)

	fft_filt = np.fft.rfft(spin_sig_filt)
	fft_filt_freqs = np.fft.rfftfreq(Ns, 1./Fs)

        fft = np.fft.rfft(spin_sig)

	#plt.loglog(fft_filt_freqs, np.abs(fft_filt))
	#plt.show()	
        
        if plot:
            plt.loglog(fft_filt_freqs, np.abs(fft))
            plt.loglog(fft_filt_freqs,np.abs(fft_filt))

            plt.show()

	window = flattop(len(spin_sig))
	z = ss.hilbert(spin_sig_filt)
	
	phase = ss.detrend(np.unwrap(np.angle(z)))	

	#f = 90.
	#Q = 3
	#b, a = ss.iirnotch(2.*f/Fs, Q)
	#freq, h = ss.freqz(b, a, worN=len(phase))
	#phase = ss.lfilter(b, a, phase)

        #f = 23.
        #Q = 3
        #b, a = ss.iirnotch(2.*f/Fs, Q)
        #freq, h = ss.freqz(b, a, worN=len(phase))
        #phase = ss.lfilter(b, a, phase)
	#for i in range(5):
	#	f = 24. * (i+1)
	#	Q = 3 
	#	b, a = ss.iirnotch(2.*f/Fs, Q)
	#	freq, h = ss.freqz(b, a, worN=len(phase))	
	#	phase = ss.lfilter(b, a, phase)

	#window = np.hanning(len(phase))
	#phase *= window
	
	fft_z = np.fft.rfft(phase)
	fft_freqs = np.fft.rfftfreq(Ns, 1./Fs)
        mask = fft_freqs > 0

        fft_z = fft_z[mask]
        fft_freqs = fft_freqs[mask]

        freq_ind_guess = np.argmax(np.abs(fft_z))


        if plot:
            plt.loglog(fft_freqs,np.abs(fft_z))
            plt.scatter(fft_freqs[freq_ind_guess],np.abs(fft_z)[freq_ind_guess])
            plt.show()

        if plot:
            plt.loglog(fft_freqs, np.abs(fft_z))
            plt.show()
        
        mask_around_peak = (fft_freqs > fft_freqs[freq_ind_guess]-low) & (fft_freqs < fft_freqs[freq_ind_guess]+high)

        num = 0
        denom = 0
        df = Fs/Ns
        for i, amp in enumerate(np.abs(fft_z)[mask_around_peak]):
            num += amp * fft_freqs[mask_around_peak][i] * df
            denom += amp * df 
        
        spin_freq = num/denom

        for i, amp in enumerate(np.abs(fft_z)[mask_around_peak]):
            num += (fft_freqs[mask_around_peak][i] - spin_freq)**2 * amp * df
            denom += amp * df   
        spin_freq_err = \
        np.sqrt(num/denom)/len(fft_freqs[mask_around_peak])

        print(Ns, Fs)
        print(spin_freq, spin_freq_err)
        print('end')
	return spin_freq, spin_freq_err#np.array([spin_freq, spin_freq_err])



def find_spin_freq(obj):
	Ns = obj.attribs['nsamp']
	Fs = obj.attribs['fsamp']

	spin_sig = obj.dat[:,0]
	spin_sig_filt = bp_filt(spin_sig,2*spinning_freq, Ns, Fs, 2000)

	fft_filt = np.fft.rfft(spin_sig_filt)
	fft_filt_freqs = np.fft.rfftfreq(Ns, 1./Fs)

        fft = np.fft.rfft(spin_sig)

	#plt.loglog(fft_filt_freqs, np.abs(fft_filt))
	#plt.show()	
        
        if plot:
            #plt.loglog(fft_filt_freqs,np.abs(fft_filt))
            plt.loglog(fft_filt_freqs, np.abs(fft))
            plt.show()

	window = flattop(len(spin_sig))
	z = ss.hilbert(spin_sig_filt)
	
	phase = ss.detrend(np.unwrap(np.angle(z)))	

	f = 90.
	Q = 3
	b, a = ss.iirnotch(2.*f/Fs, Q)
	freq, h = ss.freqz(b, a, worN=len(phase))
	phase = ss.lfilter(b, a, phase)
	
	#for i in range(5):
	#	f = 14. * (i+1)
	#	Q = 3 
	#	b, a = ss.iirnotch(2.*f/Fs, Q)
	#	freq, h = ss.freqz(b, a, worN=len(phase))	
	#	phase = ss.lfilter(b, a, phase)

	window = np.hanning(len(phase))
	phase *= window
	
        mask = fft_filt_freqs > 100
        mask = fft_filt_freqs > 0

	fft_z = np.fft.rfft(phase)
	fft_freqs = np.fft.rfftfreq(Ns, 1./Fs)

        freq_ind_guess = np.argmax(np.abs(fft_z)[mask])

        plt.loglog(fft_freqs[mask],np.abs(fft_z)[mask])
        plt.scatter(fft_freqs[freq_ind_guess],np.abs(fft_z)[freq_ind_guess])
        plt.show()

        print(fft_freqs[freq_ind_guess])
        if plot:
            plt.loglog(fft_freqs, np.abs(fft_z))
            plt.show()
        

        mask = (fft_filt_freqs > freq_ind_guess-low) & (fft_filt_freqs < freq_ind_guess+high)
	mask = fft_filt_freqs > 0

        #plt.plot(fft_filt_freqs[mask], np.abs(fft_z)[mask])
        #plt.show()
        if gauss_fit:
		init_params = [np.abs(fft_z[freq_ind_guess]), fft_freqs[freq_ind_guess],1, 0]

                print(init_params)
		try:
			popt, pcov = curve_fit(gauss, fft_freqs[mask], np.abs(fft_z)[mask], p0=init_params)
		except RuntimeError:
			popt = [0,0,0,0]
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

	#plt.plot(fft_freqs[mask], np.abs(fft_z)[mask])
	#plt.plot(x_axis,gauss(x_axis,*popt))
	
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
    
    fft_phase_drive = np.fft.rfft(phase_drive)

    phase_drive_filt = bp_filt(phase_drive, 380, Ns, Fs, pm_bandwidth)
    
    fft_phase_drive = np.fft.rfft(phase_drive_filt)

    plt.loglog(freqs, np.abs(fft_phase_drive))
    plt.show()

    freq_ind_max = np.argmax(np.abs(fft_phase_drive))
    freq_guess = freqs[freq_ind_max]

    print(freq_guess)

    p0 = [0, freq_guess, 0]
    
    popt, pcov = curve_fit(sine, t, phase_drive_filt, p0)

    E_pm_freq = popt[1]
    print(popt[1])
    #plt.plot(sine(t,*popt))
    #plt.plot(phase_drive_filt)
    #plt.show()
    
    if E_pm_freq < 30:
        plt.plot(sine(t,*popt))
        plt.plot(phase_drive_filt)
        plt.show()

    #plt.loglog(freqs, np.abs(fft_phase_drive))
    #plt.show()

    spin_sig = obj.dat[:,0]
    
    fft = np.fft.rfft(spin_sig)

    z = ss.hilbert(spin_sig)
    phase = ss.detrend(np.unwrap(np.angle(z)))
  
    fft = np.fft.rfft(phase)
    #plt.loglog(freqs,np.abs(fft))
    #plt.show()

    phase_filt = bp_filt(phase,E_pm_freq, Ns, Fs, 10)

    x = np.arange(0,len(phase_filt))

    cut = (x > 700) & (x < len(phase_filt)-700) 
    
    #phase_filt = flattop(len(phase_filt)) * phase_filt
        
    z_phase_filt = ss.hilbert(phase_filt)
    
    pm_amp = np.abs(z_phase_filt)
    #pm_amp = pm_amp[cut]

    fft_phase = np.fft.rfft(phase_filt)
    
    plt.loglog(freqs,np.abs(fft_phase))
    plt.show()

    
    #plt.plot(phase)
    #plt.show()
    
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

if __name__ == "__main__":
    for k, path in enumerate(fils):
        paths = []
    	
        out_path = out_paths[k]
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
    	if save:
        	buf.make_all_pardirs(out_path)
    
    		
    	if end_path == 0:
    	    paths = paths[start_path:]
    	else:
    	    paths = paths[start_path:end_path]
    	
        for j, path in enumerate(paths):
            meas_name = path.split('/')[-2]
            save_path = out_path + '{}'.format(meas_name)
                
            print(save_path)
    	    if not overwrite and os.path.exists(save_path + '.npy'):		
    	    	continue
    	
    	    files, zero = bu.find_all_fnames(path)
    
            if end_file == 0:
                files = files[start_file:]
            else:
                files = files[start_files:end_files]
    
    	    #efield_amps = []
    	    #efield_amp_errs = []
    	    #spin_freqs = []	
    	    #spin_freq_errs = []
    	    #times = []
                #E_pm_freqs = []
                #pm_amp_avgs = []
    
            if parallel:
                print('finding spin freqs')
                spin_freq, spin_freq_err = zip(*Parallel(n_jobs=n_jobs)(delayed(find_spin_freq_parallel_new)(f) for i, f in enumerate(files[10:])))
                print('estimating field amplitude')
                e_field, e_field_err = zip(*Parallel(n_jobs=n_jobs)(delayed(find_efield_amp_parallel)(f) for i, f in enumerate(files)))

                print(e_field)
                
                

                #efield_amps.append(e_field)
                #efield_amp_errs.append(e_field_err)
                #spin_freqs.append(spin_freq)
                #spin_freq_errs.append(spin_freq_err)
                #could introduce error here if spin_freqs_arr is not same as efield_arr somehow
                #efield_amps.append(efield_arr[0])
                #efield_amp_errs.append(efield_arr[1])
                #spin_freqs.append(spin_freqs_arr[0])
                #spin_freq_errs.append(spin_freqs_arr[1])
            #print(spin_freqs_arr)
    
            
            elif not parallel:
                for i,f in enumerate(files[-5:]):
                    print(f)
                    buf.progress_bar(i, len(files))
    	            data = hd.hsDat(f)
    	            
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
    
            e_field = np.array(e_field)
            spin_freq = 2*np.pi*np.array(spin_freq)
            spin_freq_err = 2*np.pi*np.array(spin_freq_err)
            
            mask = spin_freq > 0 #2*np.pi*120.
            #plt.plot(efield_amps, spin_freqs)
            plt.scatter(e_field[mask], spin_freq[mask])
            plt.show()
            
    
            popt, pcov = curve_fit(line, e_field[mask], spin_freq[mask], sigma=spin_freq_err[mask])
            x_axis = np.linspace(0, np.amax(e_field), len(spin_freq)*10)
            
            #d = popt[0]**2 * (0.5 * 63.1e-15 * ((3/(4*np.pi)) * (63.1e-15/1.85e3))**(2./3.) *(1./1.602e-19) * 1e6) 
            d = popt[0]**2 
            d = popt[0]**2 * (0.5 * mass * (radius)**2 * (1./1.602e-19) * 1e6)
    
            filename = fils[0].split('/')[-2]
            plt.plot(e_field[mask], spin_freq[mask])
            #plt.plot(x_axis, sqrt(x_axis,*popt), label=r'd/I = {}'.format(round(d,2)) + ' $C m/kgm^{2}$' + ', {}'.format(filename))
            #plt.plot(x_axis, sqrt(x_axis,*popt), label=r'd = {}'.format(round(d,2)) + ' $e\mu m$' + ', {}'.format(filename))
            plt.plot(x_axis, line(x_axis,*popt))
            plt.xlabel('E-Field strength [V/m]')
            plt.ylabel(r'$\omega_{\phi}$')
            plt.legend()
            plt.show()
    
    
            #print(E_pm_freqs,pm_amp_avgs)
    
    
            print(popt)
            #print(E_pm_freqs,pm_amp_avgs)
    
            #popt,pcov = curve_fit(lorentzian, E_pm_freqs,pm_amp_avgs)
    
            #x = np.arange(0, E_pm_freqs[-1], len(E_pm_freqs)*100)
    
            #plt.plot(x, lorentzian(x,*popt))
            #plt.scatter(E_pm_freqs,pm_amp_avgs)
            #plt.show()
            
            #if save and parallel:
            #    np.save(save_path, np.array([
    
            if save and wobble:
    	        np.save(save_path, np.array([e_field,efield_amp_errs,\
    	    							spin_freq,spin_freq_err,times]))
    
            if save and libration:
                np.save(save_path, np.array([E_pm_freqs,pm_amp_avgs]))

                

#popt, pcov = curve_fit(sqrt, efield_amps, spin_freqs, sigma=spin_freq_errs)
#x_axis = np.linspace(efield_amps[0], efield_amps[-1], len(spin_freqs*2))
#
