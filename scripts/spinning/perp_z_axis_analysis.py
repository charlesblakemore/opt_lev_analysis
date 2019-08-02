import numpy as np
import matplotlib.pyplot as plt
import hs_digitizer as hd
import bead_util_funcs as buf
import scipy.signal as ss
from amp_ramp_3 import filt

plot_close = False 
bandwidth = 2000.

path = '/daq2/20190626/bead1/spinning/wobble/after_pramp_series/perp_z_axis/2kHz_2Vpp/'
#path = '/daq2/20190626/bead1/spinning/pramp/Ar/50kHz_4Vpp_1/'

drive_freq = 2e3

files, zero = buf.find_all_fnames(path)

ext_freqs = []

for i, filename in enumerate(files):
	obj = hd.hsDat(filename)
	Ns = obj.attribs['nsamp']
	Fs = obj.attribs['fsamp']
	tarr = np.arange(Ns)/Fs
	
	filt_sig = filt(obj.dat[:,0],2*drive_freq,Ns,Fs,100)
	
	freqs = np.fft.rfftfreq(Ns, 1./Fs)
	fft = np.fft.rfft(filt_sig)
	fft_drive = np.fft.rfft(obj.dat[:,1])
	fft_sig = np.fft.rfft(obj.dat[:,0])


	if plot_close:
		mask = np.abs(freqs-2*drive_freq) < bandwidth/2.
		plt.semilogy(freqs[mask], np.abs(fft_drive)[mask], label='drive')
		plt.semilogy(freqs[mask], np.abs(fft_sig)[mask], label='$P_{perp}$')
	else:	
		plt.loglog(freqs,np.abs(fft_drive), label='drive')
		plt.loglog(freqs,np.abs(fft_sig), label='$P_{perp}$')
		#plt.loglog(freqs,np.abs(fft))
		#plt.show()
	
	z = ss.hilbert(filt_sig)
	phi = ss.detrend(np.unwrap(np.angle(z)))
	phi_fft = np.fft.rfft(phi)
	
	
	#f = np.argmax(np.abs(phi_fft))
	#ext_freqs.append(freqs[f])

	
	#filt_phi = filt(phi,freqs[f],Ns,Fs,100)

	#filt_phi_fft = np.fft.rfft(filt_phi)
	
	#plt.loglog(freqs,np.abs(filt_phi_fft))	
	
	#plt.scatter(freqs[f],np.abs(phi_fft[f]))
	#plt.loglog(freqs,np.abs(phi_fft))
	plt.ylabel('Amplitude [arb.]')
	plt.xlabel('Frequency [Hz]')
	plt.legend()
	plt.show()

	dphase_unsum = np.angle(fft_drive)
	sig_phase_unsum = np.angle(fft_sig)

	dphase = np.angle(np.sum(fft_drive))
	sig_phase = np.angle(np.sum(fft_sig))
	
	phase = sig_phase - 2.*dphase

	if phase > np.pi:
		phase -= 2*np.pi
	if phase < -np.pi:
		phase += 2*np.pi

	print(phase)
	plt.plot(freqs,dphase_unsum)
	plt.plot(freqs,sig_phase_unsum)
	plt.show()

	
print(ext_freqs)
