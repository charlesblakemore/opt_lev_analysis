from hs_digitizer import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import window_func as window
import bead_util as bu
from amp_ramp_3 import bp_filt
from scipy import signal
from transfer_func_util import damped_osc_amp 
from scipy.optimize import curve_fit

mpl.rcParams['figure.figsize'] = [7,5]
mpl.rcParams['figure.dpi'] = 150
#scales = [1,1,0.1]
#labels = ['FS w/ buffer', 'GS w/ buffer', 'GS w/o buffer']

plot_many_perp = False 
plot_files = False # plots files chosen by user
plot_freq_range = False
use_dir = True
plot_drive = True
plot_raw_sig = False
plot_many_phi_fft = False
hilbert = True

filt = False

#files = ['/data/old_trap/20191105/bead4/phase_mod/deriv_feedback/deriv_feedback_1/turbombar_powfb_xyzcool_mod_pos_0.h5',
#        '/data/old_trap/20191105/bead4/phase_mod/deriv_feedback/deriv_feedback_1/turbombar_powfb_xyzcool_mod_neg_0.h5',
#         '/data/old_trap/20191105/bead4/phase_mod/deriv_feedback/deriv_feedback_1/turbombar_powfb_xyzcool_mod_no_dg_1_0.h5']


files = ['/data/old_trap/20191204/bead1/spinning/change_pressure/xy/turbombar_powfb_xyzcool_xy_3Vpp_0.h5','/data/old_trap/20191204/bead1/spinning/change_pressure/xy/turbombar_powfb_xyzcool_xy_3Vpp_0_1.h5','/data/old_trap/20191204/bead1/spinning/change_pressure/xy/turbombar_powfb_xyzcool_xy_3Vpp_0_40.h5']
#files = ['
files = ['/data/old_trap/20191204/bead1/spinning/deriv_feedback/20191213/test_pm/turbombar_powfb_xyzcool_test_sig_50kHz_300Hz_pm_in_25kHz_dds_out_1dg_10down_samp_num_0.h5']

directory = '/data/old_trap/20191204/bead1/tests/dg/no_dg/'
directory = '/data/old_trap/20191204/bead1/spinning/deriv_feedback/20191210/dg/pos_0_0025_amp/'
directory = '/data/old_trap/20191204/bead1/spinning/change_pressure/xy/'
directory = '/data/old_trap/20191204/bead1/spinning/low_f_sidebands/25kHz_8Vpp_to_2Vpp_1/'
directory = '/data/old_trap/20191105/bead4/phase_mod/change_press/change_press_2/'
directory = '/data/old_trap/20191204/bead1/spinning/deriv_feedback/20191211/dg_xz/pos_0_005amp_base_press/' 
directory = '/data/old_trap/20191204/bead1/spinning/deriv_feedback/20191216/change_dg/change_dg_0000/'
directory = '/data/old_trap/20191220/phi_fb_test/'
#directory = '/data/old_trap/20191204/bead1/spinning/deriv_feedback/20191216/tests/srs_filter/'

labels = ['Positive','Negative','Zero']
labels = ['']
scales = [1]

if use_dir:
	files, lengths = bu.find_all_fnames(directory, sort_time=True)



files = files[-2:]
drive_ind = 1
data_ind = 0
hilbert_ind = data_ind

freq_spinning = 25.e3
wind_width = 5320.
bandwidth = 1.e3

if plot_many_perp or plot_freq_range and not plot_files:
	colors = bu.get_color_map(len(files), 'inferno')
	plot_p_perp = True


obj0 = hsDat(files[0])
t0 = obj0.attribs['time']

fig, ax = plt.subplots(figsize=(7,4), dpi=150)

def sine(x, A, f):
    return A * np.sin(2*np.pi*f*x)

def lorentzian(x, A, f0, g):

    w0 = 2*np.pi*f0
    w = 2*np.pi*x
    denom = (w0**2 - w**2)**2 + (w*g)**2 

    return (A/denom)

def plot(filename, ind, hilbert_=True):
    obj = hsDat(filename)

    Ns = obj.attribs['nsamp']
    Fs = obj.attribs['fsamp']
    freqs = np.fft.rfftfreq(Ns, 1./Fs)

    fft = np.fft.rfft(obj.dat[:,ind])

    z = signal.hilbert(obj.dat[:,ind])

    amp = np.abs(z)

    fft_amp = np.fft.rfft(amp)
    
    if hilbert_:
        phase = signal.detrend(np.unwrap(np.angle(z)))

        fft_phase = np.fft.rfft(phase)
        
        mask = (freqs > 320) & (freqs < 350)

        

        psd, freqs_ = mlab.psd(phase, NFFT=len(phase), Fs=Fs, window=mlab.window_none)

        max_ind_guess = np.argmax(psd[mask])
        freq_guess = freqs[mask][max_ind_guess]
        amp_guess = psd[mask][max_ind_guess]

        print(freq_guess)

        p0 = [1e8,freq_guess,0.1]
       
        popt,pcov = curve_fit(lorentzian,freqs_[mask],psd[mask],p0=p0)

        print(popt)

        freqs_x = np.linspace(freqs_[mask][0], freqs_[mask][-1], len(freqs_[mask])*10)
        
        #plt.loglog(freqs_[mask], psd[mask])
        plt.loglog(freqs_, psd)
        plt.loglog(freqs_x, lorentzian(freqs_x, *popt))
        plt.show()


def plot_many_phase_fft(files, varied_e_amp=False, varied_pressure=False, plot_many=False):
    if plot_many:
        for i in range(len(files)):
                obj = hsDat(files[i])
                
                try:
                        M = len(obj.dat[:,1])
                except AttributeError:
                        continue
        
                Ns = obj.attribs['nsamp']
                Fs = obj.attribs['fsamp']
                freqs = np.fft.rfftfreq(Ns,1./Fs)
                tarr = np.arange(Ns)/Fs
                
                label = ''


                if varied_phase:
                   pressures = obj.attribs['pressures']
                   print(pressures) 
                   
                   label = '{:0.3f}'.format(pressures[0]) + ' Torr'

                if varied_e_amp:

                   drive_sig = obj.dat[:,drive_ind]
                   
                   drive_sig =  bp_filt(drive_sig,freq_spinning, Ns, Fs, bandwidth)
        
                   z = signal.hilbert(drive_sig)
        
                   amp = np.abs(z)
        
                   zeros = np.zeros(Ns)
                   Vpp = np.array([zeros,zeros,zeros,2*amp,zeros,zeros,zeros,zeros])
                   ef = bu.trap_efield(100 *Vpp)
        
                   mask = (tarr > 0.5) & (tarr < 1.0)
        
                   avg_amp = np.mean(ef[0][mask])
                   print(avg_amp)
                   
                   label = '{:02.0f}'.format(avg_amp/1000.) + ' kV/m'
                
                
                   sig = obj.dat[:,hilbert_ind]
        
                   if filt:
                      sig = bp_filt(sig, 2*freq_spinning, Ns, Fs, bandwidth)
                   z = signal.hilbert(sig)
        
                   amp = np.abs(z)
                   phase = signal.detrend(np.unwrap(np.angle(z)))
        
                   fft_phase = np.fft.rfft(phase)
                   fft_amp = np.fft.rfft(amp)
        
                   #plt.plot(tarr,amp)
                   #plt.plot(tarr,sig)
                   #plt.xlabel('Time [s]')
                   #plt.ylabel(r'$P_{\perp}$ signal [arb.]')
                   #plt.show()

                   plt.loglog(freqs, np.abs(fft_amp), color=colors[i], label=label)
                   plt.xlabel('Frequency [Hz]')
                   plt.ylabel(r'Amplitude [arb.]')
                   plt.title(r'Instantaneous Amplitude FFT [arb.]')
                   plt.grid(b=True, which='minor', axis='x')
                   plt.grid(b=True, which='major', axis='y')
                   plt.legend()            
        plt.show()
    else:
        for i in range(len(files)):
            obj = hsDat(files[i])

            try:
                    M = len(obj.dat[:,1])
            except AttributeError:
                    continue

            Ns = obj.attribs['nsamp']
            Fs = obj.attribs['fsamp']
            freqs = np.fft.rfftfreq(Ns,1./Fs)
            tarr = np.arange(Ns)/Fs

            sig = obj.dat[:,hilbert_ind]

            if filt:
               sig = bp_filt(sig, 2*freq_spinning, Ns, Fs, bandwidth)
            z = signal.hilbert(sig)

            amp = np.abs(z)
            phase = signal.detrend(np.unwrap(np.angle(z)))

            fft_phase = np.fft.rfft(phase)
            fft_amp = np.fft.rfft(amp)

            #plt.plot(tarr,amp)
            #plt.plot(tarr,sig)
            #plt.xlabel('Time [s]')
            #plt.ylabel(r'$P_{\perp}$ signal [arb.]')
            #plt.show()

            plt.loglog(freqs, np.abs(fft_phase))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(r'Amplitude [arb.]')
            plt.title(r'$\phi$ FFT [arb.]')
            plt.grid(b=True, which='minor', axis='x')
            plt.grid(b=True, which='major', axis='y')
            plt.legend()
            plt.show()

if plot_many_phi_fft:
    plot_many_phase_fft(files,False,False,False)

if plot_files:

    files = files
    for i in range(len(files)):
        plot(files[i],data_ind, True)

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [arb.]')
    plt.grid(b=True, which='minor', axis='x')
    plt.grid(b=True, which='major', axis='y')
    plt.legend()
    plt.show()

    raw_input()



for i in range(len(files)):
        
        print(files[i])
	obj = hsDat(files[i])

	try:
		M = len(obj.dat[:,1])
	except AttributeError:
		continue
	
	Ns = obj.attribs['nsamp']
	Fs = obj.attribs['fsamp']
	freqs = np.fft.rfftfreq(Ns,1./Fs)
        
        print(Ns)

	t_curr = obj.attribs['time']
	if plot_files:
        	fig1, ax1 = plt.subplots(figsize=(7,4), dpi=150)
		fig2, ax2 = plt.subplots(figsize=(7,4), dpi=150)
		fig3, ax3 = plt.subplots(figsize=(7,4), dpi=150)


		for j in range(len(labels)):
			obj = hsDat(files[j])
			
                        sig = obj.dat[:,0]

                        if filt:
                            sig = bp_filt(sig,2*freq_spinning, Ns, Fs, bandwidth)

			fft = np.fft.rfft(sig)
			drive_fft = np.fft.rfft(obj.dat[:,drive_ind])
			
                        
                        ax1.loglog(freqs,scales[j]* np.abs(drive_fft),label='{}*'.format(scales[j]) + labels[j])
			ax1.set_title('Drive FFT')
			ax1.legend()
			ax1.set_ylabel('Amplitude [arb.]')
			ax1.set_xlabel('Frequency [Hz]')
		
			ax2.loglog(freqs, np.abs(fft), label=labels[j])
			ax2.set_title('$P_{\perp}$ FFT')
			ax2.legend()
			ax2.set_ylabel('Amplitude [arb.]')
			ax2.set_xlabel('Frequency [Hz]')

                        ax3.plot(sig)


		plt.show()


                
	if plot_many_perp:
		fft = np.fft.rfft(obj.dat[:, 0])

		plt.loglog(freqs, np.abs(fft), c=colors[i], label='$P_{\perp}$')
		plt.ylabel('Amplitude [arb.]')
		plt.xlabel('Frequency [Hz]')
	 
	if plot_freq_range:
		fft = np.fft.rfft(obj.dat[:, 0])
		drive_fft = np.fft.rfft(obj.dat[:,1])

		freq_range = np.abs(freqs - 2*freq_spinning) < wind_width 
	
		if plot_p_perp:	
			plt.semilogy(freqs[freq_range], np.abs(fft)[freq_range],label='{:2f}s'.format(1.e-9*(t_curr-t0)), c=colors[i])
			plt.title('$P_{\perp}$')
		if plot_drive:
			plt.semilogy(freqs[freq_range], np.abs(drive_fft)[freq_range],label='{:2f}s'.format(1.e-9*(t_curr-t0)), c=colors[i])
			plt.title('Drive')
                plt.ylabel('Amplitude [arb.]')
		plt.xlabel('Frequency [Hz]')
		#plt.grid(b=True, which='both', axis='x')
                plt.legend(loc='lower right')
                plt.show()

	elif not plot_many_perp or not plot_freq_range:
                if plot_raw_sig:
                    plt.plot(obj.dat[:,data_ind])
                    plt.show()
                sig = obj.dat[:,data_ind]

                if filt:
                    sig = bp_filt(sig,2*freq_spinning, Ns, Fs, bandwidth) 
                    
		fft = np.fft.rfft(sig)
		drive_fft = np.fft.rfft(obj.dat[:,drive_ind])
		
                
		plt.semilogy(freqs,np.abs(fft), label='$P_{\perp}$')
                
                if plot_drive:
                    
                    plt.semilogy(freqs,np.abs(drive_fft), label='drive')

		plt.ylabel('Amplitude [arb.]')
		plt.xlabel('Frequency [Hz]')
		plt.legend()
		plt.show()
		
                
		if plot_freq_range:	
			freq_range = np.abs(freqs - 2*freq_spinning) < wind_width 
			
			plt.semilogy(freqs[freq_range], np.abs(fft)[freq_range],label='$P_{\perp}$')
		        if plot_drive:
                            plt.semilogy(freqs[freq_range], np.abs(drive_fft)[freq_range],label='drive')
			
			plt.ylabel('Amplitude [arb.]')
			plt.xlabel('Frequency [Hz]')

                        plt.legend()
                        plt.show()
    
        if hilbert:
            tarr = np.arange(0, Ns/Fs, 1/Fs)

            #plt.plot(tarr, obj.dat[:,hilbert_ind])
           # plt.show()
            
            sig = obj.dat[:,hilbert_ind]

            if filt: 
               sig = bp_filt(sig, 2*freq_spinning, Ns, Fs, bandwidth)
            z = signal.hilbert(sig)

            amp = np.abs(z)
            phase = signal.detrend(np.unwrap(np.angle(z)))
            
            fft_phase = np.fft.rfft(phase)
            fft_amp = np.fft.rfft(amp)

            plt.plot(tarr,amp)
            plt.plot(tarr,sig)
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [arb.]')
            plt.show()
            
            sig = np.unwrap(sig)

            plt.plot(tarr, sig)
            plt.show()
                
            plt.loglog(freqs, np.abs(fft_amp))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude [arb.]')
            plt.show()

            plt.plot(tarr, phase)
            plt.xlabel('Time [s]')
            plt.ylabel('Phase Amplitude [arb.]')
            plt.show()

            plt.loglog(freqs,np.abs(fft_phase))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude [arb.]')
            plt.grid(b=True, which='minor', axis='x')
            plt.grid(b=True, which='major', axis='y')
            plt.show()

