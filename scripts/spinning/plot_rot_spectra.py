from hs_digitizer import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import window_func as window
import bead_util as bu
from amp_ramp_3 import bp_filt, lp_filt, hp_filt
from scipy import signal
from transfer_func_util import damped_osc_amp 
from scipy.optimize import curve_fit
from memory_profiler import profile
from memory_profiler import memory_usage
from plot_phase_vs_pressure_many_gases import build_full_pressure


import gc

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
files = ['/data/old_trap/20191204/bead1/spinning/deriv_feedback/20191213/test_pm/turbombar_powfb_xyzcool_test_sig_50kHz_300Hz_pm_in_25kHz_dds_out_1dg_10down_samp_num_0.h5']

directory = '/data/old_trap/20191204/bead1/tests/dg/no_dg/'
directory = '/data/old_trap/20200312/TIA_change/'
directory = '/data/old_trap/20200322/gbead1/spinning/ringdown/terminal_velocity_check_1/'
directory = '/data/old_trap/20200322/gbead1/spinning/wobble/25kHz_xz_1/wobble_0000/'
directory = '/data/old_trap/20200322/gbead1/spinning/wobble/wobble_init/wobble_0000/'
directory = '/data/old_trap/20200322/gbead1/spinning/wobble/faster_rot/wobble_0000/'
directory = '/data/old_trap/20200307/gbead1/spinning/wobble/wobble_0/wobble_0000/'
directory = '/data/old_trap/20200322/gbead1/spinning/ringdown/terminal_velocity_check_1/'
directory = '/data/old_trap/20200322/gbead1/spinning/ringdown/110kHz_2/'
#directory = '/data/old_trap/20200330/gbead3/spinning/wobble/
#directory = '/data/old_trap/20200330/gbead3/spinning/wobble/110kHz_xy_1/wobble_0000/'
directory = '/data/old_trap/20200330/gbead3/spinning/pramp/pramp_0/50kHz_3Vpp_xy_1/'
#directory = '/data/old_trap/20190626/bead1/spinning/pramp/N2/50kHz_4Vpp_1/'
#directory = '/data/old_trap/20190626/bead1/spinning/wobble/wobble_many/wobble_0000/'
#directory = '/data/old_trap/20200330/gbead3/spinning/pramp/pramp_0/50kHz_5Vpp_xy_1/'
#directory = '/data/old_trap/20200330/gbead3/spinning/efield_mod/test/'
#directory = '/data/old_trap/20200330/gbead3/spinning/ringdown/high_press/high_press_3/50kHz_8Vpp_xy_ringdown_6/'
#directory = '/data/old_trap/20200330/gbead3/spinning/ringdown/high_press/high_press_6/110kHz_8Vpp_xy_ringdown_2/'
#directory = '/data/old_trap/20181204/bead1/high_speed_digitizer/pramp/50k_zhat_1vpp_0/'
#directory = '/data/old_trap/20200330/gbead3/spinning/test/check_rot_3/'
#directory = '/data/old_trap/20200330/gbead3/spinning/ringdown/high_press_1/50kHz_8Vpp_0.02torr_ringdown_1/'
#directory = '/data/old_trap/20200330/gbead3/spinning/ringdown/high_press/high_press_5/110kHz_8Vpp_xy_long_int_1/'
#directory = '/data/old_trap/20200330/gbead3/spinning/long_int/high_press/50kHz_4Vpp_2/'
directory = '/data/old_trap/20200130/bead1/spinning/series_5/change_phi_offset_0_3_to_0_6_dg_1/change_phi_offset/change_phi_offset_0000/'
directory = '/data/old_trap/20200601/bead2/spinning/libration_cooling/long_int/long_int_3_basep_light_off/'
directory = '/data/old_trap/20200601/bead2/spinning/libration_cooling/change_phi_offset/change_phi_offset_1/change_phi_0005/'
#directory = '/data/old_trap/20200601/bead2/spinning/libration_cooling/vary_E_amp/0_3dg/'
#directory = '/data/old_trap/20200601/bead2/spinning/libration_cooling/long_int/tests/'
directory = '/data/old_trap/20200924/bead1/spinning/dds_8Vpp_test/'


skip = False
mask_on = True
file_range = True
start = 0#380
end = 10#382

skip_files = 10
NFFT_fac = 10

labels = ['Positive','Negative','Zero']
labels = ['']
scales = [1]

if use_dir:
	files, lengths = bu.find_all_fnames(directory, sort_time=True)



files = files[:]

drive_ind = 1
data_ind = 0
hilbert_ind = data_ind

freq_spinning = 50.e3
wind_width = 5320.
bandwidth = 200 
libration_freq =  350.

time_low = 4.02
time_high = 4.15


spin_bw = 150000e3
freq = 50e3
#libration_freq = 269
#bandwidth = 80

if plot_many_perp or plot_freq_range and not plot_files:
	colors = bu.get_color_map(len(files), 'inferno')
	plot_p_perp = True


obj0 = hsDat(files[0])
t0 = obj0.attribs['time']

def sine(x, A, f, c):
    return A * np.sin(2*np.pi*f*x) + c

def lorentzian(x, A, f0, g):

    w0 = 2*np.pi*f0
    w = 2*np.pi*x
    denom = (w0**2 - w**2)**2 + (w*g)**2 

    return (A/denom)

def exp(x, g, A, B):
    return A*np.exp(-g*x) + B 

def fit_phase_decay(filename, t_low, t_high, lib_freq, bw, filt=False, x_arr_size_fac=5):

    obj = hsDat(filename)
    Ns = obj.attribs['nsamp']
    Fs = obj.attribs['fsamp']

    print(Ns/Fs)

    freqs = np.fft.rfftfreq(Ns, 1./Fs)
    tarr = np.arange(Ns)/Fs
    
    spin_sig = obj.dat[:, data_ind]
    
    print(spin_sig.shape, len(tarr))
    
    z = signal.hilbert(spin_sig)


    amp = np.abs(z)
    phase = signal.detrend(np.unwrap(np.angle(z)))
    phase_phase = signal.hilbert(phase)
    amp_phase = np.abs(phase_phase)


    #phase /= amp_phase

    #if j == 0:
    #    phase = np.gradient(phase/(2*np.pi), 1./Fs)
    #else:
    #    phase /= 2*np.pi
    #    phase *= Fs

    if filt:
        phase = bp_filt(phase, lib_freq, Ns, Fs, bw)

    label = [r'MS Instaneous Phase ($\phi(t)$)', 'fit']

    mask = (tarr > t_low) & (tarr < t_high)

    z_phase = signal.hilbert(phase)
    phase_cut = phase[mask]
    tarr_cut = tarr[mask]

    amp_phase = np.abs(z_phase)
    amp_phase = amp_phase[mask]

    #plt.plot(tarr_cut, phase_cut)
    #plt.plot(tarr_cut, amp_phase)
    #plt.show()

    x_arr = np.linspace(tarr_cut[0], tarr_cut[-1], len(tarr_cut)*x_arr_size_fac)


    grad = np.gradient(amp_phase/(2*np.pi), 1./Fs)

    plt.plot(grad)
    plt.show()

    g_guess = np.amax(grad/amp_phase)
    A_guess = np.amax(amp_phase)
    B_guess = amp_phase[-1]
    
    p0 = [1, A_guess+B_guess, B_guess] 
    #popt, pcov = curve_fit(exp, tarr_cut, amp_phase, p0=p0)

    #plt.plot(tarr_cut, phase_cut, label=label[0])
    #plt.scatter(x_arr, exp(x_arr, *popt), label=label[1])
    print(p0)

    plt.plot(tarr_cut, amp_phase)
    plt.plot(tarr_cut, grad/amp_phase)
    plt.legend()
    plt.show()


@profile
def plot_pm(files, libration_freq, bw, corr, mask_on=False, inst_freq=True):
    obj_init = hsDat(files[0])
    Ns = obj_init.attribs['nsamp']
    Fs = obj_init.attribs['fsamp']

    print(Ns/Fs)

    freqs = np.fft.rfftfreq(Ns, 1./Fs)
    tarr = np.arange(Ns)/Fs
    
    for i in range(len(files)):
        print(files[i])
        obj = hsDat(files[i])

        #spin_sig = obj.dat[:,2]

        #plt.plot(spin_sig)
        #plt.show()
        ##
        #fft = np.fft.rfft(spin_sig)
       
        #plt.loglog(freqs, np.abs(fft))
        #plt.show()
        
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()

        if corr:
            fig4, ax4 = plt.subplots()

        
        for j in range(2):
            spin_sig = obj.dat[:,j]
            tarr = np.arange(len(spin_sig))/Fs

            print(spin_sig.shape, len(tarr))
            
            if mask_on:
                mask = (tarr > 0) & (tarr < 13)
                spin_sig = spin_sig[mask]
                tarr = tarr[mask]
                
            Ns = len(spin_sig)

            freqs = np.fft.rfftfreq(Ns, 1./Fs)

            fft = np.fft.rfft(spin_sig)
            
            print('hilbert')
            #plt.loglog(freqs, np.abs(fft))
            z = signal.hilbert(spin_sig)

            
            print('amp')
            amp = np.abs(z)
            print('phase')
            phase = signal.detrend(np.unwrap(np.angle(z)))
            phase_phase = signal.hilbert(phase)
            amp_phase = np.abs(phase_phase)
          
            
            #phase /= amp_phase
            
            if inst_freq:

                label = [r'MS Instaneous Frequency ($\frac{d\phi}{dt}$)', \
                        r'Drive Instaneous Phase ($m(t)=a\frac{d\phi}{dt}$)']

                print('diff')
                if j == data_ind:

                    phase = np.gradient(phase)
                
            else:
                label = [r'MS Instaneous Phase ($\phi(t)$)',\
                     r'Drive Instaneous Phase ($m(t)=a\frac{d\phi}{dt}$)']
            print('filt')
            if filt:
                phase_filt = bp_filt(phase, libration_freq, Ns, Fs, bw)
                fft_filt = np.fft.rfft(phase_filt) 
            else:
                phase_filt=phase
                fft_filt=np.fft.rfft(phase_filt)

            fft = np.fft.rfft(phase)

            sum_phase = np.angle(np.sum(fft))
            print(sum_phase)
            print('plots')
            ax1.plot(tarr, phase_filt, label=label[j])
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Instaneous Phase [arb.]')
            #ax1.set_ylabel('Instaneous Frequency [rad/s]')
            ax1.legend()
            
            #ax2.plot(tarr, phase/amp_phase, label='Norm. ' + label[j])
            #ax2.set_xlabel('Time [s]')
            #ax2.set_ylabel('Norm. Instaneous Frequency [rad/s]')
            #ax2.legend()

            if False:#True:
                mask = (freqs > libration_freq-bw/2) & (freqs < libration_freq+bw/2)
            else: 
                mask = freqs > 0
            ax2.loglog(freqs[mask], np.abs(fft[mask]), label='FFT. ' + label[j])
            ax2.set_xlabel('Frequency [Hz]')
            ax2.set_ylabel('FFT of Instaneous Frequency [arb.]')
            ax2.legend()

            ax3.loglog(freqs[mask], np.abs(fft_filt[mask]), label='FFT. ' + label[j])
            ax3.set_xlabel('Frequency [Hz]')
            ax3.set_ylabel('FFT of Instaneous Frequency [arb.]')
            ax3.legend()

        plt.show()
        gc.collect()

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

def plot_phi_fb_out(filenames):
    for i, f in enumerate(filenames):
        obj = hsDat(f)
        
        print(f)

        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        tarr = np.arange(Ns)/Fs
        
        drive = obj.dat[:,drive_ind]
        sig = obj.dat[:,2]
        
        drive_fft = np.fft.rfft(drive)

        print('sig & sig_filt')
        
        freq_cut = 12e3

        label1=r'Instaneous frequency $\frac{d\phi}{dt}$'
        label2='Filtered instanteous frequency, {} kHz cut off'.format(freq_cut/1000.)

    

        plt.plot(tarr, sig, label=label1 )

        sig_filt = lp_filt(sig, freq_cut, Ns, Fs)

        plt.plot(tarr, sig_filt, label=label2)
        plt.legend()
        plt.show()
        
        gc.collect()

        window = np.hanning(len(sig))

        fft = np.fft.rfft(sig)
        fft_filt = np.fft.rfft(sig_filt)
        freqs = np.fft.rfftfreq(Ns, 1./Fs)

        plt.loglog(freqs, np.abs(fft), label=label1)
        plt.loglog(freqs, np.abs(fft_filt), label=label2)
        plt.legend()
        plt.show()
        
        gc.collect()

        plt.loglog(freqs, np.abs(drive_fft))
        plt.show()
        gc.collect()

def plot_many_spectra_lib(filenames, lib_freq, bw, pressure_ramp=False, skip=False, mask_on=False):
    colors = bu.get_color_map(len(filenames), 'inferno')
    
    pressures = np.zeros((len(files), 3))

    if pressure_ramp:
        for i, f in enumerate(files):

            obj = hsDat(f)

            print(f, obj.attribs['start_pm_dg'], obj.attribs['stop_pm_dg'])


            pressures[i,:] = obj.attribs['pressures']

        func, press = build_full_pressure(pressures, plot=True)

        press_mbar = 1.33322 * press

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for i, f in enumerate(filenames):
        if skip:
            if i%10 or i ==0:
                continue
        
        obj = hsDat(f)
                
        print(f)
        

        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        tarr = np.arange(Ns)/Fs
        freqs = np.fft.rfftfreq(Ns, 1./Fs)
        
        rot_sig = obj.dat[:,data_ind]

        psd, freqs_ = mlab.psd(rot_sig, NFFT=len(rot_sig), Fs=Fs, window = np.hanning(len(rot_sig)))

        ax1.loglog(freqs_, psd)
            
        window = np.hanning(len(rot_sig))
        

        
        rot_sig_wind = rot_sig*window

        z = signal.hilbert(rot_sig)
        z_wind = signal.hilbert(rot_sig_wind)

        phase = signal.detrend(np.unwrap(np.angle(z)))
        phase_wind = signal.detrend(np.unwrap(np.angle(z_wind))) 

        psd, freqs_ = mlab.psd(phase, NFFT=len(phase), Fs=Fs, window = np.hanning(len(phase)))
        psd_wind, freqs_ = mlab.psd(phase_wind, NFFT=len(phase), Fs=Fs, window = np.hanning(len(phase)))
        phase_fft = np.fft.rfft(phase)

        print(Ns/Fs)
        if mask_on:
            mask = (freqs > lib_freq-(bw/2)) & (freqs < lib_freq+(bw/2))
            if piressure_ramp:
                label='{} mbar'.format(press_mbar[i].round(2))

                ax2.semilogy(freqs[mask], np.abs(psd[mask]), color=colors[i], label=label)
                plt.legend()
            else:
                ax2.semilogy(freqs[mask], np.abs(psd[mask]), color=colors[i], label='no wind') 
                ax2.semilogy(freqs[mask], np.abs(psd_wind[mask]), color=colors[i], label='wind') 
        else:
            mask = freqs > 0
            if pressure_ramp:
                label='{} mbar'.format(press_mbar[i].round(2))

                plt.loglog(freqs[mask], np.abs(psd[mask]), color=colors[i], label=label)
                plt.legend()
 
            else:
                ax2.semilogy(freqs[mask], np.abs(psd[mask]), color=colors[i], label='no wind')
                ax2.semilogy(freqs[mask], np.abs(psd_wind[mask]), color=colors[i], label='wind')

        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel(r'PSD of $\phi$ [$rad^{2}$/Hz]')
    plt.show()

def plot_many_spectra(filenames, freq, bw, pressure_ramp=False, skip=False, mask_on=False):

    if file_range:
        #if end >= len(filenames): 
        #    print('bad end')
        #    end = len(filenames)-1

        filenames = filenames[start:end]

    colors = bu.get_color_map(len(filenames), 'inferno')
    
    

    fig1, ax1 = plt.subplots(2,1, sharex=False)
    for i, f in enumerate(filenames):
        if skip:
            if i%skip_files or i == 0:
                continue
            
        obj = hsDat(f)
                
        print(f)
        
        

        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        tarr = np.arange(Ns)/Fs
        freqs = np.fft.rfftfreq(Ns, 1./Fs)
       
        #plt.plot(tarr[tarr < 10], obj.dat[:,1][tarr < 10])
        #plt.show()
        print('pressures', obj.attribs['pressures'])
        print(Ns, Fs)
        rot_sig = obj.dat[:,data_ind]
        drive_sig = obj.dat[:,drive_ind]

        filt_sig = bp_filt(rot_sig, 12e3, Ns, Fs, 2000)
        #filt_sig = rot_sig
        fft = np.fft.rfft(filt_sig)


        z =  signal.hilbert(filt_sig)
        phase = np.unwrap(np.angle(z))

        int_freq = np.gradient(phase)*(1./(2*np.pi)) *Fs
        #plt.plot(tarr, int_freq)
        #plt.show()
        #plt.loglog(freqs, np.abs(fft))

        #plt.show()
        psd, freqs_ = mlab.psd(rot_sig, NFFT=len(rot_sig)/NFFT_fac, Fs=Fs, window = np.hanning(len(rot_sig)/NFFT_fac))

        psd_drive, freqs_ = mlab.psd(drive_sig, NFFT=len(drive_sig)/NFFT_fac, Fs=Fs, window = np.hanning(len(drive_sig)/NFFT_fac))

        freqs = freqs_
        if mask_on:
            mask = (freqs_ > freq-(bw/2)) & (freqs_ < freq+(bw/2))
            mask_drive = (freqs > 0.5*freq-(bw/2)) & (freqs < freq*0.5+(bw/2))
            ax1[0].semilogy(freqs_[mask]/1000, psd[mask], color=colors[i], label='cross polarized light')
            ax1[1].semilogy(freqs_[mask_drive]/1000, psd_drive[mask_drive], color=colors[i], label='drive')
            #ax1[0].loglog(freqs[mask], psd[mask], color=colors[i], label='cross polarized light')
            #ax1[1].loglog(freqs[mask_drive], psd_drive[mask_drive], color=colors[i], label='drive')
        else:
            ax1[0].loglog(freqs_/1000, np.sqrt(psd), color=colors[i], label='cross polarized light')
            ax1[1].loglog(freqs_/1000, np.sqrt(psd_drive), color=colors[i], label='drive')
            #ax1[0].set_xlabel('Frequency [Hz]')
        ax1[0].set_ylabel(r'PSD [arb.]')
        ax1[1].set_ylabel(r'PSD [arb.]')
        ax1[1].set_xlabel('Frequency [kHz]')
    ax1[0].set_title('Cross polarized light')
    ax1[1].set_title('Drive')
    fig1.tight_layout()
    plt.show()

    psd, freqs_ = mlab.psd(signal.detrend(phase), NFFT=len(phase)/NFFT_fac, Fs=Fs, window = np.hanning(len(phase)/NFFT_fac))

    plt.loglog(freqs_, np.sqrt(psd))
    
    plt.show()



#for i, f in enumerate(files):
#    fit_phase_decay(f, time_low, time_high, libration_freq, bw=bandwidth, filt=True, x_arr_size_fac=10)
#plot_phi_fb_out(files)
#plot_pm(files[:], libration_freq, bandwidth, corr=False)
def plot_long_int(files):
    for i, f in enumerate(files):
        obj = hsDat(f)

        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        freqs = np.fft.rfftfreq(Ns, 1./Fs)

        crossp = obj.dat[:,0]
        
        print(Ns/Fs)
        for j in range(len(crossp)/10):

            print(j)
            crossp_cut = crossp[j*(len(crossp)/10):]
            
            Ns = len(crossp_cut)

            freqs = np.fft.rfftfreq(Ns, 1./Fs)

            rot_cut = (freqs > 50e3) & (freqs < 51e3)
            
            rot_fft = np.fft.rfft(crossp_cut)

            plt.semilogy(freqs[rot_cut], np.abs(rot_fft[rot_cut]))
        
        plt.show()

        #z = signal.hilbert(crossp)
        #phase = signal.detrend(np.unwrap(np.angle(z)))

        #lib_cut = (freqs > 400) & (freqs < 500)

        #lib_fft = np.fft.rfft(phase)

        #plt.semilogy(freqs[lib_cut], np.abs(lib_fft[lib_cut]))
        #plt.show()
        
#plot_long_int(files)    
#plot_pm(files, libration_freq, bandwidth, False)
#plot_many_spectra(files[:], libration_freq, bandwidth)

def plot_drive(files):
    for i, f in enumerate(files):
        obj = hsDat(f)
        
        print(f)


        Fs = obj.attribs['fsamp']
        Ns = obj.attribs['nsamp']

        tarr = np.arange(Ns)/Fs
        drive = obj.dat[:,drive_ind]
            
        crossp = obj.dat[:,data_ind]

        freqs = np.fft.rfftfreq(obj.attribs['nsamp'], 1./obj.attribs['fsamp'])
        
        fft = np.fft.rfft(drive)
        
        psd_drive, freqs = mlab.psd(drive, NFFT=len(drive), Fs=Fs, window = np.hanning(len(drive)))
        psd_crossp, freqs = mlab.psd(crossp, NFFT=len(drive), Fs=Fs, window = np.hanning(len(drive)))

        z_crossp = signal.hilbert(crossp)
        z = signal.hilbert(drive)
        phase = signal.detrend(np.unwrap(np.angle(z)))
        phase_crossp = np.unwrap(np.angle(z_crossp))

        phase_crossp = bp_filt(phase_crossp, 380, Ns, Fs, 180)

        inst_freq = np.gradient(phase_crossp)/(2*np.pi) * Fs

        psd_phase_crossp, freqs = mlab.psd(phase_crossp, NFFT=len(phase), Fs=Fs, window = np.hanning(len(phase)))
        psd_phase, freqs = mlab.psd(phase, NFFT=len(phase), Fs=Fs, window = np.hanning(len(phase)))
        
        plt.plot(tarr,inst_freq)
        #plt.plot(phase_crossp)
        #plt.loglog(freqs, psd_phase_crossp)
        #plt.loglog(freqs, psd_phase)
        #plt.loglog(freqs, crossp)
        #plt.loglog(freqs, psd_drive)
        #plt.show()

        #plt.loglog(freqs, np.abs(fft))
    plt.show()

#plot_drive(files)
#plot_many_spectra(files[:], 2*freq, spin_bw, skip=skip, mask_on=mask_on)
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

        t = np.arange(Ns)/Fs
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
                gc.collect()

                
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
                gc.collect()

	elif not plot_many_perp or not plot_freq_range:
                if plot_raw_sig:
                    plt.plot(obj.dat[:,data_ind])
                    plt.show()
                sig = obj.dat[:,data_ind]
                drive_sig = obj.dat[:,drive_ind]
                #sig = sig[t >10]
                #drive_sig = drive_sig[t > 10]
                #Ns = len(sig)

                freqs = np.fft.rfftfreq(Ns, 1./Fs)
                if filt:
                    sig = bp_filt(sig,2*freq_spinning, Ns, Fs, bandwidth) 
                


                plt.plot(sig)
                plt.show()
                fft = np.fft.rfft(sig)
		drive_fft = np.fft.rfft(drive_sig)
		
                
		plt.loglog(freqs,np.abs(fft), label='$P_{\perp}$')
                
                if plot_drive:
                    
                    plt.loglog(freqs,np.abs(drive_fft), label='drive')

		plt.ylabel('Amplitude [arb.]')
		plt.xlabel('Frequency [Hz]')
		plt.legend()
		plt.show()
		gc.collect()
                
        if hilbert:
            tarr = np.arange(0, Ns/Fs, 1/Fs)

            #plt.plot(tarr, obj.dat[:,drive_ind])
            #plt.show()
            
            #sig = obj.dat[:,hilbert_ind]

            if filt: 
               sig = bp_filt(sig, 2*freq_spinning, Ns, Fs, bandwidth)
            z = signal.hilbert(sig)

            amp = np.abs(z)
            phase = signal.detrend(np.unwrap(np.angle(z)))
            
            #phase = bp_filt(phase, 120, Ns, Fs, 200)

            window = np.hanning(Ns)

            #phase *= window

            fc = 542
            bw = 75
           
            psd, freqs_ = mlab.psd(phase, NFFT=len(phase), Fs=Fs, window = np.hanning(len(phase)))

            fft_phase = np.fft.rfft(phase)

            mask = (freqs < fc+bw) & (freqs > fc-bw)

            freqs_mod = freqs[mask]-fc

            #plt.semilogy(freqs_mod, np.abs(fft_phase)[mask])
            #plt.xlabel('Frequency [Hz]')
            #plt.ylabel(r'$\phi$ [rad]')
            #plt.grid(b=True, which='minor', axis='x')
            #plt.show()

            #
            ##phase = bp_filt(phase,fc , Ns, Fs, bw)

            #fft_phase = np.fft.rfft(phase)
            #
            #plt.semilogy(freqs_mod, np.abs(fft_phase)[mask])
            #plt.xlabel('Frequency [Hz]')
            #plt.ylabel(r'$\phi$ [rad]')
            #plt.show()

            #z_phase = signal.hilbert(phase)
            #phase_phase = np.unwrap(np.angle(z_phase))
            #inst_freq = np.gradient(phase_phase)/(2*np.pi) * Fs 

            #t_mask = (tarr < 1.3) & (tarr > 0.3)

            #p0 = [20, 25, 570]
            #popt, pcov = curve_fit(sine, tarr[t_mask], inst_freq[t_mask], p0)

            #t = np.linspace(tarr[t_mask][0],tarr[t_mask][-1],len(tarr[t_mask])*10)

            #label = r'A$\sin(2\pi ft)$ + c, ' + 'A = {}, f = {}, c = {}'.format(popt[0].round(1), popt[1].round(1), popt[2].round(1))
            #
            #plt.plot(tarr[t_mask],inst_freq[t_mask], label='data')
            #plt.plot(t, sine(t, *popt), label=label)
            #plt.xlabel('Time [s]')
            #plt.ylabel(r'$\frac{1}{2\pi}\frac{d\phi}{dt}$ [Hz]')
            #plt.legend()
            #plt.show()
            #gc.collect()

            #fft_phase = np.fft.rfft(phase)
            #fft_amp = np.fft.rfft(amp)

            #plt.plot(tarr,amp)
            #plt.plot(tarr,sig)
            #plt.xlabel('Time [s]')
            #plt.ylabel('Amplitude [arb.]')
            #plt.show()
            #gc.collect()
            #
            #sig = np.unwrap(sig)

            #plt.plot(tarr, sig)
            #plt.show()
            #gc.collect()

            #plt.loglog(freqs, np.abs(fft_amp))
            #plt.xlabel('Frequency [Hz]')
            #plt.ylabel('Amplitude [arb.]')
            #plt.show()
            #gc.collect()

            plt.plot(tarr, phase)
            plt.xlabel('Time [s]')
            plt.ylabel('Phase Amplitude [arb.]')
            plt.show()
            gc.collect()

            print(Ns/Fs)
            plt.loglog(freqs,psd)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel(r'PSD of $\phi$ [$rad^{2}$/s]')
            plt.grid(b=True, which='minor', axis='x')
            plt.grid(b=True, which='major', axis='y')
            plt.show()
            gc.collect()
