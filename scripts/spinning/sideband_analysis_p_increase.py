import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import curve_fit
from amp_ramp_3 import flattop
from amp_ramp_3 import hp_filt
from amp_ramp_3 import bp_filt
from ring_down_analysis_v3 import track_frequency
from plot_phase_vs_pressure_many_gases import build_full_pressure
import bead_util_funcs as buf
import bead_util as bu
import hs_digitizer as hd
import os 

matplotlib.rcParams['figure.figsize'] = [7,5]
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['agg.path.chunksize'] = 10000

fit = False
save = True

overwrite = True

fils = ['/data/old_trap/20191105/bead4/phase_mod/change_dg/neg_2_0_to_0_7/']
fils = ['/data/old_trap/20191105/bead4/phase_mod/deriv_feedback/deriv_feedback_1/']
fils = ['/data/old_trap/20191204/bead1/spinning/deriv_feedback/20191211/dg_x/neg_0_005amp/']

base_name = '/data/old_trap/20191204/bead1/spinning/deriv_feedback/20191216/change_dg/'
#base_name = '/data/old_trap/20191105/bead4/phase_mod/deriv_feedback/deriv_feedback_1/'
fils = ['no_dg_0000','no_dg_0001','no_dg_0002','no_dg_0003','no_dg_0004',\
        'no_dg_0005','no_dg_0006','no_dg_0007','no_dg_0008','no_dg_0009']
fils = ['no_dg']

save_base_name = '/home/dmartin/Desktop/analyzedData/20191204/spinning/deriv_feedback/20191216/change_dg_window/crossp_psds/' 

bu.make_all_pardirs(save_base_name)

out_paths = ['/home/dmartin/Desktop/analyzedData/20191204/dg_xz/']

files, zero, folders = bu.find_all_fnames(base_name, add_folders=True, sort_time=True)
folders.pop(0)

overwrite = True

tabor_fac = 100.
spinning_freq = 25e3
filt = False


pm_bandwidth = 200
drive_pm_freq = 330

num_files = 40

plot = False
avg_psd = True
press_increase = False
dg_increase = False 

data_ind = 0 
drive_ind = 1
ind = data_ind

def lorentzian(x, A, x0, g, B):
    return A * (1./ (1 + ((x - x0)/g)**2)) + B

def psd_lorentzian(x, A, f0, g, c):
    w = 2*np.pi*x
    w0 = 2*np.pi*f0
    denom = (((g+c)*w)**2 + (w0**2 - w**2)**2)

    return (A/denom) * (g)

def psd_lorentzian_cut(x, A, f0, g, mask_low, mask_high):
    cut = (x > mask_low) & (x < mask_high)

    w = 2*np.pi*x
    w0 = 2*np.pi*f0
    denom = ((g*w)**2 + (w0**2 - w**2)**2)

    return ((A/denom) )[cut]


def gauss(x, A, mean, std):
    return  A * np.exp(-(x-mean)**2/(2.*std**2))

def sqrt(x, a, b):
    return np.sqrt(b**2. - a*x**2.)

def sine(x, A, f, c):
    return A * np.sin(2.*np.pi*f*x + c)
def line(x,a,b):
    return a * x + b

def avg_phase_fft(files):
    obj_init = hd.hsDat(files[0])
    Ns = obj_init.attribs['nsamp']
    Fs = obj_init.attribs['fsamp']

    freqs = np.fft.rfftfreq(Ns, 1./Fs)
    tarr = np.arange(Ns)/Fs
    #psd_sum = np.zeros((Ns/2) + 1)

    psds = []
    for i in range(len(files)):
        print(files[i])
        obj = hd.hsDat(files[i])

        spin_sig = obj.dat[:,ind]


        if filt:
            spin_sig = bp_filt(spin_sig,2*spinning_freq,Ns,Fs,1800)

        fft = np.fft.rfft(spin_sig)

        phase = np.angle(fft)


def avg_phase_psd(files):
    
    obj_init = hd.hsDat(files[0])
    Ns = obj_init.attribs['nsamp']
    Fs = obj_init.attribs['fsamp']
    
    freqs = np.fft.rfftfreq(Ns, 1./Fs)
    tarr = np.arange(Ns)/Fs
    #psd_sum = np.zeros((Ns/2) + 1)

    psds = []
    for i in range(len(files)):
        print(files[i])    
        obj = hd.hsDat(files[i])

        spin_sig = obj.dat[:,ind]


        if filt:
            spin_sig = bp_filt(spin_sig,2*spinning_freq,Ns,Fs,1800)
        
        fft = np.fft.rfft(spin_sig)

        #plt.loglog(freqs,np.abs(fft))
        #plt.ylabel(r'FFT Drive [arb units]')
        #plt.xlabel('Frequency [Hz]')
        #plt.show()

        z = ss.hilbert(spin_sig)
        phase = ss.detrend(np.unwrap(np.angle(z)))
        
        #plt.plot(tarr, phase)
        #plt.show()

        psd, freqs_ = mlab.psd(phase, NFFT=len(phase), Fs=Fs, window = np.hanning(len(phase)))
        
        psds.append(psd)

        #plt.loglog(freqs,psd,label='{}'.format(files[i].split('/')[-1]))
        
        #plt.show()

#    plt.ylabel(r'PSD $[rad^{2}/Hz]$')
#    plt.xlabel('Frequency [Hz]')
    #plt.legend()
#    plt.show()

    #avg = psd_sum/len(files)

    avg = np.mean(psds,axis=0)
    std = np.std(psds,axis=0)
    
    mask_high = 860
    mask_low = 810
    mask = (freqs > mask_low) & (freqs < mask_high)

    max_ind = np.argmax(avg[mask])
    freq_guess = freqs[mask][max_ind]
    
    g_high = 1e2
    g_low = 1e-4


    #plt.semilogy(freqs[mask],avg[mask])
    #plt.show()

    print(freq_guess)

    p0 = [1e6,freq_guess,1e0, mask_low, mask_high]
    bounds = ([1e5,freq_guess-10,1e-2,mask_low-100,mask_high-100],[1e10,freq_guess+10,1e3,mask_low+100, mask_high+100])
    
#    popt, pcov = curve_fit(psd_lorentzian_cut, freqs[mask], avg[mask], p0=p0, bounds=bounds)
#
#    print(popt)
#
#    x = np.linspace(freqs[mask][0],freqs[mask][-1],100000)
#
#    plt.loglog(freqs, avg)
#    plt.loglog(x, psd_lorentzian_cut(x, *popt))
#    plt.ylabel(r'Amplitude [rad^{2}/Hz]')
#    plt.xlabel('Frequency [Hz]')
#    
#    plt.show()
#   
    if plot:
        if fit:
            res = (freqs[1]-freqs[0])/Ns
#            
#
            p0 = [1e6, freq_guess, 1e-1, 1e-1]
            bounds = ([0,freq_guess-res,1e-2, 1e-2], [1e10,freq_guess+res, 1e4, 1e4])

            popt, pcov = curve_fit(psd_lorentzian, freqs[mask], avg[mask], p0=p0, bounds=bounds)

            print(popt)
   
            label1 = r'$1/\tau$' +' = {} Hz'.format(popt[2].round(2))
            label2 = r'a' + ' = {} Hz'.format(popt[3].round(2))
            label3 = r'$\omega_{peak}$' + ' = {} radHz'.format(popt[1].round(2))

            full = label1 + ', ' + label2 + ', ' + label3

            x = np.linspace(freqs[mask][0], freqs[mask][-1], 10000)
            plt.loglog(freqs, avg)
            plt.loglog(x, psd_lorentzian(x, *popt),label=full)

            leg = plt.legend()
            leg.get_frame().set_linewidth(0.0)
        else:
            plt.loglog(freqs, avg)

            leg = plt.legend()
            leg.get_frame().set_linewidth(0.0)

        plt.ylabel(r'PSD $[rad^{2}/Hz]$')
        plt.xlabel('Frequency [Hz]')
        plt.show()
    
    num_files = len(psds)
    return freqs, avg, Ns, Fs, num_files, np.array(psds), std

def extract_libration_freq(obj):
    Ns = obj.attribs['nsamp']
    Fs = obj.attribs['fsamp']
    freqs = np.fft.rfftfreq(Ns,1./Fs)

    spin_sig = obj.dat[:,0]
  
    if filt:
        spin_sig = bp_filt(spin_sig,2*spinning_freq,Ns,Fs,1000)


    fft = np.fft.rfft(spin_sig)

    plt.loglog(freqs,np.abs(fft))
    plt.show()

    z = ss.hilbert(spin_sig)
    phase = ss.detrend(np.unwrap(np.angle(z)))
  
    fft_phase = np.fft.rfft(phase)

    plt.loglog(freqs,np.abs(fft_phase))
    plt.show()

    phase_filt = bp_filt(phase, 300, Ns, Fs,300)

    fft_phase = np.fft.rfft(phase_filt)
   
    plt.loglog(freqs, np.abs(fft_phase))
    plt.show()

    #plt.plot(phase_filt)
    #plt.show()
   
    z_phase = ss.hilbert(phase_filt)
    
    inst_freq = np.diff(np.unwrap(np.angle(z_phase)))/(2*np.pi) * Fs
    
    samples = np.arange(Ns-1)
    
    cut = (samples > 700) & (samples < Ns-1000)   
    inst_freq = inst_freq[cut]
    
    inst_freq_mean = np.mean(inst_freq)
    print(inst_freq_mean)
    

    #plt.plot(inst_freq)
    #plt.show()
    
    #plt.plot(phase_filt, label=r'$\phi$')
    #plt.plot(pm_amp, label=r'Envelope of $\phi$')
    #plt.ylabel(r'Amplitude')
    #plt.xlabel('Sample Number')
    #plt.legend()
    #plt.show()

    #plt.loglog(freqs,np.abs(fft_phase))
    #plt.show()

    libration_freq = inst_freq_mean

    return libration_freq

def change_dg(obj):
    Ns = obj.attribs['nsamp']
    Fs = obj.attribs['fsamp']
    freqs = np.fft.rfftfreq(Ns, 1./Fs)

    spin_sig = obj.dat[:,0]

    fft = np.fft.rfft(spin_sig)

    plt.loglog(np.abs(fft))
    plt.show()

    z = ss.hilbert(spin_sig)
    phase = ss.detrend(np.unwrap(np.angle(z)))

    fft_phase = np.fft.rfft(phase)

    plt.loglog(freqs,np.abs(fft_phase))
    plt.show()
    
    max_ind = np.argmax(np.abs(fft_phase))
    
    phase_filt = bp_filt(phase, 300, Ns, Fs,200)
    

    #for i in range(5):
    #    psd, freqs = mlab.psd(phase, NFFT=len(phase_filt)*(2*(i+1)), Fs=Fs, window=mlab.window_none)

    #    plt.loglog(freqs,psd,label= 2*(i+1))
        
    #plt.legend()
    #plt.show()

    
    psd, freqs = mlab.psd(phase, NFFT=len(phase_filt), Fs=Fs, window=mlab.window_none)

    mask = (freqs > 280) & (freqs < 380)
    
    max_ind = np.argmax(psd[mask])
    freq_guess = freqs[mask][max_ind]
    
    bw = 20

    mask_low = freq_guess-bw/2.
    mask_high = freq_guess+bw/2.
    
    mask = (freqs > mask_low) & (freqs < mask_high)
    plt.semilogy(freqs[mask],psd[mask])
    plt.show()

    
    print(freq_guess)
    
    #p0 = [1e6,freq_guess,1e-1,mask_low,mask_high]  
    #bounds = ([1e6,mask_low,1e-4,mask_low-bw,mask_high-bw],[1e8,mask_high,1e1,mask_low+bw,mask_high+bw])  
    #popt, pcov = curve_fit(psd_lorentzian_cut, freqs[mask], psd[mask], p0=p0, bounds=bounds)

   
    p0 = [1e6,freq_guess,1e-1]  
    bounds = ([1e6,mask_low,1e-4],[1e10,mask_high,1e1])  
    popt, pcov = curve_fit(psd_lorentzian, freqs[mask], psd[mask], p0=p0, bounds=bounds)

    print(popt)
    
    
    x = np.linspace(freqs[mask][0],freqs[mask][-1],100000)
   
    

    plt.loglog(freqs, psd)  
    plt.loglog(x, psd_lorentzian(x, *popt))
    plt.ylabel(r'Amplitude [rad^{2}/Hz]')
    plt.xlabel('Frequency [Hz]')
    #plt.loglog(freqs, p.abs(fft_phase))
    plt.show()
 
    
folders.sort(key=lambda f: int(filter(str.isdigit,f)))

if avg_psd:
    #for i in range(len(files)):
    for i in range(len(folders)):
        
        print(folders[i])
        files, zero = bu.find_all_fnames(folders[i], sort_time=True, ext='.h5')
        meas_name = folders[i].split('/')[-1] 
       
        if ind == data_ind:
            save_name = save_base_name + meas_name
            print(save_name)
        
        elif ind == drive_ind:
            save_name = save_base_name + meas_name + '_drive_psds'
            print('drive_psds', save_name)

        if not os.path.exists(save_name + '.npz') or overwrite:
            
            freqs, avgs, Ns, Fs, num_files, psds, std = avg_phase_psd(files)
            
            if save: 
                np.savez(save_name, freqs=freqs, avgs=avgs, std=std, Ns=Ns, Fs=Fs, num_files=num_files, psds=psds)
        else:
            print(save_name + '.npz')
            print('File exits. Skipping.')
            continue
if press_increase:
    pressures = np.zeros((len(files), 3))
    libration_freqs = np.zeros(len(files))

    for i in range(len(files)):
        obj = hd.hsDat(files[i])
        
        pressures[i,:] = obj.attribs["pressures"]
        libration_freqs[i] = extract_libration_freq(obj)  
       
        
    func, press = build_full_pressure(pressures, plot=True)
        
    data = np.array([press,libration_freqs])
    
    popt, pcov = curve_fit(sqrt, press, libration_freqs)
    print(popt)
    
    plt.plot(press, sqrt(press, *popt))
    plt.plot(press, libration_freqs)
    plt.show()
    
    print(data)
    if raw_input('save?') == '1':
        np.save('/home/dmartin/Desktop/analyzedData/20191105/phase_mod/change_press/change_press_2/change_press_2.npy', data)

if dg_increase:
    for i in range(len(files)):
        obj = hd.hsDat(files[i])
        
        #extract_libration_freq(obj)
        libration_freqs[i] = change_dg(obj)#extract_libration_freq(obj)


