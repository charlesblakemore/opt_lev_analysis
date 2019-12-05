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

save = False

overwrite = True

#fils = ['/data/old_trap/20191105/bead4/phase_mod/change_dg/neg_2_0_to_0_7/']
#fils = ['/data/old_trap/20191105/bead4/phase_mod/deriv_feedback/deriv_feedback_1/']
#fils = ['/data/old_trap/20191017/bead1/spinning/pramp/He/50kHz_4Vpp_2/']
fils = ['/data/old_trap/20190905/bead1/spinning/ringdown_manual/100kHz_start_1/']

out_paths = ['/home/dmartin/Desktop/analyzedData/20191105/changing_pm_freq_7_8/']


files, zero = bu.find_all_fnames(fils[0])

files = files[0:]

tabor_fac = 100.
spinning_freq = 100e3
filt = True
avg_psd = True


pm_bandwidth = 200
drive_pm_freq = 330

num_files = 10


press_increase = False
dg_increase = True 
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

def avg_phase_psd(files, num_files=1):
    
    obj_init = hd.hsDat(files[0])
    Ns = obj_init.attribs['nsamp']
    Fs = obj_init.attribs['fsamp']
    
    freqs = np.fft.rfftfreq(Ns, 1./Fs)
    
    psd_sum = np.zeros((Ns/2) + 1)

    psds = []

    for i in range(len(files)):
        obj = hd.hsDat(files[i])

        spin_sig = obj.dat[:,0]

        if filt:
            spin_sig = bp_filt(spin_sig,2*spinning_freq,Ns,Fs,1000)

        fft = np.fft.rfft(spin_sig)

       # plt.loglog(freqs,np.abs(fft))
        #plt.show()

        z = ss.hilbert(spin_sig)
        phase = ss.detrend(np.unwrap(np.angle(z)))
        
        psd, freqs_ = mlab.psd(phase, NFFT=len(phase), Fs=Fs, window = mlab.window_none)

        psds.append(psd)

        plt.loglog(freqs,psd)
        #plt.show()

    plt.ylabel(r'PSD $[rad^{2}/Hz]$')
    plt.xlabel('Frequency [Hz]')
    plt.show()

    #avg = psd_sum/len(files)

    avg = np.mean(psds,axis=0)
    std = np.std(psds,axis=0)
    
    mask_high = 357
    mask_low = 210
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
    
    plt.ylabel(r'PSD $[rad^{2}/Hz]$')
    plt.xlabel('Frequency [Hz]')
    plt.show()
    
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
 
    


pressures = np.zeros((len(files), 3))
libration_freqs = np.zeros(len(files))

if avg_psd:
    files = files[:-(len(files)-num_files)]
    print(files)
    avg_phase_psd(files,num_files)

if press_increase:
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


