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
from joblib import Parallel, delayed
import bead_util_funcs as buf
import bead_util as bu
import hs_digitizer as hd
import os 

matplotlib.rcParams['figure.figsize'] = [7,5]
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['agg.path.chunksize'] = 10000

fit = False

fils = ['/data/old_trap/20191105/bead4/phase_mod/change_dg/neg_2_0_to_0_7/']
fils = ['/data/old_trap/20191105/bead4/phase_mod/deriv_feedback/deriv_feedback_1/']
fils = ['/data/old_trap/20191204/bead1/spinning/deriv_feedback/20191211/dg_x/neg_0_005amp/']

base_name = '/data/old_trap/20191204/bead1/spinning/deriv_feedback/20191216/change_dg/'
base_name = '/data/old_trap/20191223/bead1/spinning/deriv_feedback_3/change_dg_3_coarse/' 
base_name = '/data/old_trap/20191223/bead1/spinning/deriv_feedback_5/'
base_name = '/data/old_trap/20191223/bead1/spinning/deriv_feedback_7/'
#base_name = '/data/old_trap/20191105/bead4/phase_mod/deriv_feedback/deriv_feedback_1/'
#base_name = '/data/old_trap/20191223/bead1/spinning/change_press_1/'
#base_name = '/data/old_trap/20191223/bead1/spinning/deriv_feedback_4/zero_dg/'
#base_name = '/data/old_trap/20200130/bead1/spinning/deriv_fb/no_dg_8Vpp/'
#base_name = '/data/old_trap/20200130/bead1/spinning/series_3/base_press/long_int/0_9_dg/'
#base_name = '/data/old_trap/20200130/bead1/spinning/series_4/base_press/change_phi_offset_3/long_int/'
#base_name = '/data/old_trap/20200130/bead1/spinning/series_5/change_phi_offset_0_to_0_3_dg/long_int/'
#base_name = '/data/old_trap/20200130/bead1/spinning/series_5/change_phi_offset_0_3_to_0_6_dg_1/long_int/'
base_name = '/data/old_trap/20200130/bead1/spinning/series_5/long_int_0_to_0_9_dg/'
base_name = '/data/old_trap/20200601/bead2/spinning/libration_cooling/change_phi_offset/change_phi_offset_1/'

#base_name = '/data/old_trap/20200130/bead1/spinning/series_5/long_int_3_to_4_dg/'
#base_name = '/data/old_trap/20190626/bead1/spinning/pramp/He/'
#base_name = '/data/old_trap/20190805/bead1/spinning/ringdown/reset_dipole_2/'
#base_name = '/data/old_trap/20200330/gbead3/spinning/ringdown/high_press/high_press_5/'
#base_name = '/data/old_trap/20200330/gbead3/spinning/long_int/base_press/50kHz_6Vpp_1/' 

fils = ['no_dg_0000','no_dg_0001','no_dg_0002','no_dg_0003','no_dg_0004',\
        'no_dg_0005','no_dg_0006','no_dg_0007','no_dg_0008','no_dg_0009']
fils = ['no_dg']

save_base_name = '/home/dmartin/Desktop/analyzedData/20191204/spinning/deriv_feedback/20191216/change_dg_window/crossp_psds/' 
save_base_name = '/home/dmartin/Desktop/analyzedData/20191223/spinning/deriv_feedback_5/change_dg/crossp_psds/'
save_base_name = '/home/dmartin/Desktop/analyzedData/20191223/spinning/tests/change_dg_5/crossp_psds/'
save_base_name = '/home/dmartin/Desktop/analyzedData/20191223/spinning/deriv_feedback_7/change_dg_highp/'
save_base_name = '/home/dmartin/Desktop/analyzedData/20191223/spinning/'
save_base_name = '/home/dmartin/Desktop/analyzedData/20191223/spinning/zero_dg/crossp_psds/'
save_base_name = '/home/dmartin/Desktop/analyzedData/20200130/spinning/deriv_fb/no_dg_8Vpp/crossp_psds/'
save_base_name = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_3/base_press/long_int/0_9_dg/crossp_psds/'
save_base_name = '/home/dmartin/Desktop/analyzedData/20200130/spinning/base_press/series_4/change_phi_offset_4/long_int/'
save_base_name = '/home/dmartin/Desktop/analyzedData/20200130/spinning/base_press/series_4/change_phi_offset_3/long_int/'
save_base_name = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_5/change_phi_offset_0_to_0_3_dg/long_int/' 
save_base_name = '/home/dmartin/Desktop/analyzedData/' + base_name.split('old_trap')[1] 

bu.make_all_pardirs(save_base_name)

files, zero, folders = bu.find_all_fnames(base_name, add_folders=True, sort_time=True)

if len(folders) > 1:
    folders.pop(0)


#folders = ['/data/old_trap/20200330/gbead3/spinning/long_int/base_press/50kHz_6Vpp_1/'
#           ]#'/data/old_trap/20200330/gbead3/spinning/long_int/base_press/50kHz_6Vpp_1/']

#high_p_files, zero = bu.find_all_fnames(folders[0], sort_time=True)
#base_p_files, zero1 = bu.find_all_fnames(folders[1], sort_time=True)

save_ind_files = True

overwrite = True

tabor_fac = 100.
spinning_freq = 50e3
filt = False

lib_freq = 240
bandwidth = 100

pm_bandwidth = 200
drive_pm_freq = 330

init_time=10

num_files = 40

plot = False
avg_psd = True
press_increase = False 
dg_increase = False 

data_ind = 0 
drive_ind = 1
inst_freq_ind = 2
ind = data_ind

threshold = 10e3
n_jobs = 5

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

def sqrt(P, a, f):
    return np.sqrt(np.cos(-np.arcsin(a*P*(1./(2*np.pi*f)**2)))*(2*np.pi*f)**2)
    #return np.sqrt(b**2. - a*x**2.)

def sine(x, A, f, c):
    return A * np.sin(2.*np.pi*f*x + c)

def line(x,a,b):
    return a * x + b


def avg_phase_psd_eff_parallel(f):   
    bad_file = False
    meas_name = f.split('/')[-2]
    raw_name = f.split('/')[-1].split('.')[0]
    
    obj = hd.hsDat(f)

    dg = obj.attribs['current_pm_dg']
    #pressures = obj.attribs['pressures']
    pressures = 0
    spin_sig = obj.dat[:,data_ind]
    Ns = obj.attribs['nsamp']
    Fs = obj.attribs['fsamp']

    tarr = np.arange(Ns)/Fs


    t_cut = tarr > init_time

    spin_sig = spin_sig[t_cut]
    
    z = ss.hilbert(spin_sig)

    phase = ss.detrend(np.unwrap(np.angle(z)))

    std_phase = np.std(phase)
     
    if std_phase > threshold:
        print('bad file')
        bad_file = True
    
    psd, freqs_ = mlab.psd(phase, NFFT=len(phase), Fs=Fs, window = np.hanning(len(phase)))

    #plt.loglog(freqs_, psd, label=r'$\phi_{crossp}(t)$ PSD')
    #plt.loglog(freqs_, psd_drive, label=r'$\phi_{Drive}(t)$ PSD')
    #plt.xlabel('Frequency [Hz]')
    #plt.ylabel(r'Instantaneous phase $\phi(t) [rad^{2}/Hz]$')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    mask = freqs_ < 1000
    
    cut = freqs_ < 70
    y_zero = psd
    y_zero[cut] = 0
    max_ind = np.argmax(y_zero)

    #plt.loglog(freqs_,y_zero)
    #plt.scatter(freqs_[max_ind], psd[max_ind])
    #plt.show()

    print(freqs_[max_ind])
    mask = (freqs_ >= freqs_[max_ind] - bandwidth*0.5) & (freqs_ <= freqs_[max_ind] + 0.5*bandwidth)


    #print(freqs_[mask])
    #plt.loglog(freqs_[mask], psd[mask])
    #plt.show()
    
    if save_ind_files:

        save_name = save_base_name + '/raw_curves_parallel/' + meas_name + '/{}'.format(raw_name)
        print(save_name) 
        bu.make_all_pardirs(save_name)

        np.savez(save_name, psd=psd[mask], freqs=freqs_[mask], Ns=Ns, Fs=Fs, num_files=num_files, bad_file=bad_file, dg=dg)   

folders.sort(key=lambda f: int(filter(str.isdigit,f)))
folders = folders[:]

if avg_psd:
    for i in range(len(folders)):
        
        files, zero = bu.find_all_fnames(folders[i], sort_time=True, ext='.h5')
        
        Parallel(n_jobs=n_jobs)(delayed(avg_phase_psd_eff_parallel)(f) for f in files) 
        meas_name = folders[i].split('/')[-1] 
        
