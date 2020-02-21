from amp_ramp_3 import bp_filt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import bead_util as bu
from scipy.optimize import curve_fit
from scipy import signal
from iminuit import Minuit

matplotlib.rcParams['figure.figsize'] = [7,5]
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['agg.path.chunksize'] = 10000

directory = '/home/dmartin/Desktop/analyzedData/20191204/spinning/deriv_feedback/20191216/change_dg_window/crossp_psds/' 
#directory = '/home/dmartin/Desktop/analyzedData/20191204/spinning/deriv_feedback/20191219/crossp_psds/'
#directory = '/home/dmartin/Desktop/analyzedData/20191223/spinning/deriv_feedback/change_dg_0_to_9_3/'
#directory = '/home/dmartin/Desktop/analyzedData/20191223/spinning/deriv_feedback/change_dg_0_to_0_15/crossp_psds/'
directory = '/home/dmartin/Desktop/analyzedData/20191223/spinning/deriv_feedback_5/change_dg/crossp_psds/'
directory = '/home/dmartin/Desktop/analyzedData/20191223/spinning/deriv_feedback_3/change_dg_3_fine/crossp_psds/'
directory = '/home/dmartin/Desktop/analyzedData/20191223/spinning/tests/change_dg_5/crossp_psds/'
directory = '/home/dmartin/Desktop/analyzedData/20191223/spinning/deriv_feedback_7/change_dg_highp/'
directory = '/home/dmartin/Desktop/analyzedData/20191223/spinning/zero_dg/'
directory = '/home/dmartin/Desktop/analyzedData/20200130/spinning/deriv_fb/no_dg/crossp_psds/'
directory = '/home/dmartin/Desktop/analyzedData/20200130/spinning/deriv_fb/no_dg_8Vpp/crossp_psds/'
directory = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_2/base_press/long_int_0_dg_2/crossp_psds/'
directory = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_2/base_press/long_int_8_5_dg/crossp_psds/'

base_directory = '/home/dmartin/Desktop/analyzedData/20200130/spinning/base_press/series_4/change_phi_offset_3/long_int/raw_curves/'

save_base_name = '/home/dmartin/Desktop/analyzedData/20200130/spinning/base_press/series_4/change_phi_offset_3/long_int/'

files, length, folders= bu.find_all_fnames(base_directory, add_folders=True)

folders.pop(0)
print(folders)
#files = ['/home/dmartin/Desktop/analyzedData/20191204/spinning/deriv_feedback/20191216/no_dg.npy']

#fit_many_in_sequency_psd arguments

dist_to_end_percentage = 0.001 
migrad_ncall = 10000

fix_A = False
fix_f0 = True
fix_g = False
fix_b = True

save_fit_p = True

###################################


filt = False

drive = True

fit = False

fit_bandwidth = 20

libration_freq = 370

freq_window_low = libration_freq-0.5*fit_bandwidth
freq_window_high = libration_freq+0.5*fit_bandwidth

bp_bandwidth = 100

files.sort(key=lambda f: int(filter(str.isdigit,f)))
file_num = 15 

def psd_lorentzian(x, A, f0, g, b):
    w = 2*np.pi*x
    w0 = 2*np.pi*f0
    #denom =( (((1./g)+c)*w)**2 + (w0**2 - w**2)**2)
    denom = (((g)*w)**2 + (w0**2 - w**2)**2)
    return (1/denom)*((A*g**2)) + b

def gaussian(x, mu, c, A):
    return np.exp(-(x-mu)**2) * A + c

A_avg_arr = []
f0_avg_arr = []
g_avg_arr = []
b_avg_arr = []
chi_sq_ndof_avg_arr = []

A_std_arr = []
f0_std_arr = []
g_std_arr = []
b_std_arr = []
chi_sq_ndof_std_arr = []
def fit_many_in_sequence_psd(filenames, dtep):
    
    meas_name = filenames[0].split('/')[-2]

    A_arr = []
    f0_arr = []
    g_arr = []
    b_arr = []
    chi_sq_ndof = []

    init_data = np.load(filenames[0], allow_pickle=True)

    init_psd = init_data['psd']

    dg = init_data['dg']

    N = 0 
    psd_sum = np.zeros_like(init_psd)

    
    def chi_sq(A, f0 , g, b):
        return np.sum(((y-psd_lorentzian(x, A, f0, g, b))/y_err)**2.)
    
    #def chi_sq(mu, c, A):
    #    return np.sum(((y-gaussian(x, mu, c, A))/y_err)**2.)

    for i, f in enumerate(filenames):
        data = np.load(f, allow_pickle=True)
        
        psd = data['psd']
        
        psd_sum += psd
        N += 1

    psd_means = psd_sum/N

    N = 0 
    diff_sum = np.zeros_like(init_psd)
    for i, f in enumerate(filenames):
        data = np.load(f)

        psd = data['psd']

        diff = (psd - psd_means)**2
        diff_sum += diff
        N += 1
    
    psd_stds = np.sqrt(diff_sum/N)
    y_err = psd_stds

    print(y_err)
    for i, f in enumerate(filenames):
        data = np.load(f, allow_pickle=True)

        psd = data['psd']
        freqs = data['freqs']

        y = psd
        x = freqs 

        end_mask = (freqs > freqs[-1]-(len(freqs)*dtep)) 

        max_ind = np.argmax(y)
        #plt.loglog(x, y)
        #plt.loglog(x[end_mask], y[end_mask])
        #plt.show()


        A_guess = 1e3
        f0_guess = x[max_ind]
        g_guess = 1
        b_guess = np.mean(y[end_mask])

        m=Minuit(chi_sq, A=A_guess, fix_A=fix_A, f0=f0_guess, fix_f0=fix_f0, g=g_guess, fix_g=fix_g, b=b_guess, fix_b=fix_b, print_level=1, errordef=1)


        m.migrad(ncall=migrad_ncall)
        p = [m.values['A'], m.values['f0'], m.values['g'], m.values['b']]
    
        print(p)

        A_arr.append(m.values['A'])
        f0_arr.append(m.values['f0'])
        g_arr.append(m.values['g'])
        b_arr.append(m.values['b'])
        chi_sq_ndof.append(chi_sq(*p)/len(y))

        x_arr = np.linspace(x[0], x[-1], len(x)*10)


        print('chi_sq', chi_sq(*p), len(y))
       
        #plt.loglog(x, y)
        #plt.loglog(x_arr, psd_lorentzian(x_arr, *p))
        #plt.loglog(x_arr, gaussian(x_arr, * p))
        #plt.show()

    
    if save_fit_p:
        save_name = save_base_name + meas_name + '_fit_params'
        print(save_name)
        bu.make_all_pardirs(save_name)

        np.savez(save_name, A_arr=A_arr, f0_arr=f0_arr, g_arr=g_arr, b_arr=b_arr, chi_sq_ndof=chi_sq_ndof, dg=dg)
        

        #    label1 = r'$\tau$' +' = {} s'.format(popt[2].round(2))
        #    #label2 = r'a' + ' = {} Hz'.format(popt[3].round(2))
        #    label2 = ''
        #    label3 = r'$f_{peak}$' + ' = {} Hz'.format(popt[1].round(2))

        #    full = label1 + ', ' + label2 + ', ' + label3

        #    x = np.linspace(freqs[mask][0], freqs[mask][-1], 10000)
        #    #leg = plt.legend()
        #    #leg.get_frame().set_linewidth(0.0)

        #    meas_name = filenames[i].split('/')[-1].split('.')[0]
        #    
        #    if False:#popt[2] > 100 or chi_sq_dof > 3:
        #        fig, ax = plt.subplots(2, 1, sharex=True)
        #        ax[0].semilogy(freqs[mask], psd[mask], label=meas_name)
        #        ax[0].semilogy(x,psd_lorentzian(x, *popt),label=full)
        #        #ax[0].semilogy(x,lorentzian(x, *popt[:-1]), label='lorentz') 

        #        ax[1].semilogy(freqs[mask], ((psd[mask]-f_arr)/sigma)**2 ) 
        #        plt.legend()

        #        plt.xlabel('Frequency [Hz]')
        #        ax[0].set_ylabel(r'PSD $[rad^{2}/Hz]$')
        #        ax[1].set_ylabel(r'$(\frac{y-f}{\sigma})^{2}$')
        #        plt.show()
        #if np.mean(chi_sq_arr) > 2:
        #    print(chi_sq_arr)
        #    raw_input()
        # 
        #A_means.append(np.mean(A))
        #A_stds.append(np.std(A))
        #freq_means.append(np.mean(freq_))
        #freq_stds.append(np.std(freq_))
        #damp_means.append(np.mean(damp))
        #damp_stds.append(np.std(damp))
        ##a_means.append(np.mean(a))
        ##a_stds.append(np.std(a))
        #b_means.append(np.mean(b))
        #b_stds.append(np.std(b))
        #chi_sq_means.append(np.mean(chi_sq_arr))
        #chi_sq_stds.append(np.std(chi_sq_arr))

    A_avg_arr.append(np.mean(A_arr))
    f0_avg_arr.append(np.mean(f0_arr))
    g_avg_arr.append(np.mean(g_arr))
    b_avg_arr.append(np.mean(b_arr))
    chi_sq_ndof_avg_arr.append(np.mean(chi_sq_ndof))
    
    A_std_arr.append(np.std(A_arr))
    f0_std_arr.append(np.std(f0_arr))
    g_std_arr.append(np.std(g_arr))
    b_std_arr.append(np.std(b_arr))
    chi_sq_ndof_std_arr.append(np.std(chi_sq_ndof))
        
    file_arr = np.arange(len(filenames))

    #plt.scatter(file_arr,a_means, label='a')
    #plt.xlabel('file number')
    #plt.ylabel('a [Hz]')
    #plt.show()
   
    #fig, ax = plt.subplots(3,1,sharex=True)

    #plt.scatter(file_arr,freq_means, label='freq')
    #plt.errorbar(file_arr, freq_means, yerr=freq_stds, fmt='.')
    #plt.xlabel('file number')
    #plt.ylabel('Libration frequency [Hz]')
    #plt.show()
    
    #plt.scatter(file_arr, damp_means, label='damp')
    #ax[0].errorbar(file_arr, damp_means, yerr=damp_stds, fmt='.')
    #plt.xlabel('file number')
    #ax[0].set_ylabel(r'$\gamma$ [Hz]')
    
    #plt.scatter(file_arr, A_means, label='A')
    #ax[1].errorbar(file_arr, A_means, yerr=A_stds, fmt='.')
    #plt.xlabel('file number')
    #ax[1].set_ylabel('A')
    
    
    #plt.scatter(file_arr, chi_sq_means, label=r'$\chi^{2}$')
    #ax[2].errorbar(file_arr, chi_sq_means, yerr=chi_sq_stds, fmt='.')
    #plt.xlabel('file number')
    #ax[2].set_ylabel(r'$\chi^{2}/DOF$')
    #plt.show()

    #plt.errorbar(file_arr, b_means, yerr=b_stds, fmt='.')
    #plt.xlabel('file number')
    #plt.ylabel(r'b')
    #plt.show()

for i, folder in enumerate(folders):
    files, zero = bu.find_all_fnames(folder, ext='.npz')

    fit_many_in_sequence_psd(files, dist_to_end_percentage)

file_arr = np.arange(len(folders))

plt.scatter(file_arr, g_avg_arr)
plt.yscale('log')
plt.show()

#plt.errorbar(dg, damp_avg_arr, yerr=damp_std_arr, fmt='o')
##plt.scatter(dg, damp_avg_arr)
#plt.ylabel(r'$\gamma$ [Hz]')
#plt.xlabel('dg scale factor [arb.]')
#plt.yscale('log')
#plt.grid(b=True, which='minor', axis='both')
#plt.grid(b=True ,which='major', axis='both')
#plt.show()

#fit_many_in_sequence_psd_mult(files[:], freq_window_low, freq_window_high)
#fit_two(files[0], freq_window_low, freq_window_high)
