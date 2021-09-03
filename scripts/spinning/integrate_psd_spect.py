import bead_util as bu
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.optimize import curve_fit

#directory = '/home/dmartin/Desktop/analyzedData/20200330/gbead3/spinning/long_int/base_press/
directory = '/home/dmartin/Desktop/analyzedData/20200330/gbead3/spinning/long_int/base_press/50kHz_6Vpp_1/raw_curves/'
#directory = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/series_5/long_int_0_to_0_9_dg/raw_curves/long_int_0009/'

files, lengths = bu.find_all_fnames(directory, ext='.npz')

I = 2e-25
k_bolt = 1.38e-23

sums=[]

bandwidth = 100

a_guess = 0
b_guess = 2e-6
fix_a = True
fix_b = False

plot_fit = False



def line(x, a, b):
    return a*x + b

def find_msd_sums(files):
    '''Take a dataset of integrations at the same derivative gain value and find the sum of the PSD over a frequency window
    This estimates the noise floor by fitting a line and subtracting that off each point on the PSD in the summation.'''
    init_data = np.load(files[0], allow_pickle=True)
    psd = init_data['psd']
    freqs = init_data['freqs']
    
    length = len(psd)

    
    T_eff_arr = []
    T_eff_err_arr = []
    psd_sum = np.zeros_like(psd)
    diff_sum = np.zeros_like(psd)
    rms_arr = []

    N = 0
    
    for i, f in enumerate(files):
        data = np.load(f, allow_pickle=True)

        pressures = data['pressures']
        pressure = pressures[0]
        psd = 2*data['psd_drive']
        
        if length != len(psd):
            continue
        
        psd_sum += psd
        N += 1

    psd_means = psd_sum/N

    for i, f in enumerate(files):
        data = np.load(f)

        psd = 2*data['psd_drive']

        if length != len(psd):
            continue

        diff = (psd - psd_means)**2
        diff_sum += diff
    
    psd_stds = np.sqrt(diff_sum/N)
    psd_unc = psd_stds/np.sqrt(N)

    for i, f in enumerate(files):
        data = np.load(f)
        psd = 2*data['psd_drive'] #Times 2 for single-sided psd. Checked this by generating a test sine signal, summing the PSD around 
                            #the sine frequency, and checking that the ampitude is the same as what what is set 
       
        if length != len(psd):
            continue

        freqs = data['freqs']
        dg = data['dg']
        
        #plt.loglog(freqs, psd)
        #plt.show()
        df = freqs[1]-freqs[0]
    
        max_ind = np.argmax(psd)
        peak_freq = freqs[max_ind]  
        mask = (freqs  < peak_freq + bandwidth*0.5) & (freqs > peak_freq - bandwidth*0.5)
        
        end_mask = freqs > peak_freq + 15# bandwidth

        if len(psd[end_mask]) > 0:
            print(len(psd[end_mask]))
            b = np.mean(psd[end_mask])
        else:
            b = 0
        
        cum_sum, cum_sum_sq_errs = integral_approx(psd[mask]-b, psd_unc, df, freqs)
        
        cum_sum_sq_err = np.sum(cum_sum_sq_errs)
        T_eff = I * (2.*np.pi*peak_freq)**2 * (1./k_bolt) * np.sum(cum_sum)
        T_eff_sq_err = (I * (2.*np.pi*peak_freq)**2 * (1./k_bolt) )**2. * cum_sum_sq_err 
        rms = np.sqrt(np.abs(np.sum(cum_sum)))

        print(np.sqrt(np.sum(cum_sum)))
        T_eff = 8.85e-12*(4e-3)**3*(49.5e3)**2*(1./(2*1.38e-23))*np.sum(cum_sum)
        #T_eff = np.sum(cum_sum)
        #if T_eff < 0:
        print(T_eff, peak_freq, np.sum(cum_sum),b)

        x = np.linspace(freqs[0],freqs[-1], len(freqs)*10)
        if plot_fit:
            plt.semilogy(freqs, psd)
            #plt.semilogy(freqs[mask], psd[mask])
            plt.semilogy(freqs[end_mask], psd[end_mask])

            plt.show()
    
        sum_psd = np.sum(psd[mask]-b)
        T_eff_arr.append(T_eff)
        T_eff_err_arr.append(T_eff_sq_err)
        rms_arr.append(rms)
    plt.hist(T_eff_arr, bins='auto', label='{} dg scale factor, {} torr'.format(dg, round(pressure,2)))
    plt.legend()
    plt.xlabel(r'$T_{eff}$')
    plt.ylabel('Counts')
    #plt.savefig(fig_dir + '{}_dg.png'.format(dg))
    #plt.close()
    plt.show()

    plt.hist(rms_arr, bins='auto', label='{} dg scale factor, {} torr'.format(dg, round(pressure,2)))
    plt.legend()
    plt.xlabel(r'$\sqrt{\langle \phi^{2} \rangle}$')
    plt.ylabel('Counts')
    plt.show()

    T_eff_avg = np.mean(T_eff_arr)
    T_eff_err = np.sum(T_eff_err_arr)
    T_eff_err = T_eff_err/len(T_eff_err_arr)
    T_eff_std = np.std(T_eff_arr)
    T_eff_std /= np.sqrt(len(T_eff_arr))

    T_eff_err += T_eff_std**2
    T_eff_err = np.sqrt(T_eff_err)

    #return sum_avg, cum_sum, cum_sum_errs
    return sum_avg, T_eff_avg, T_eff_err

def integral_approx(psd, psd_unc, df, freqs):
    cum_sum = []
    cum_sum_sq_errs = []

    for i, x_n in enumerate(psd):
        x_n_unc = psd_unc[i]
        x_n_1 = psd[(i+1)%len(psd)]
        x_n_1_unc = psd_unc[(i+1)%len(psd)]
        
        cum_sum.append(df * 0.5 * (x_n + x_n_1))
        cum_sum_sq_errs.append((df*0.5)**2 * (x_n_unc**2 + x_n_1_unc**2))

    
    return cum_sum, cum_sum_sq_errs




sum_avg, T_eff_avg, T_eff_err = find_msd_sums(files)
