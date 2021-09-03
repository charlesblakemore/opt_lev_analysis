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

base_directory = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/series_5/change_phi_offset_0_to_0_3_dg/long_int/raw_curves/'

#base_directory = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/series_5/change_phi_offset_0_6_to_0_9_dg/long_int/raw_curves/'

#base_directory = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/series_5/change_phi_offset_0_3_to_0_6_dg_1/long_int/raw_curves/'

base_directory = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/series_5/long_int_0_to_0_9_dg/long_int/raw_curves/'

base_directory = '/home/dmartin/Desktop/analyzedData/20200601/bead2/spinning/libration_cooling/change_phi_offset/change_phi_offset_1/raw_curves_1/'

#base_directory = '/home/dmartin/Desktop/analyzedData/20200330/gbead3/spinning/ringdown/high_press/high_press_5/'

save_base_name = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_5/change_phi_offset_0_to_0_3_dg/long_int/fit_data/'

save_base_name = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_5/change_phi_offset_0_3to_0_6_dg/long_int/fit_data/'

save_base_name = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_5/change_phi_offset_0_6_to_0_9_dg/long_int/fit_data/'

save_base_name = base_directory.split('/long_int/')[0] + '/long_int'

#save_base_name = base_directory.split('spinning')[0] + '/spinning/series_5_using_new_psd/' + base_directory.split('/')[-4] + '/long_int'

print(save_base_name)

files, length, folders= bu.find_all_fnames(base_directory, add_folders=True)

#folders.pop(0)



#folders = ['/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/series_5/long_int_0_to_0_9_dg/long_int/raw_curves/long_int_base_press/']


#fit_many_in_sequency_psd arguments

dist_to_end_percentage = 0.001 
migrad_ncall = 10000

fix_A = False
fix_f0 = True
fix_g = False
fix_b = True
fix_h = False
fix_B = False

save_fit_p = False 

fit_avg_psd = False
fit_psd = True #Fit each PSD one at time
plot_fit = True
plot_fit_single = True
new_psd = False

param_arr = [fix_A, fix_f0, fix_g, fix_b]
param_arr = np.array(param_arr)

if new_psd:
    param_arr = [fix_A, fix_B, fix_f0, fix_g, fix_b, fix_h]
    param_arr = np.array(param_arr)


###################################


filt = False

drive = True

fit = False


fit_bandwidth = 100

libration_freq = 390

x_y_label_size = 18
tick_size = 18
legend_size = 16
alpha = 0.70

folders.sort(key=lambda f: int(filter(str.isdigit,f)))
#folders = folders[1:2]
print(folders)

colors = bu.get_color_map(3, 'copper')
colors = colors[::-1]


def psd_lorentzian(x, A, f0, g, b, h):
    w = 2*np.pi*x
    w0 = 2*np.pi*f0
    #denom =( (((1./g)+c)*w)**2 + (w0**2 - w**2)**2)
    #denom = (((g*w0**2+h)*w)**2 + (w0**2 - w**2)**2)
    denom = ((g*w)**2 + (w0**2 - w**2)**2)
    return (1/denom)*((A)) + b

def new_psd_lorentzian(x, A, B, f0, g, b, h):
    w = 2*np.pi*x
    w0 = 2*np.pi*f0
    #denom =( (((1./g)+c)*w)**2 + (w0**2 - w**2)**2)
    denom = ((((g*w0**2)+h)*w)**2 + (w0**2 - w**2)**2)
    return (1/denom)*((A*h) + ((w0**2-w**2)**2 + (h*w)**2)*B) + b 


def gaussian(x, mu, c, A):
    return np.exp(-(x-mu)**2) * A + c

A_avg_arr = []
B_avg_arr = []
f0_avg_arr = []

g_avg_arr = []
b_avg_arr = []
chi_sq_ndof_avg_arr = []

A_std_arr = []

f0_std_arr = []
g_std_arr = []
b_std_arr = []
chi_sq_ndof_std_arr = []


def fit_many_in_sequence_psd_avg(filenames, dtep, ind):
    """Takes many PSD of the instantaneous phase of the cross-polarized light and takes an average (psd_means). Then the average 
    PSD is windowed and fit by a Lorentzian as determined by the driven, damped harmonic oscillator equation for the libratonal 
    mode. The standard deviation of the mean for the peak structure is found and used as the error when fitting."""

    meas_name = filenames[0].split('/')[-2]

    A_arr = []
    f0_arr = []
    g_arr = []
    g_errs = []
    b_arr = []
    chi_sq_ndof = []
    g_eff_arr = []


    init_data = np.load(filenames[0], allow_pickle=True)

    init_psd = init_data['psd']

    dg = init_data['dg']

    N = 0 
    psd_sum = np.zeros_like(init_psd)

    
    def chi_sq(A, f0 , g, b, h):
        return np.sum(((y-psd_lorentzian(x, A, f0, g, b, h))/y_err)**2.)
    
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
        print(f)
        data = np.load(f)

        psd = data['psd']

        diff = (psd - psd_means)**2
        diff_sum += diff
        N += 1
    
    psd_stds = np.sqrt(diff_sum/N)
    y_err_unmask = psd_stds
    
    psd = data['psd']
    freqs = data['freqs']

    y = psd_means
    x = freqs 
    
    max_ind = np.argmax(y)

    
    mask_around_peak = (freqs > freqs[max_ind] - fit_bandwidth*0.5) & (freqs < freqs[max_ind] + 0.5*fit_bandwidth)#(freqs > libration_freq - fit_bandwidth*0.5) & (freqs < libration_freq + 0.5*fit_bandwidth) 
    end_mask = (freqs > freqs[-1]-(len(freqs)*dtep)) 
   
    #plt.loglog(freqs, y)
    #plt.scatter(freqs[max_ind], y[max_ind])
    #plt.show()

    #A_guess = 1e3
    #f0_guess = x[max_ind]
    #g_guess = 1
    #b_guess = np.mean(y[end_mask])

    A_guess = 40e1
    f0_guess = x[max_ind]
    g_guess = 1e-6
    b_guess = np.mean(y[end_mask])
    h_guess = 0.005

    #plt.loglog(x, y)
    #plt.loglog(x[end_mask], y[end_mask])
    #plt.show()

    y = y[mask_around_peak]
    x = x[mask_around_peak]
    y_err = y_err_unmask[mask_around_peak]/np.sqrt(N)

    #plt.semilogy(x,y)
    #plt.show()
    
    #limit_A = (200000, 500000)
    #m=Minuit(chi_sq, A=A_guess, fix_A=fix_A, f0=f0_guess, fix_f0=fix_f0, g=g_guess, fix_g=fix_g, b=b_guess, fix_b=fix_b, print_level=1, errordef=1)
    m=Minuit(chi_sq, A=A_guess, fix_A=fix_A, f0=f0_guess, fix_f0=fix_f0, g=g_guess, fix_g=fix_g,\
                    limit_g = (-10000, 10000), b=b_guess, fix_b=fix_b, fix_h=False, h=h_guess, limit_h = (-10000,10000), print_level=1, errordef=1)


    m.migrad(ncall=migrad_ncall)
    p = [m.values['A'], m.values['f0'], m.values['g'], m.values['b'], m.values['h']]
    
    #g_eff = np.abs(p[2])*(2*np.pi*p[1])**2 + np.abs(p[4])
    g_eff = np.abs(p[2])
    print(p)
    print(g_eff)
    
    chi_sq_red = m.fval/(len(y)-len(param_arr))

    A_arr.append(m.values['A'])
    f0_arr.append(m.values['f0'])
    g_arr.append(m.values['g'])
    g_errs.append(m.errors['g'])
    b_arr.append(m.values['b'])
    chi_sq_ndof.append(chi_sq_red)

    x_arr = np.linspace(x[0], x[-1], len(x)*10)

    print('chi_sq_fval', chi_sq_red, len(y)) 
    

    #m.hesse()
    if plot_fit:
        plt.loglog(x, y)
        plt.loglog(x_arr, psd_lorentzian(x_arr, *p), label='{} dg'.format(dg))
        plt.legend()
        plt.show()

    mask = (x > 325) & (x < 375)
    mask_1 = (x_arr > 325) & (x_arr < 375)
    if dg in [0.0, 0.3, 0.9]:
        print(ind)
        ax1.semilogy(x[mask], y[mask], label=r'$a_{dg}$ ' + '= {}'.format(dg), color=colors[ind], alpha=alpha)
        ax1.semilogy(x_arr[mask_1], psd_lorentzian(x_arr, *p)[mask_1], color=colors[ind])
        ax1.tick_params(axis='both', which='major', labelsize=tick_size)
        ax1.set_ylabel(r'$\phi$ PSD [$rad^{2}$/Hz]', size=x_y_label_size)
        ax1.set_xlabel('Frequency [Hz]', size=x_y_label_size)
        ax1.legend(fontsize=legend_size)
        fig1.tight_layout()
        #fig1.savefig('/home/dmartin/Desktop/analyzedData/20200130/images/paper_figs/psd_vary_dg.svg', format='svg', dpi=1200)
        #plt.close()        
        #plt.grid(b=True, which='minor', axis='both')
        #plt.grid(b=True ,which='major', axis='both')
        #fig1.tight_layout()
        ind = ind + 1 
    if save_fit_p:
        if fit_avg_psd:
            save_name = save_base_name + '/fit_data_avg_psd/' + meas_name + '_fit_params'
        print(save_name)
        bu.make_all_pardirs(save_name)

        np.savez(save_name, A_arr=A_arr, f0_arr=f0_arr, g_arr=g_arr, b_arr=b_arr, chi_sq_ndof=chi_sq_ndof, dg=dg, g_errs=g_errs, g_eff=g_eff)
    return ind
def fit_many_in_sequence_psd(filenames, dtep):
    """Takes many PSDs of the instantaneous phase of the cross-polarized light and fits them individually.The standard deviation 
    for the peak structure is found and used as the error when fitting. This fitting procedure makes a window arond the 
    librational peak structure and fits the structure with a Lorentzian as determined by the driven, damped harmonic oscillator 
    equation for the libratonal mode."""

    meas_name = filenames[0].split('/')[-2]

    A_arr = []
    B_arr = []
    f0_arr = []
    g_arr = []
    g_errs = []
    b_arr = []
    chi_sq_ndof = []

    init_data = np.load(filenames[0], allow_pickle=True)

    init_psd = init_data['psd']

    dg = init_data['dg']

    N = 0 
    psd_sum = np.zeros_like(init_psd)
    def chi_sq(A, f0 , g, b, h):
        return np.sum(((y-psd_lorentzian(x, A, f0, g, b, h))/y_err)**2.)
    
    if new_psd:
        def chi_sq(A, B, f0, g, b, h):
            return np.sum(((y-new_psd_lorentzian(x, A, B,f0, g, b, h))/y_err)**2.)
    
    #def chi_sq(mu, c, A):
    #    return np.sum(((y-gaussian(x, mu, c, A))/y_err)**2.)

    for i, f in enumerate(filenames):
        data = np.load(f, allow_pickle=True)
        
        psd = data['psd']
        
        print(len(psd))
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
    y_err_unmask = psd_stds
    #y_err_unmask = np.ones_like(psd_stds)
    
    for i, f in enumerate(filenames):
        data = np.load(f, allow_pickle=True)

        dg = data['dg']
        psd = data['psd']
        freqs = data['freqs']

        y = psd
        x = freqs 
        
        max_ind = np.argmax(y)

        #plt.semilogy(x,y)
        #plt.scatter(x[max_ind], y[max_ind])
        #plt.show()

        print(freqs[max_ind])
        mask_around_peak = (freqs > freqs[max_ind] - fit_bandwidth*0.5) & (freqs < freqs[max_ind] + 0.5*fit_bandwidth)
        end_mask = (freqs > freqs[-1]-(len(freqs)*dtep))

        
        A_guess = 160e3
        f0_guess = x[max_ind]
        #g_guess = 1
        g_guess = 1e-7
        b_guess = np.mean(y[end_mask])
        h_guess = 0.05
        
        #plt.loglog(x, y)
        #plt.loglog(x[end_mask], y[end_mask])
        #plt.show()

        y = y[mask_around_peak]
        x = x[mask_around_peak]
        y_err = y_err_unmask[mask_around_peak]
        #y_err = y_err_unmask

        #plt.semilogy(x,y)
        #plt.show()
        if not new_psd:
            m=Minuit(chi_sq, A=A_guess, fix_A=fix_A, limit_A= (0, 1e9), f0=f0_guess, fix_f0=fix_f0, g=g_guess, fix_g=fix_g,\
                    limit_g = (-10e3, 10e3), b=b_guess, fix_b=fix_b, fix_h=False, h=h_guess, limit_h = (-10e3,10e3), print_level=1, errordef=1)
        
        elif new_psd:
            h_guess = .01
            B_guess = 0.001**2

            m=Minuit(chi_sq, limit_A = (10e3, 100e3), A=A_guess, fix_A=fix_A, f0=f0_guess, fix_f0=fix_f0, g=g_guess, fix_g=fix_g, \
                    b=b_guess, fix_b=fix_b,h=h_guess, fix_h=fix_h, fix_B=fix_B, B=B_guess, print_level=1, errordef=1)
            
        m.migrad(ncall=migrad_ncall)
        p = [m.values['A'], m.values['f0'], m.values['g'], m.values['b'], m.values['h']]
        label = r'A = {}, f_0 = {}, g = {}, b = {}, h = {}'.format(round(p[0],2), round(p[1],2), round(p[2],2), round(p[3],2), round(p[4],2))
         
        print_state = r'A = {}, f_0 = {}, g = {}, b = {}, h = {}'.format(p[0],p[1],p[2],p[3],p[4])
    
        print(print_state)

    
        g_eff = m.values['g']*(2*np.pi*m.values['f0'])**2 + m.values['h']
        g_eff = m.values['g']
        #print(p, 'gamma_eff = {}'.format(np.abs(m.values['g'])*(2*np.pi*m.values['f0'])**2 + m.values['h']))
        print(p, 'gamma_eff = {}'.format(g_eff))
        if new_psd:
            p = [m.values['A'], m.values['B'], m.values['f0'], m.values['g'], m.values['b'], m.values['h']]
            label = r'A = {}, B = {}, f_0 = {}, g = {}, b = {}, h = {}'.format(round(p[0],2), round(p[1],2), round(p[2],2), \
                    round(p[3],2), round(p[4],2), round(p[5],2))
            g_eff = m.values['g']*(2*np.pi*m.values['f0'])**2 + m.values['h']
            
            B_arr.append(m.values['B'])
        
            print(p, 'gamma_eff = {}'.format(np.abs(m.values['g'])*(2*np.pi*m.values['f0'])**2 + m.values['h']))

        

        chi_sq_red = m.fval/(len(y)-len(param_arr))

        A_arr.append(m.values['A'])
        f0_arr.append(m.values['f0'])
        g_arr.append(g_eff)
        g_errs.append(m.errors['g'])
        b_arr.append(m.values['b'])
        chi_sq_ndof.append(chi_sq_red)

        x_arr = np.linspace(x[0], x[-1], len(x)*10)


        print('chi_sq', chi_sq_red, len(y))
       
        if plot_fit:
            fig, ax = plt.subplots(2, 1, sharex=True)
            if not new_psd:
                ax[0].semilogy(x, y)
                ax[0].semilogy(x_arr, psd_lorentzian(x_arr, *p), label=label)
                ax[1].plot(x, y_err)
            elif new_psd:
                ax[0].semilogy(x, y)
                ax[0].semilogy(x_arr, new_psd_lorentzian(x_arr, *p), label=label)
                ax[1].plot(x, y_err)
            #plt.loglog(x_arr, gaussian(x_arr, * p))
            ax[0].legend()
            plt.show()
        
        low_freq = 340
        high_freq = 360

        mask = (x > low_freq) & (x < high_freq )
        x_arr_mask = (x_arr > low_freq) & (x_arr < high_freq)

        x_y_label_size = 22
        tick_size = 22
        legend_size = 20

        if plot_fit_single and dg==0.3:
            fig1,ax1 = plt.subplots()
            ax1.semilogy(x[mask],y[mask], color=colors[1], alpha=0.75)
            ax1.semilogy(x_arr[x_arr_mask], psd_lorentzian(x_arr,*p)[x_arr_mask], label= "$a_{dg}$ = " + "{}".format(dg), color=colors[1])
            
            ax1.tick_params(axis='both', which='major', labelsize=tick_size)
            ax1.set_ylabel(r'$\phi$ PSD [$rad^{2}$/Hz]', size=x_y_label_size)
            ax1.set_xlabel('Frequency [Hz]', size=x_y_label_size)
            ax1.legend(fontsize=legend_size)
            fig1.tight_layout()
            plt.show()        
    if save_fit_p:
        save_name = save_base_name + '/fit_data/' + meas_name + '_fit_params'
        print(save_name)
        bu.make_all_pardirs(save_name)

        np.savez(save_name, A_arr=A_arr, f0_arr=f0_arr, g_arr=g_arr, b_arr=b_arr, chi_sq_ndof=chi_sq_ndof, dg=dg, g_errs=g_errs)
    if new_psd:
        B_avg_arr.append(np.mean(B_arr))


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

if fit_avg_psd:
    fig1, ax1 = plt.subplots()
    ind = 0
    for i, folder in enumerate(folders):
        files, zero = bu.find_all_fnames(folder, ext='.npz')

        ind = fit_many_in_sequence_psd_avg(files, dist_to_end_percentage, ind)
    plt.show()
if fit_psd:
    for i, folder in enumerate(folders[:]):
        files, zero = bu.find_all_fnames(folder, ext='.npz')
    
        fit_many_in_sequence_psd(files, dist_to_end_percentage)
    
    file_arr = np.arange(len(folders))
   
    #plt.scatter(file_arr, np.abs(A_avg_arr))
    #plt.show()

    plt.scatter(file_arr, g_avg_arr)
    #plt.yscale('log')
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
