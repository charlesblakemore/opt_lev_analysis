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
from iminuit import Minuit

import gc

mpl.rcParams['figure.figsize'] = [7,5]
mpl.rcParams['figure.dpi'] = 150


base_folder = '/home/dmartin/Desktop/analyzedData/20200130/spinning/base_press/series_4/change_phi_offset_3/change_phi_offset/raw_curves/'
base_folder = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/series_5/change_phi_offset_0_6_to_0_9_dg/change_phi_offset/raw_curves/'
#base_folder = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/series_5/change_phi_offset_0_3_to_0_6_dg_1/change_phi_offset/raw_curves/'
base_folder = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/series_5/change_phi_offset_6_to_9_dg/raw_curves/'
#base_folder = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/test/change_phi_offset_30_dg/change_phi_offset/raw_curves/'
base_folder = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/test/change_phi_offset_30_dg/change_phi_offset/raw_curves/'
base_folder = '/home/dmartin/Desktop/analyzedData/20200130/bead1/spinning/test/change_phi_offset_30_dg/change_phi_offset/raw_curves_env_and_osc/'

files, zeros, folders = bu.find_all_fnames(base_folder, sort_time=True, add_folders=True)

print(folders)

save = False 
save_transients = False
multiple_folders = True

#gaussian check params:
save_histograms = True
inum_save_hist = 10
num_y_data_points = 500000
num_bins = 10
max_length_from_end = 50000
num_curves = 10
hist_end_time = 3
save_base_name_hist = '/home/dmartin/Desktop/analyzedData/20200130/images/hist/'

#fit transients params:
#dist_to_end = 10 # for long 33s int
dist_to_end = 0.1
a_fix = True
b_fix = True
c_fix = False
d_fix = True
start_c = -2
migrad_ncall = 100

end_time = 1

just_env = False
env_and_osc = True

fit_curves = False
plot = False
plot_multiple = False
plot_drive_and_inst_phase = False
save_base_name_fit_trans = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_5/change_phi_offset_0_to_0_3_dg/change_phi_offset/fit_data/'
save_base_name_fit_trans = base_folder.split('raw_curves')[0] + 'fit_data/'


if save:
    bu.make_all_pardirs(save_base_name_fit_trans)
    bu.make_all_pardirs(save_base_name_hist)

def exp(x, a, b , c, d):
    return a*np.exp((x-b)*c) + d
def exp_sine(x, a, b, c, d, f0, p):
    return a*np.exp((x-b)*c)*np.sin(2*np.pi*f0*x + p) + d

def fit_transients(filenames, dte, max_lfe, start_c, migrad_ncall):

    init_data = np.load(filenames[0])
    y = init_data['crossp_phase_amp']
    dg = init_data['dg']

    fit_params = []
    chi_sq_arr = []
    y_sum = np.zeros_like(y)
    y_mean = np.zeros_like(y)
    N = 0

    if just_env:
        print('just_env')
        def chi_sq(a, b ,c, d):
            return np.sum(((y-exp(x, a, b, c, d))/y_err)**2.)
    
    if env_and_osc:
        print('env_and_osc')
        def chi_sq(a, b, c, d, f0, p):
            return np.sum(((y-exp_sine(x, a, b, c, d, f0, p))/y_err)**2.)

    num_bad_files = 0
    bad_file_inds = []
    #Compute sum of y points for calculating mean later, 
    #opening one file at a time
    for i, f in enumerate(filenames):
        data = np.load(f)

        if len(data['crossp_phase_amp']) != len(y_sum):
            print('crossp_phase_amp not same len as y_sum. Probably because array did not get cut properly in process_trans_inst_phase.py')
            num_bad_files += 1
            bad_file_inds.append(i)
            continue



        y = data['crossp_phase_amp']
       
        y_sum += y
        N += 1

    y_means = y_sum/N
    
    N=0
    diff_sum = np.zeros_like(y)
    #compute std and fit curves
    
    for i, f in enumerate(filenames):
        if i in bad_file_inds:
        
            continue
        
        data = np.load(f)
        
        y = data['crossp_phase_amp']

        diff = (y - y_means)**2
        diff_sum += diff
        N += 1

    y_stds = np.sqrt(diff_sum/N)
    y_err_unmask = y_stds

    for i, f in enumerate(filenames):
        data = np.load(f)

        y = data['crossp_phase_amp']
        x = data['tarr']
        
        mask_exp = (x < end_time)

        end_mask = (x > x[-1]-dte)

        print(x[-1]-dte)
        plt.plot(x,y)
        plt.plot(x[end_mask], y[end_mask])
        plt.show()

        if just_env:
            a_guess = y[0]
            b_guess = x[0]
            c_guess = start_c
            d_guess = np.mean(y[end_mask])

            print('d_guess', d_guess)
                    
            y=y[mask_exp]
            x=x[mask_exp]
            y_err = y_err_unmask[mask_exp]


            m=Minuit(chi_sq, a=a_guess, fix_a=a_fix, error_a=0.1, b=b_guess, fix_b=b_fix, \
                    c=c_guess, fix_c=c_fix, limit_c=[-100000,0],  error_c=0.1, d=d_guess,fix_d=d_fix, print_level=1) 
             
            m.migrad(ncall=migrad_ncall)

            x_arr = np.linspace(x[:][0], x[:][-1], len(x)*10)

            p = [m.values['a'], m.values['b'], m.values['c'], m.values['d']]
            

            plt.semilogy(x_arr , exp(x_arr, *p))
            plt.semilogy(x, y)
            plt.ylabel(r'log(Envelope of $\phi (t)$)')
            plt.xlabel('Time [s]')
            plt.show()
            gc.collect()
        
        if env_and_osc:
            a_guess = y[0]
            b_guess = x[0]
            c_guess = start_c
            d_guess = np.mean(y[end_mask])
            f0_guess = 350
            p_guess = np.pi*0.5

            print('d_guess', d_guess)
                    
            y=y[mask_exp]
            x=x[mask_exp]
            y_err = y_err_unmask[mask_exp]


            m=Minuit(chi_sq, a=a_guess, fix_a=a_fix, error_a=0.1, b=b_guess, fix_b=b_fix, \
                    c=c_guess, fix_c=c_fix, limit_c=[-100000,0],  error_c=0.1, d=d_guess,fix_d=d_fix, f0=f0_guess, fix_f0=False,\
                    error_f0=0.1, p=p_guess, fix_p=False, error_p=0.1, print_level=1) 
             
            m.migrad(ncall=migrad_ncall)

            x_arr = np.linspace(x[:][0], x[:][-1], len(x)*10)

            p = [m.values['a'], m.values['b'], m.values['c'], m.values['d'], m.values['f0'], m.values['p']]
            

            plt.plot(x,y)
            plt.plot(x_arr , exp_sine(x_arr, *p))
            plt.xlabel('Time [s]')
            plt.show()
            gc.collect()
             

        print(p)
        chi_sq_ndof = chi_sq(*p)/(len(y)-1)
        print('chi squared ', chi_sq_ndof)

        #fit_label = r'a$e^{c(t-b)}$, ' + 'a={} rad, b={} s, c={} Hz'.format(popt[0].round(2), popt[1].round(2), popt[2].round(2))
        
        fit_params.append(p)
        chi_sq_arr.append(chi_sq_ndof)
    #raw_input('continue?')
    return fit_params, dg, chi_sq_arr, num_bad_files


for i, folder in enumerate(folders[1:]):
    print(folder, i)
    files, zero = bu.find_all_fnames(folder, ext='.npz')

    fit_params, dg, chi_sq_arr, num_bad_files = fit_transients(files, dist_to_end, max_length_from_end, start_c, migrad_ncall) 
     
    meas_name = folder.split('/')[-1]

    if save:
        print('saving')
        save_name = save_base_name_fit_trans + meas_name + '_exp_fit_params' 
        print('save name ' + save_name)
        np.savez(save_name, fit_params=fit_params, dg=dg, chi_sq_arr=chi_sq_arr, num_bad_files=num_bad_files)
