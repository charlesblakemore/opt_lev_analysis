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

directory = '/data/old_trap/20200130/bead1/spinning/transient/change_dc_offset_5_003mbar/'
directory = '/data/old_trap/20200130/bead1/spinning/series_1/base_press/change_phi_dc_offset_1/'
directory = '/data/old_trap/20200130/bead1/spinning/test/using_hp/'
directory = '/data/old_trap/20191223/bead1/spinning/tests/noise_inject/'
directory = '/data/old_trap/20200130/bead1/spinning/test/tabor_pulse/'
directory = '/data/old_trap/20200130/bead1/spinning/series_2/base_press/change_phi_offset_0_85_dg/'
directory = '/data/old_trap/20200130/bead1/spinning/series_3/base_press/change_phi_offset/0_dg_noise_added_0_009/'

folders = ['/data/old_trap/20200130/bead1/spinning/series_3/base_press/change_phi_offset/0_dg/',\
           '/data/old_trap/20200130/bead1/spinning/series_3/base_press/change_phi_offset/0_8_dg/',\
           '/data/old_trap/20200130/bead1/spinning/series_3/base_press/change_phi_offset/0_85_dg/',\
           '/data/old_trap/20200130/bead1/spinning/series_3/base_press/change_phi_offset/0_9_dg/']

base_folder = '/data/old_trap/20200130/bead1/spinning/series_4/base_press/change_phi_offset_3/change_phi_offset/'

files, zeros, folders = bu.find_all_fnames(base_folder, ext='.h5', sort_time=True, \
                    add_folders=True)

save = True 
save_transients = True
multiple_folders = True

#gaussian check params:
save_histograms = True
num_save_hist = 10
num_y_data_points = 500000
num_bins = 10
max_length_from_end = 150000
num_curves = 10
hist_end_time = 3
save_base_name_hist = '/home/dmartin/Desktop/analyzedData/20200130/images/hist/'

#fit transients params:
dist_to_end = 10
a_fix = True
b_fix = True
c_fix = False
d_fix = True
start_c = -2
migrad_ncall = 100

fit_curves = False
plot = False
plot_multiple = False
plot_drive_and_inst_phase = False
save_base_name_fit_trans = '/home/dmartin/Desktop/analyzedData/20200130/spinning/base_press/series_4/change_phi_offset/'
save_base_name_save_trans = '/home/dmartin/Desktop/analyzedData/20200130/spinning/base_press/series_4/change_phi_offset_3/change_phi_offset/raw_curves/' 

if save:
    bu.make_all_pardirs(save_base_name_fit_trans)
    bu.make_all_pardirs(save_base_name_hist)
    bu.make_all_pardirs(save_base_name_save_trans)

crossp_ind = 0
drive_ind = 1

wind_bandwidth = 100
libration_freq = 370

threshold = 0.6

def exp(x, a, b , c, d):
    return a*np.exp((x-b)*c) + d 

def extract_y_err(filenames, lf, bw, thr, dte, max_lfe):

    print('extract_y_err')
    y_curves = []
    tarrs = []
    start_inds = []
    dg_arr = []
    for i, f in enumerate(filenames):
        print(f, i)
        obj = hsDat(f)
        
        dg = obj.attribs['current_pm_dg']
        
        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        freqs = np.fft.rfftfreq(Ns, 1./Fs)
    
        tarr = np.arange(Ns)/Fs
        

        crossp = obj.dat[:,crossp_ind]
        #drive = obj.dat[:, drive_ind]

        crossp_fft = np.fft.rfft(crossp)

        crossp_z = signal.hilbert(crossp)
        #drive_z  = signal.hilbert(drive)

        crossp_phase = signal.detrend(np.unwrap(np.angle(crossp_z)))
        #drive_phase = signal.detrend(np.unwrap(np.angle(drive_z)))

        crossp_phase_unfilt = np.fft.rfft(crossp_phase)

        crossp_phase = bp_filt(crossp_phase, lf, Ns, Fs, bw)

        crossp_phase_z = signal.hilbert(crossp_phase)
        crossp_phase_amp = np.abs(crossp_phase_z)

        max_val = np.amax(crossp_phase_amp)
        max_ind = np.argmax(crossp_phase_amp)
        print(max_val)

        start_ind = max_ind
        
        length_of_arrays = len(tarr) - max_lfe #length of all arrays after cutting out data points greater than this value

        mask = (tarr > tarr[start_ind])

        
        if len(crossp_phase_amp[mask]) < length_of_arrays: #Check if array is too short for proper binning
            print('array is too short')

        else:
            crossp_phase_amp = crossp_phase_amp[mask][:length_of_arrays]
            tarr = tarr[mask][:length_of_arrays]
            print('cut')

        #plt.plot(tarr, crossp_phase_amp)
        #plt.show()

        x = tarr
        y = crossp_phase_amp
        start_inds.append(start_ind)
        y_curves.append(y)
        tarrs.append(tarr)
            
        print(len(y_curves))
    for i, curve in enumerate(y_curves):
        plt.plot(tarrs[i], curve)
    plt.show()
    y_stds = np.std(y_curves, axis=0)

    return y_stds, start_inds, y_curves, tarrs,  Ns , Fs, dg

def extract_y_err_save(filenames, lf, bw, thr, dte, max_lfe, fs):

    print('extract_y_err_save')
    y_curves = []
    tarrs = []
    start_inds = []
    dg_arr = []
    save_arr = []
    for i, f in enumerate(filenames):
        print(f, i)
        
        raw_file_name = f.split('/')[-1]
        meas_name = f.split('/')[-2]
        
        obj = hsDat(f)
        
        dg = obj.attribs['current_pm_dg']
        
        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        freqs = np.fft.rfftfreq(Ns, 1./Fs)
    
        tarr = np.arange(Ns)/Fs
        

        crossp = obj.dat[:,crossp_ind]
        #drive = obj.dat[:, drive_ind]

        crossp_fft = np.fft.rfft(crossp)

        crossp_z = signal.hilbert(crossp)
        #drive_z  = signal.hilbert(drive)

        crossp_phase = signal.detrend(np.unwrap(np.angle(crossp_z)))
        #drive_phase = signal.detrend(np.unwrap(np.angle(drive_z)))

        crossp_phase_unfilt = np.fft.rfft(crossp_phase)

        crossp_phase = bp_filt(crossp_phase, lf, Ns, Fs, bw)

        crossp_phase_z = signal.hilbert(crossp_phase)
        crossp_phase_amp = np.abs(crossp_phase_z)

        max_val = np.amax(crossp_phase_amp)
        max_ind = np.argmax(crossp_phase_amp)
        print(max_val)

        start_ind = max_ind
        
        length_of_arrays = len(tarr) - max_lfe #length of all arrays after cutting out data points greater than this value

        mask = (tarr > tarr[start_ind])

        
        if len(crossp_phase_amp[mask]) < length_of_arrays: #Check if array is too short for proper binning
            print('array is too short')

        else:
            crossp_phase_amp = crossp_phase_amp[mask][:length_of_arrays]
            tarr = tarr[mask][:length_of_arrays]
            print('cut')

        save_base_name= save_base_name_save_trans + '{}/'.format(fs) 
        save_transients_name = save_base_name + '{}'.format(raw_file_name.split('.')[0])
        print(save_transients_name)
    
        bu.make_all_pardirs(save_base_name)

        if save_transients:
            np.savez(save_transients_name, tarr=tarr, crossp_phase_amp=crossp_phase_amp, dg=dg, Ns=Ns, Fs=Fs)


def fit_transients(filenames, lf, bw, thr, dte, max_lfe, fix_a, fix_b, fix_c, fix_d, start_c, migrad_ncall):

    phi_inst_amps_x = []
    phi_inst_amps_y = []
    phi_inst_amps_stds = []
    fit_params = []
    chi_sq_arr = []

    def chi_sq(a, b ,c, d):
        return np.sum(((y-exp(x, a, b, c, d))/y_err)**2.)

    y_stds, start_inds, y_curves, tarrs, Ns, Fs, dg = extract_y_err(filenames, lf, bw, thr, dte, max_lfe)

    print('fit_transients')
    
    for i, curve in enumerate(y_curves):
        print('iteration {}'.format(i))
        x = tarrs[i]
        y = curve
        y_err = y_stds 

        #end_mask = (tarr > tarr[-1]-dte)
        end_mask = (x > x[-1]-dte)
        a_guess = y[0]
        b_guess = x[0]
        c_guess = start_c
        d_guess = np.mean(y[end_mask])

        #x = tarr[start_inds[i]:]
        #y = curve[start_inds[i]:]
        #y_err = y_stds[start_inds[i]:]

        m=Minuit(chi_sq, a=a_guess, fix_a=fix_a, error_a=0.1, b=b_guess, fix_b=fix_b, \
                c=c_guess, fix_c=fix_c, limit_c=[-100000,0],  error_c=0.1, d=d_guess,fix_d=fix_d, print_level=1) 
         
        m.migrad(ncall=migrad_ncall)

        x_arr = np.linspace(x[:][0], x[:][-1], len(x[start_inds[i]:])*10)

        p = [m.values['a'], m.values['b'], m.values['c'], m.values['d']]
        
        chi_sq_ndof = chi_sq(*p)/(len(y)-1)
        print('chi squared ', chi_sq_ndof)
                
        #fit_label = r'a$e^{c(t-b)}$, ' + 'a={} rad, b={} s, c={} Hz'.format(popt[0].round(2), popt[1].round(2), popt[2].round(2))
        
        fit_params.append(p)
        chi_sq_arr.append(chi_sq_ndof)
        phi_inst_amps_x.append(x)
        phi_inst_amps_y.append(y)
        phi_inst_amps_stds.append(y_err)

    return fit_params, phi_inst_amps_x, phi_inst_amps_y, phi_inst_amps_stds, dg, chi_sq_arr

def gaussian_check(filenames, lf, bw, thr, pl, max_lfe, nbins, n, nsave, save_hists, ncurves, end_time):
    
    print('gaussian_check')

    y_arr = [0 for i in range(len(filenames))]

    for i, f in enumerate(filenames):
        print(f, i)
        obj = hsDat(f)

        dg = obj.attribs['current_pm_dg']

        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        tarr = np.arange(Ns)/Fs

        crossp = obj.dat[:,crossp_ind]
        
        crossp_z = signal.hilbert(crossp)

        crossp_phase = signal.detrend(np.unwrap(np.angle(crossp_z)))

        crossp_phase = bp_filt(crossp_phase, lf, Ns, Fs, bw)
        
        crossp_phase_z = signal.hilbert(crossp_phase)
        crossp_phase_amp = np.abs(crossp_phase_z)
        
        max_val = np.amax(crossp_phase_amp)
        max_ind = np.argmax(crossp_phase_amp)

        start_ind = max_ind

        
        mask = (tarr > tarr[start_ind])
        
        length_of_arrays = len(tarr) - max_lfe #length of all arrays after cutting out data points greater than this value
        
        if len(crossp_phase_amp[mask]) < length_of_arrays: #Check if array is too short for proper binning
            print('array is too short')
            
        else:
            crossp_phase_amp = crossp_phase_amp[mask][:length_of_arrays] 
            tarr = tarr[mask][:length_of_arrays]
            print('cut')
        
        #plt.plot(crossp_phase_amp)
        
        print(len(crossp_phase_amp))
        
        y_arr[i] = crossp_phase_amp
   
    tarr_cut = tarr < end_time

    tarr_length = len(tarr[tarr_cut])

    y_arr = np.array(y_arr)
    #data_point_inds = np.arange(0,y_arr.shape[1], y_arr.shape[1]/nsave) 
    data_point_inds = np.arange(0, tarr_length, tarr_length/nsave)
    print(len(data_point_inds))
   
    hist_arr = []
    bins_arr = []

    for j, col in enumerate(data_point_inds):
        if col < y_arr.shape[1]:
            hist, bins = np.histogram(y_arr[:,col], bins=nbins)
            hist_arr.append(hist)
            bins_arr.append(bins)
        else:
            print('break')
            break
    
    hist_arr = np.array(hist_arr)
    bins_arr = np.array(bins_arr)
   
    y_plot = np.zeros_like(data_point_inds)
    
    if pl:
        for ind, curve in enumerate(y_arr):
            if ind < ncurves:
                plt.plot(curve)
            else:
                break
        plt.scatter(data_point_inds, y_arr[0][data_point_inds], s=50, c='g')
        plt.show()

    for i in range(nsave):
        meas_name = f.split('/')[-2]     
        
        save_name = save_base_name_hist +  meas_name + '_{}_dg_{}'.format(dg, i) + '.png'
        
        hist = hist_arr[i]
        bins = bins_arr[i][:-1]
        width = np.abs(bins_arr[i][1]-bins_arr[i][0])
        
        if save_hists:
            plt.bar(bins, hist, width=width)
            plt.ylabel('Counts')
            plt.xlabel(r'y_{}'.format(i))
            plt.savefig(save_name)
            plt.close()
            gc.collect()
            #add way to save the histograms to be fitted by gaussians later
    

def plot_multiple(files, lf, bw):
    for i, f in enumerate(files):

        print(f, i)
        obj = hsDat(f)

        dg = obj.attribs['start_pm_dg']

        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        freqs = np.fft.rfftfreq(Ns, 1./Fs)

        tarr = np.arange(Ns)/Fs
        
        crossp = obj.dat[:,0]

        crossp_z = signal.hilbert(crossp)


        crossp_phase = signal.detrend(np.unwrap(np.angle(crossp_z)))

        crossp_phase = bp_filt(crossp_phase, lf, Ns, Fs, bw)

        crossp_phase_z = signal.hilbert(crossp_phase)
        crossp_phase_amp = np.abs(crossp_phase_z)

        plt.plot(tarr, crossp_phase, label='dg={}'.format(dg))
        
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel(r'Filtered $\phi(t)$')
    plt.show()

def plot_drive_w_phase(f, lf, bw):

    print(f)
    obj = hsDat(f)

    dg = obj.attribs['start_pm_dg']

    Ns = obj.attribs['nsamp']
    Fs = obj.attribs['fsamp']
    freqs = np.fft.rfftfreq(Ns, 1./Fs)

    tarr = np.arange(Ns)/Fs

    crossp = obj.dat[:,0]
    drive = obj.dat[:,1]
    crossp_z = signal.hilbert(crossp)
    drive_z = signal.hilbert(drive)


    drive_phase = signal.detrend(np.unwrap(np.angle(drive_z)))
    crossp_phase = signal.detrend(np.unwrap(np.angle(crossp_z)))

    #crossp_phase = bp_filt(crossp_phase, lf, Ns, Fs, bw)

    crossp_phase_z = signal.hilbert(crossp_phase)
    crossp_phase_amp = np.abs(crossp_phase_z)

    plt.plot(tarr,crossp_phase, label=r'$P_{\perp}$')
    plt.plot(tarr,drive_phase, label='Drive')
    plt.ylabel('Instaneous Phase [rad]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()


if multiple_folders:
    for i, folder in enumerate(folders[1:]):
        files, zero = bu.find_all_fnames(folder, sort_time=True)
        files = files[1:]
        
        #gaussian_check(files[:], libration_freq, wind_bandwidth, threshold, plot, max_length_from_end, num_bins, \
        #        num_y_data_points, num_save_hist, save_histograms, num_curves, hist_end_time)
        
        print(folder)
        folder_save = folder.split('/')[-1]
        extract_y_err_save(files, libration_freq, wind_bandwidth, threshold, dist_to_end, max_length_from_end, folder_save)

        #fit_params, phi_inst_amps_x, phi_inst_amps_y, phi_inst_amps_stds, dg, chi_sq_arr = fit_transients(files[:], \
        #        libration_freq, wind_bandwidth, threshold, dist_to_end, max_length_from_end, a_fix, b_fix, c_fix, \
        #        d_fix, start_c, migrad_ncall)
        
        #if save:
        #    meas_name = files[-1].split('/')[-2]
        #    
        #    if meas_name == '':
        #        print('empty name')
        #        meas_name = '{}'.format(i)
        #    
        #    save_name = save_base_name_fit_trans + meas_name +  '_{}'.format(dg)
        #    
        #    np.savez_compressed(save_name, fit_params=fit_params, phi_inst_amps_x=phi_inst_amps_x,\
        #            phi_inst_amps_y=phi_inst_amps_y, phi_inst_amps_stds=phi_inst_amps_stds, dg=dg, chi_sq_arr=chi_sq_arr)
            
else:
    fit_params, fit_curves, phi_inst_amps = plot_transient(files[1:], libration_freq, wind_bandwidth, threshold)

    if save:
            meas_name = files[i].split('/')[-2]


            if meas_name == '':
                print('empty name')
                meas_name = '{}'.format(i)

            save_name = save_base_name + meas_name
            np.savez(save_name, fit_params=fit_params, fit_curves=fit_curves, phi_inst_amps=phi_inst_amps)

#if plot_drive_and_inst_phase:
#    files, zero = bu.find_all_fnames(folders[1], sort_time=True)
#    
#    for i,f in enumerate(files):
#        plot_drive_w_phase(f, libration_freq, wind_bandwidth)
#
#if plot_multiple:
#    file_arr = []
#    for i, folder in enumerate(folders):
#        files, zero = bu.find_all_fnames(folder, sort_time=True)
#
#        file_arr.append(files[-1])
#        
#    plot_multiple(file_arr, libration_freq, wind_bandwidth)











