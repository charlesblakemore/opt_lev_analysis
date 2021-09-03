from hs_digitizer import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import window_func as window
import bead_util as bu
import bead_util_funcs_1 as buf
from amp_ramp_3 import bp_filt, lp_filt, hp_filt
from scipy import signal
from transfer_func_util import damped_osc_amp
from scipy.optimize import curve_fit
from memory_profiler import profile
from memory_profiler import memory_usage
from iminuit import Minuit
from joblib import Parallel, delayed
from plot_transient_inst_phase_v2 import exp_sine 
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

base_folder = '/data/old_trap/20200130/bead1/spinning/series_5/change_phi_offset_0_to_0_3_dg/change_phi_offset/'

base_folder = '/data/old_trap/20200130/bead1/spinning/series_5/change_phi_offset_0_3_to_0_6_dg_1/change_phi_offset/'

#base_folder = '/data/old_trap/20200130/bead1/spinning/series_5/change_phi_offset_0_6_to_0_9_dg/change_phi_offset/'

#base_folder = '/data/old_trap/20200130/bead1/spinning/series_2/base_press/change_phi_offset_0_dg/'

base_folder = '/data/old_trap/20200601/bead2/spinning/libration_cooling/change_phi_offset/change_phi_offset_1/change_phi_0010/'
base_folder ='/data/old_trap/20200601/bead2/spinning/libration_cooling/long_int/tests/' 

#base_folder = '/data/old_trap/20200130/bead1/spinning/series_5/change_phi_offset_3_to_6_dg/'

#base_folder = '/data/old_trap/20200130/bead1/spinning/series_5/change_phi_offset_6_to_9_dg/'

#base_folder = '/data/old_trap/20200130/bead1/spinning/test/change_phi_offset_50_dg/change_phi_offset/'


n_jobs = 1

save_inst_phase_env = True
save_inst_phase_env_and_osc = False
save_inst_phase_rebin = False


save = False
save_transients = False
multiple_folders = False

if multiple_folders:
    files, zeros, folders = bu.find_all_fnames(base_folder, ext='.h5', sort_time=True, \
                    add_folders=True)
else:
    files, zeros = bu.find_all_fnames(base_folder, ext='.h5', sort_time=True)


plot_phase_env = False
plot_phase_fft = False
plot = False
plot_multiple = False
plot_drive_and_inst_phase = False

save_base_name_fit_trans = '/home/dmartin/Desktop/analyzedData/20200130/spinning/base_press/series_4/change_phi_offset/'
#save_base_name_save_trans = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_5/change_phi_offset_0_6_to_0_9_dg/change_phi_offset/raw_curves/'
save_base_name_save_trans = '/home/dmartin/Desktop/analyzedData' + base_folder.split('old_trap')[1]

if save:
    bu.make_all_pardirs(save_base_name_fit_trans)
    bu.make_all_pardirs(save_base_name_hist)
    bu.make_all_pardirs(save_base_name_save_trans)

crossp_ind = 0
drive_ind = 1

max_length_from_end=90000
downs_num = 20
wind_bandwidth = 65
#wind_bandwidth = 250
libration_freq = 357


def exp(x, a, b , c, d):
    return a*np.exp((x-b)*c) + d 

def extract_y_err(filenames, lf, bw, max_lfe):

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

def extract_y_err_save(filenames, lf, bw, max_lfe, fs):

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

        fft = np.fft.rfft(crossp_phase)
       
        #plt.loglog(freqs, np.abs(fft))
        #plt.show()

        crossp_phase_z = signal.hilbert(crossp_phase)
        crossp_phase_amp = np.abs(crossp_phase_z)

        max_val = np.amax(crossp_phase_amp)
        max_ind = np.argmax(crossp_phase_amp)
        print(max_val)

        start_ind = max_ind
        
        length_of_arrays = len(tarr) - max_lfe #length of all arrays after cutting out data points greater than this value

        mask = (tarr > tarr[start_ind])
        
        
        if len(crossp_phase_amp[mask]) < length_of_arrays: #Check if array is too short for proper binning
            print('array is too short', len(crossp_phase_amp[mask]))
            plt.plot(crossp_phase_amp)
            plt.show()

        else:
            crossp_phase_amp = crossp_phase_amp[mask][:length_of_arrays]
            tarr = tarr[mask][:length_of_arrays]
            print('cut')

        #plt.semilogy(tarr,crossp_phase_amp)
        #plt.show()
        save_base_name= save_base_name_save_trans + 'raw_curves/' + '{}/'.format(fs) 
        save_transients_name = save_base_name + '{}'.format(raw_file_name.split('.')[0])
        print(save_transients_name)
    
        bu.make_all_pardirs(save_base_name)

        if save_transients:
            np.savez(save_transients_name, tarr=tarr, crossp_phase_amp=crossp_phase_amp, dg=dg, Ns=Ns, Fs=Fs)

def extract_y_err_save_parallel(f, lf, bw, max_lfe, fs):

    print('extract_y_err_save')
    y_curves = []
    tarrs = []
    start_inds = []
    dg_arr = []
    save_arr = []
        
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
    
    crossp = bp_filt(crossp, 50e3, Ns, Fs, 5000)

    crossp_fft = np.fft.rfft(crossp)
    
    plt.loglog(freqs, np.abs(crossp_fft))
    plt.show()

    crossp_z = signal.hilbert(crossp)
    #drive_z  = signal.hilbert(drive)

    crossp_phase = signal.detrend(np.unwrap(np.angle(crossp_z)))
    #drive_phase = signal.detrend(np.unwrap(np.angle(drive_z)))

    crossp_phase_unfilt = np.fft.rfft(crossp_phase)


    crossp_phase_z = signal.hilbert(crossp_phase)
    crossp_phase_amp = np.abs(crossp_phase_z)


    
    
    plt.plot(crossp_phase)
    plt.plot(crossp_phase_amp)
    plt.show()
   
    plt.loglog(freqs,np.abs(crossp_phase_unfilt))
    plt.show()

    max_ind = np.argmax(crossp_phase_unfilt[(freqs > 40) & (freqs < 1000)])
    
    lf = freqs[max_ind]
    print(lf)
    crossp_phase = bp_filt(crossp_phase, lf, Ns, Fs, bw)

    fft = np.fft.rfft(crossp_phase)
    

    if True:#plot_phase_fft and n_jobs == 1:
        plt.loglog(freqs, np.abs(fft))
        plt.show()

    crossp_phase_z = signal.hilbert(crossp_phase)
    crossp_phase_amp = np.abs(crossp_phase_z)
    
    plt.semilogy(crossp_phase)
    plt.semilogy(crossp_phase_amp)
    plt.show()
    
    crossp_phase_amp, errs = buf.rebin_vectorized(crossp_phase_amp, Ns/downs_num)
    tarr = buf.rebin_mean(tarr, Ns/downs_num)
   
    max_val = np.amax(crossp_phase_amp)
    max_ind = np.argmax(crossp_phase_amp)
    print(max_val)

    start_ind = max_ind
    
    length_of_arrays = len(tarr) - max_lfe #length of all arrays after cutting out data points greater than this value

    mask = (tarr > tarr[start_ind])

    
    if len(crossp_phase_amp[mask]) < length_of_arrays: #Check if array is too short for proper cutting of array
        print('array is too short', len(crossp_phase_amp[mask]))
        if n_jobs == 1:
            plt.plot(crossp_phase_amp)
            plt.show()

    else:
        crossp_phase_amp = crossp_phase_amp[mask][:length_of_arrays]
        tarr = tarr[mask][:length_of_arrays]
        errs = errs[mask][:length_of_arrays]
        print('cut')

    print(len(crossp_phase_amp))
    
    if plot_phase_env and n_jobs == 1:
        plt.plot(tarr,crossp_phase_amp)
        plt.show()
    
    save_base_name= save_base_name_save_trans + 'raw_curves/' + '{}/'.format(fs) 
    save_transients_name = save_base_name + '{}'.format(raw_file_name.split('.')[0])
    print(save_transients_name)
    
    bu.make_all_pardirs(save_base_name)

    if save_transients:
        np.savez(save_transients_name, tarr=tarr, crossp_phase_amp=crossp_phase_amp, dg=dg, Ns=Ns, Fs=Fs, errs=errs)

    
def extract_y_err_env_and_osc_save(filenames, lf, bw, max_lfe, fs):

    print('extract_y_err_env_and_osc_save')
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

        #crossp_phase = bp_filt(crossp_phase, lf, Ns, Fs, bw)
        crossp_phase = hp_filt(crossp_phase, 70, Ns, Fs) 

        fft = np.fft.rfft(crossp_phase)
       
        #plt.loglog(freqs, np.abs(fft))
        #plt.show()

        max_val = np.amax(crossp_phase)
        max_ind = np.argmax(crossp_phase)
        print(max_val)

        start_ind = max_ind
        
        length_of_arrays = len(tarr) - max_lfe #length of all arrays after cutting out data points greater than this value

        mask = (tarr > tarr[start_ind])
        
        
        if len(crossp_phase[mask]) < length_of_arrays: #Check if array is too short for proper binning
            print('array is too short', len(crossp_phase[mask]))
            plt.plot(crossp_phase)
            plt.show()

        else:
            crossp_phase = crossp_phase[mask][:length_of_arrays]
            tarr = tarr[mask][:length_of_arrays]
            print('cut')

        #plt.plot(tarr,crossp_phase)
        #plt.show()
        save_base_name= save_base_name_save_trans + 'raw_curves_env_and_osc/' + '{}/'.format(fs) 
        save_transients_name = save_base_name + '{}'.format(raw_file_name.split('.')[0])
        print(save_transients_name)
    
        bu.make_all_pardirs(save_base_name)

        if save_transients:
            np.savez(save_transients_name, tarr=tarr, crossp_phase_amp=crossp_phase, dg=dg, Ns=Ns, Fs=Fs)

def extract_trans_rebin(filenames, lf, bw, max_lfe, fs):

    print('extract_trans_rebin')
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

        #plt.plot(tarr,crossp_phase)
        #plt.show()
        
        crossp_phase = bp_filt(crossp_phase, lf, Ns, Fs, bw)
        #crossp_phase = hp_filt(crossp_phase, 70, Ns, Fs) 

        fft = np.fft.rfft(crossp_phase)
       
        #plt.loglog(freqs, np.abs(fft))
        #plt.show()

        max_val = np.amax(crossp_phase)
        max_ind = np.argmax(crossp_phase)
        print(max_val)

        #plt.plot(tarr, crossp_phase)
        #plt.show()

        start_ind = max_ind
        
        length_of_arrays = len(tarr) - max_lfe #length of all arrays after cutting out data points greater than this value

        mask = (tarr > tarr[start_ind])
        
        
        if len(crossp_phase[mask]) < length_of_arrays: #Check if array is too short for proper binning
            print('array is too short', len(crossp_phase[mask]))
            plt.plot(crossp_phase)
            plt.show()

        else:
            crossp_phase = crossp_phase[mask][:length_of_arrays]
            tarr = tarr[mask][:length_of_arrays]
            print('cut')

        #plt.plot(tarr,crossp_phase)
        #plt.show()
        save_base_name= save_base_name_save_trans + 'raw_curves_env_and_osc/' + '{}/'.format(fs) 
        save_transients_name = save_base_name + '{}'.format(raw_file_name.split('.')[0])
        print(save_transients_name)

        x, y, errs = buf.rebin(tarr, crossp_phase, nbins=Ns/downs_num, plot=False)

        
        bu.make_all_pardirs(save_base_name)

        if save_transients:
            np.savez(save_transients_name, tarr=x, crossp_phase_amp=y, errs=errs, dg=dg, Ns=Ns, Fs=Fs, downsamp_num=downs_num)

def extract_trans_rebin_parallel(filename, lf, bw, max_lfe, fs):

    print('extract_trans_rebin')
    y_curves = []
    tarrs = []
    start_inds = []
    dg_arr = []
    save_arr = []
    
    print(filename)
    
    raw_file_name = filename.split('/')[-1]
    meas_name = filename.split('/')[-2]
    
    obj = hsDat(filename)
    
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

    #plt.plot(tarr,crossp_phase)
    #plt.show()
    
    crossp_phase = bp_filt(crossp_phase, lf, Ns, Fs, bw)
    #crossp_phase = hp_filt(crossp_phase, 70, Ns, Fs) 

    fft = np.fft.rfft(crossp_phase)
    
    #errs = buf.rebin_mean(crossp_phase, Ns/downs_num) 
    #crossp_phase = buf.rebin_mean(crossp_phase, Ns/downs_num)
    #crossp_phase, errs = buf.rebin_vectorized(crossp_phase, Ns/downs_num)
    #tarr = buf.rebin_mean(tarr, Ns/downs_num)
    
    #plt.loglog(freqs, np.abs(fft))
    #plt.show()

    max_val = np.amax(crossp_phase)
    max_ind = np.argmax(crossp_phase)
    print(max_val)

    #plt.plot(tarr, crossp_phase)
    #plt.show()

    start_ind = max_ind
    
    length_of_arrays = len(tarr) - max_lfe #length of all arrays after cutting out data points greater than this value

    mask = (tarr > tarr[start_ind])
   
    
    if len(crossp_phase[mask]) < length_of_arrays: #Check if array is too short for proper binning
        print('array is too short', len(crossp_phase[mask]))
        plt.plot(crossp_phase)
        plt.show()

    else:
        crossp_phase = crossp_phase[mask][:length_of_arrays]
        tarr = tarr[mask][:length_of_arrays]
        print('cut')

    #crossp_phase, errs = buf.rebin_vectorized(crossp_phase, len(crossp_phase)/downs_num)
    #tarr = buf.rebin_mean(tarr, len(tarr)/downs_num)
    errs = np.ones_like(crossp_phase)
    #plt.plot(tarr,crossp_phase)
    #plt.show()
    save_base_name= save_base_name_save_trans + 'raw_curves_env_and_osc/' + '{}/'.format(fs) 
    save_transients_name = save_base_name + '{}'.format(raw_file_name.split('.')[0])
    print(save_transients_name)

    #plt.scatter(tarr,crossp_phase)
    #plt.scatter(x,y)
    #plt.show()

    bu.make_all_pardirs(save_base_name)

    if save_transients:
        np.savez(save_transients_name, tarr=tarr, crossp_phase_amp=crossp_phase, errs=errs, dg=dg, Ns=Ns, Fs=Fs, downsamp_num=downs_num)



def gaussian_check(filenames, lf, bw, pl, max_lfe, nbins, n, nsave, save_hists, ncurves, end_time):
    
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
        
        #if len(crossp_phase_amp[mask]) < length_of_arrays: #Check if array is too short for proper binning
        #    print('array is too short')
            
        #else:
        #    crossp_phase_amp = crossp_phase_amp[mask][:length_of_arrays] 
        #    tarr = tarr[mask][:length_of_arrays]
        #    print('cut')
        
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
    for i, folder in enumerate(folders[:]):
        files, zero = bu.find_all_fnames(folder, sort_time=True)
        
        folder_save = folder.split('/')[-1]
        
        if save_inst_phase_env: 
            Parallel(n_jobs=n_jobs)(delayed(extract_y_err_save_parallel)(f, libration_freq, wind_bandwidth, max_length_from_end, folder_save) for i, f in enumerate(files))

        elif save_inst_phase_env_and_osc:
            extract_y_err_env_and_osc_save(files, libration_freq, wind_bandwidth, max_length_from_end, folder_save) 

        elif save_inst_phase_rebin:
            Parallel(n_jobs=n_jobs)(delayed(extract_trans_rebin_parallel)(f, libration_freq, wind_bandwidth, \
                    max_length_from_end,folder_save) for i, f in enumerate(files)) 

            #extract_trans_rebin(files, libration_freq, wind_bandwidth, max_length_from_end, folder_save)
else:
    if save_inst_phase_env:
        folder_save = base_folder.split('/')[-1]

        Parallel(n_jobs=n_jobs)(delayed(extract_y_err_save_parallel)(f, libration_freq, wind_bandwidth, max_length_from_end, folder_save) for i, f in enumerate(files))

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











