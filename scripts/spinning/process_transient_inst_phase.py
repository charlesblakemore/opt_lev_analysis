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

base_folder = '/data/old_trap/20200130/bead1/spinning/series_4/base_press/change_phi_offset_3/'

files, zeros, folders = bu.find_all_fnames(base_folder, ext='.h5', sort_time=True, \
                    add_folders=True)
print(folders)
raw_input()

#folders = ['/data/old_trap/20200130/bead1/spinning/series_2/base_press/change_phi_offset_0_85_dg/']

#folders = ['/data/old_trap/20200130/bead1/spinning/series_3/base_press/change_phi_offset/0_dg_noise_added_0_009/']

save = False
multiple_folders = True

fit_curves = False
plot = True
plot_multiple = False
plot_drive_and_inst_phase = True

if save:
    save_base_name = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_3/base_press/'
    bu.make_all_pardirs(save_base_name)


files, zero = bu.find_all_fnames(directory, sort_time=True)
crossp_ind = 0
drive_ind = 1

wind_bandwidth = 100
libration_freq = 373

threshold = 0.6

def exp(x, a, b , c):
    return a*np.exp((x-b)*c)  

def chi_sq_single(y, f, yvar):
    return ((y-f)/yvar)**2

def plot_transient(filenames, lf, bw, thr, pl, fit):

    obj0 = hsDat(filenames[0])

    t0 = obj0.attribs['time']

    colors = bu.get_color_map(len(filenames), 'inferno')


    phi_inst_amps_x = []#[0 for i in range(len(filenames))]
    phi_inst_amps_y = []#[0 for i in range(len(filenames))]
    fit_curves_x= []#[0 for i in range(len(filenames))]
    fit_curves_y= []#[0 for i in range(len(filenames))]
    fit_params = []#[0 for i in range(len(filenames))]
    fit_pcov = []#[0 for i in range(len(filenames))]

    for i, f in enumerate(filenames):
        print(f, i)
        obj = hsDat(f)
        
        dg = obj.attribs['start_pm_dg']
        
        Ns = obj.attribs['nsamp']
        Fs = obj.attribs['fsamp']
        freqs = np.fft.rfftfreq(Ns, 1./Fs)

        tarr = np.arange(Ns)/Fs

        crossp = obj.dat[:,crossp_ind]
        drive = obj.dat[:,drive_ind]

        drive_fft = np.fft.rfft(drive)
        crossp_fft = np.fft.rfft(crossp)

        #plt.loglog(freqs, np.abs(crossp_fft))
        #plt.loglog(freqs, np.abs(drive_fft))
        #plt.show()

        crossp_z = signal.hilbert(crossp)
        drive_z = signal.hilbert(drive)
        

        drive_phase = signal.detrend(np.unwrap(np.angle(drive_z)))
        crossp_phase = signal.detrend(np.unwrap(np.angle(crossp_z)))
       
        crossp_phase = bp_filt(crossp_phase, lf, Ns, Fs, bw)

        crossp_phase_z = signal.hilbert(crossp_phase)
        crossp_phase_amp = np.abs(crossp_phase_z)

        window = np.hanning(Ns)
        crossp_phase_fft = np.fft.rfft(crossp_phase*window)

        crossp_phase_psd, freqs = mlab.psd(crossp_phase, Ns, Fs, window=mlab.window_none(crossp_phase))#mlab.window_hanning(crossp_phase))
        drive_phase_psd, freqs = mlab.psd(drive_phase, Ns, Fs, window=mlab.window_none(drive_phase))#(drive_phase))

        
        diff_max = 0
        for j, val in enumerate(drive_phase):
            diff = np.abs(drive_phase[0] - val)
            
            if diff > diff_max:
                diff_max = diff
            if diff_max > thr:
                start = val
                start_ind = j
                print('change found')
                break



        max_val = np.amax(crossp_phase)
        max_ind = np.argmax(crossp_phase)

        start_ind = max_ind
        #plt.plot(tarr[start_ind:],crossp_phase[start_ind:])
        #plt.plot(tarr,drive_phase)
        #plt.plot(tarr[start_ind:], crossp_phase_amp[start_ind:])
        #plt.scatter(tarr[start_ind], max_val)
        #plt.show()        
       
        print(tarr[max_ind])
        p0 = [max_val, tarr[start_ind:][0], crossp_phase_amp[-1]] 
        bounds = ([0, 0, -10], [5, 0.5 , 10])      
        print(p0)

        end = 1#tarr[-1]
        
        #mask = (tarr > tarr[start_ind]) & (tarr < end )
        mask = tarr > 0
        #sigma = np.ones(len(tarr[mask]))

        x = np.linspace(tarr[start_ind:][0], tarr[start_ind:][-1], len(tarr[start_ind:])*10)

        if fit:
            try:
                popt, pcov = curve_fit(exp, tarr[mask], crossp_phase_amp[mask], p0=p0, bounds=bounds)#, sigma=sigma)
            except:
                print('Failed Fit')
                #popt = [0 for k in range(len(p0))]
                #pcov = [0]
                #fit_curve = np.zeros_like(x)
                #fit_x = np.zeros_like(x)
               
                #fit_params[i] = popt, pcov
                #fit_curves[i] = fit_x, fit_curve
                #phi_inst_amps[i] = tarr[start_ind:], crossp_phase_amp[start_ind:]
                
                continue
            
            
            mask = (tarr > tarr[start_ind]) & (tarr < end )

            print(popt)

            fit_label = r'a$e^{c(t-b)}$, ' + 'a={} rad, b={} s, c={} Hz'.format(popt[0].round(2), popt[1].round(2), popt[2].round(2))

            #sigma = np.ones(len(x))

            #chi_sq = 0
            #for j, y in enumerate(avg[mask]):
            #    f = exp(x[j], *popt)
            #    
            #    chi_sq += chi_sq_single(y, f, sigma[j])

            fit_curve = exp(x, *popt)
            fit_x = x

            fit_params.append(popt)
            fit_pcov.append(pcov)
            fit_curves_x.append(fit_x)
            fit_curves_y.append(fit_curve)
            phi_inst_amps_x.append(tarr[start_ind:])
            phi_inst_amps_y.append(crossp_phase_amp[start_ind:])

            #fit_params[i] = popt
            #fit_pcov[i] = pcov
            #fit_x[i] = fit_x
            #fit_y[i] = fit_curve
            #phi_inst_amps_x[i] = tarr[start_ind:]
            #phi_inst_amps_y[i] = crossp_phase_amp[start_ind:]

        if pl:
           
            #plt.plot(tarr[mask], crossp_phase_amp[mask], label='data')
            #plt.plot(fit_x, fit_curve, label=fit_label)
            plt.plot(tarr, crossp_phase_amp)

            plt.xlabel('Time [s]')
            plt.ylabel(r'Instantaneous Amplitude of $\phi$')
            plt.legend()
            #plt.loglog(freqs, np.abs(crossp_phase_fft))
    #if pl:
            plt.show()
            
        
    return fit_params, fit_pcov, phi_inst_amps_x, phi_inst_amps_y, dg

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

if plot_drive_and_inst_phase:
    files, zero = bu.find_all_fnames(folders[1], sort_time=True)
    
    for i,f in enumerate(files):
        plot_drive_w_phase(f, libration_freq, wind_bandwidth)

if False:#plot_multiple:
    file_arr = []
    for i, folder in enumerate(folders):
        files, zero = bu.find_all_fnames(folder, sort_time=True)

        file_arr.append(files[-1])
        
    plot_multiple(file_arr, libration_freq, wind_bandwidth)

if multiple_folders:
    
    measured_damp_avg = []
    measured_damp_std = []
    measured_t_offset = []
    measured_a_avg = []
    measured_a_std = []
    dg_arr = []
    for i, folder in enumerate(folders):
        files, zero = bu.find_all_fnames(folder, sort_time=True)
        files = files[1:]
        fit_params, fit_pcov, phi_inst_amps_x, phi_inst_amps_y, dg = \
                plot_transient(files, libration_freq, wind_bandwidth, threshold, plot, fit_curves)
        
        #fit_params = np.array(fit_params)

        #print(fit_params)
        #avg = np.mean(fit_params, axis=0)
        #std = np.std(fit_params, axis=0)
        #print(avg)
        #measured_a_avg.append(avg[0])
        #measured_a_std.append(std[0])
        #measured_t_offset = avg[1], std[1]
        #measured_damp_avg.append(avg[2])
        #measured_damp_std.append(std[2])
        #dg_arr.append(dg)
    
        

        
        if save:
            meas_name = files[-1].split('/')[-2]
            
            
            if meas_name == '':
                print('empty name')
                meas_name = '{}'.format(i)
            
            save_name = save_base_name + meas_name
            
            np.savez(save_name, fit_params=fit_params, fit_pcov=fit_pcov, phi_inst_amps_x=phi_inst_amps_x, phi_inst_amps_y=phi_inst_amps_y, dg=dg)
            
    #plt.plot(dg_arr, measured_damp_avg)
    #plt.show()
else:
    fit_params, fit_curves, phi_inst_amps = plot_transient(files[1:], libration_freq, wind_bandwidth, threshold)

    if save:
            meas_name = files[i].split('/')[-2]


            if meas_name == '':
                print('empty name')
                meas_name = '{}'.format(i)

            save_name = save_base_name + meas_name
            np.savez(save_name, fit_params=fit_params, fit_curves=fit_curves, phi_inst_amps=phi_inst_amps)












