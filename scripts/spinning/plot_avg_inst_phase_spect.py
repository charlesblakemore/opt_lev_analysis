from amp_ramp_3 import bp_filt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import bead_util as bu
from scipy.optimize import curve_fit
from scipy import signal

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

files, length = bu.find_all_fnames(directory, ext='.npz')
#files = ['/home/dmartin/Desktop/analyzedData/20191204/spinning/deriv_feedback/20191216/no_dg.npy']
files = files[:]
fils = ['/home/dmartin/Desktop/analyzedData/20191204/spinning/deriv_feedback/20191216/change_dg/drive_psds/change_dg_0049_drive_psds.npz', '/home/dmartin/Desktop/analyzedData/20191204/spinning/deriv_feedback/20191216/change_dg/crossp_psds/change_dg_0049.npz']
fils = ['/home/dmartin/Desktop/analyzedData/20200130/spinning/series_3/base_press/long_int/0_dg/crossp_psds/', \
        '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_3/base_press/long_int/0_8_dg/crossp_psds/',\
        '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_3/base_press/long_int/0_85_dg/crossp_psds/',\
        '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_3/base_press/long_int/0_9_dg/crossp_psds/']

sig_file = '/home/dmartin/Desktop/analyzedData/20191204/spinning/deriv_feedback/20191216/change_dg/crossp_psds/change_dg_0049.npz'
backg_file = '/home/dmartin/Desktop/analyzedData/20191204/spinning/deriv_feedback/20191219/crossp_psds/.npz'

crossp_dir = '/home/dmartin/Desktop/analyzedData/20191223/spinning/deriv_feedback/change_dg_0_to_0_15/crossp_psds/' 
drive_dir = '/home/dmartin/Desktop/analyzedData/20191223/spinning/deriv_feedback/change_dg_0_to_0_15/drive_psds/'

filt = True

drive = True

fit = False

fit_bandwidth = 20

libration_freq = 370

freq_window_low = libration_freq-0.5*fit_bandwidth
freq_window_high = libration_freq+0.5*fit_bandwidth

bp_bandwidth = 100

files.sort(key=lambda f: int(filter(str.isdigit,f)))
file_num = 15 


def load_data(filename):
    data = np.load(filename, allow_pickle=True)

    
    freqs = data['freqs']
    avg = data['avgs']
    Ns = data['Ns'][()]
    Fs = data['Fs'][()]
    psds = data['psds']
    num_files = data['num_files']
    z_arr = data['z_arr'] 

    return freqs, avg, psds, Ns, Fs, num_files

def psd_lorentzian(x, A, f0, g, b):
    w = 2*np.pi*x
    w0 = 2*np.pi*f0
    #denom =( (((1./g)+c)*w)**2 + (w0**2 - w**2)**2)
    denom = (((g)*w)**2 + (w0**2 - w**2)**2)
    return (1/denom)*((A*g**2)) + b

def chi_sq_single(y, f, yvar):
    return ((y-f)/yvar)**2

def two_psd_lorentzian(x, A1, A2, f01, f02, g1, g2, c1, c2, b1):
    w = 2*np.pi*x
    w01 = 2*np.pi*f01
    w02 = 2*np.pi*f02

    denom =( (((1/g1)+c1)*w)**2 + (w01**2 - w**2)**2)
    
    denom2 =  ((((1/g1)+c1)*w)**2 + (w02**2 - w**2)**2) 
    return (A1/denom)*((1./g1)+c1) + (A2/denom2)*((1./g1)+c1) + b1

def plot_sigs(directory, low, high, lib_freq=660, filt=False, bw=100, inst_freq=False):
    
    crossp_files, c_length = bu.find_all_fnames(directory + 'crossp_psds/', ext='.npz')
    drive_files, d_length = bu.find_all_fnames(directory + 'drive_psds', ext='.npz')
    
    crossp_files = crossp_files[24:]
    drive_files = drive_files[24:]

    
    for i in range(len(crossp_files)):
        crossp_data = np.load(crossp_files[i], allow_pickle=True) 
        drive_data = np.load(drive_files[i], allow_pickle=True)

        print(crossp_files[i])
        print(drive_files[i])
        
        Ns = int(crossp_data['Ns'])
        Fs = crossp_data['Fs']
        crossp_z = crossp_data['z_arr']
        drive_z = drive_data['z_arr']
        
        freqs = np.fft.rfftfreq(Ns, 1./Fs)
        tarr = np.arange(Ns)/Fs

        for j in range(len(crossp_z)):
            z = np.array([crossp_z[j], drive_z[j]])

            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()

            for k in range(len(z)):
                phase = signal.detrend(np.unwrap(np.angle(z[k])))

                phase_phase = signal.hilbert(phase)
                
                amp_phase = np.abs(phase_phase)

                if k == 0 and inst_freq:
                    phase_unnorm = phase
                    phase_norm = phase/amp_phase
                    #phase = np.gradient(phase_norm/(2*np.pi), 1./Fs)
                    phase = np.gradient(phase_unnorm)
                    label = [r'MS Instaneous Frequency ($\frac{d\phi}{dt}$)', \
                         r'Drive Instaneous Phase ($m(t)$)']
                elif k == 0 and not inst_freq:
                    label = [r'MS Instaneous phase ($\phi$)', \
                         r'Drive Instaneous Phase ($m(t)$)']
                    
                if filt:
                    phase = bp_filt(phase, lib_freq, Ns, Fs, bw)
                        
                    mask = (freqs > low) & (freqs < high)


                print('plots')
                fft = np.fft.rfft(phase)
                
                if filt:
                    ax1.semilogy(freqs[mask], np.abs(fft[mask]), label='FFT. ' + label[k])
                else:
                    ax1.loglog(freqs, np.abs(fft), label='FFT. ' + label[k])
                ax1.set_xlabel('Frequency [Hz]')
                ax1.set_ylabel('FFT of Instaneous Frequency [arb.]')
                ax1.legend()
                
                #if skip_drive and k == 1:
                #    continue
                    
                ax2.plot(tarr, phase, label= label[k])

                ax2.set_xlabel('Time [s]')
                ax2.set_ylabel('Instaneous Frequency [arb.]')
                ax2.legend()

                
            plt.show()

def plot_freq_range(filenames, low, high, cmap='viridis' ):
    colors = bu.get_color_map(len(filenames), cmap=cmap)
    
    avgs=[]
    for i in range(len(filenames)):
        data = np.load(filenames[i], allow_pickle=True)
        

        #freqs = data[0]
        #avg = data[1]
        #Ns = data[2]
        freqs = data['freqs']
        avg = data['avgs']
        Ns = data['Ns'][()]
        
        if len(avg) != 0 :

            avg_of_avg = np.mean(avg)

            if False:#avg_of_avg > 0.0004:
                print('miss')
                continue
            avgs.append(avg_of_avg)

            meas_name = filenames[i].split('/')[-1].split('.')[0]

            mask = (freqs < high) & (freqs > low)

            #plt.semilogy(freqs[mask],avg[mask],label=meas_name, color=colors[i])
            
            #hot to cold
            
            plt.semilogy(freqs[mask],avg[mask], color=colors[len(filenames)-1-i], label=i) 
        else:
            continue

    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'PSD $[rad^{2}/Hz]$')
    plt.legend()
    plt.show()
   
    #plt.plot(avgs)
    #plt.show()

def plot_many(filenames, cmap='inferno'):
    colors = bu.get_color_map(len(filenames),cmap=cmap)

    avgs=[]
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for i in range(len(filenames)):
        data = np.load(filenames[i], allow_pickle=True)
    
        #freqs = data[0]
        #avg = data[1]
        #Ns = data[2]
    
        freqs = data['freqs']
        avg = data['avgs']
        Ns = data['Ns'][()]
        z_arr = data['z_arr']

        print(z_arr.shape)
   
        for j, z in enumerate(z_arr):
        
            phase = signal.detrend(np.unwrap(np.angle(z)))

            plt.plot(phase)
            plt.show()
        #print(data)
    
        avg_of_avg = np.mean(avg)

        avgs.append(avg_of_avg)
        meas_name = filenames[i].split('/')[-1].split('.')[0]
    
        #plt.loglog(freqs,avg, label=meas_name, color=colors[i])

        ax1.loglog(freqs, avg, color=colors[i])
    
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel(r'PSD $[rad^{2}/Hz]$')
    plt.legend()
    plt.show()

    plt.plot(avgs)
    plt.show()
    
chi_sq_arr = []
damp = []
freq_ = []
a= []
A=[]
def fit_many_in_sequence(filenames, low, high):
    for i in range(len(filenames)):
        data = np.load(filenames[i], allow_pickle=True)

        print(data.files)
        freqs = data['freqs']
        avg = data['avgs']
        Ns = data['Ns'][()]
        stds = data['std']
        num_files = data['num_files']
        print(Ns)

        if len(avg) == 0:
            print('bad file')

            continue

        meas_name = filenames[i].split('/')[-1].split('.')[0]
        
        mask = (freqs > low) & (freqs < high)
    
        max_ind = np.argmax(avg[mask])
        freq_guess = freqs[mask][max_ind]
    
        #plt.loglog(freqs, avg)
        #plt.show()

        g_high = 1e2
        g_low = 1e-4
    
        #plt.semilogy(freqs[mask],avg[mask])
        #plt.show()
    
        print(freq_guess)
    
    
        res = (freqs[1]-freqs[0])
        
        p0 = [4e4, freq_guess, 1e-3, 200., 9.8e-7]
        bounds = ([1e2,freq_guess-res,0.,4e4,9e-7], [5e7,freq_guess+res, 1e4, 5e5, 9e-6])

        sigma = stds[mask]/(np.sqrt(num_files)) #This sigma should be a std of the mean. num_files is num
                                                                         #of files used in the mean
        popt, pcov = curve_fit(psd_lorentzian, freqs[mask], avg[mask], p0=p0, bounds=bounds, sigma=sigma)
        
        #p0 = [1,1, freq_guess,freq_guess, 1e-3,1e-3, 0,0, 0]
        #bounds = ([0,0,freq_guess-res,freq_guess-res,0,0,-1e5,-1e5,0], [1e10,1e10,freq_guess+res, freq_guess+res, 1e4, 1e4,1e5, 1e5, 1e5])
        chi_sq = 0
        for j, y in enumerate(avg[mask]):
            f = psd_lorentzian(freqs[mask][j], *popt)

            chi_sq += chi_sq_single(y, f, sigma[j]) 

        print(chi_sq/(len(avg[mask])-5), 'chi_sq')
        print(chi_sq, len(avg[mask]))
        print(popt)
        
        A.append(popt[0])
        chi_sq_arr.append(chi_sq/(len(avg[mask])-5))
        freq_.append(popt[1])
        damp.append(popt[2])
        a.append(popt[3])
        
        label1 = r'$\tau$' +' = {} s'.format((1/popt[2]).round(2))
        label2 = r'a' + ' = {} Hz'.format(popt[3].round(2))
        label3 = r'$f_{peak}$' + ' = {} Hz'.format(popt[1].round(2))

        full = label1 + ', ' + label2 + ', ' + label3
        full = label1 + ', ' + label3 
        x = np.linspace(freqs[mask][0], freqs[mask][-1], 10000)

        #leg = plt.legend()
        #leg.get_frame().set_linewidth(0.0)

        meas_name = filenames[i].split('/')[-1].split('.')[0]
        if False: #popt[2] > 100:
            plt.semilogy(freqs[mask], avg[mask], label=meas_name)
            plt.semilogy(x,psd_lorentzian(x, *popt),label=full)
            plt.legend()

            plt.xlabel('Frequency [Hz]')
            plt.ylabel(r'PSD $[rad^{2}/Hz]$')
            plt.show()
    file_arr = np.arange(len(freq_))

    #plt.scatter(file_arr,a, label='a')
    #plt.xlabel('file number')
    #plt.ylabel('a [Hz]')
    #plt.show()

    plt.scatter(file_arr,freq_, label='freq')
    plt.xlabel('file number')
    plt.ylabel('Libration frequency [Hz]')
    plt.show()
    
    plt.scatter(file_arr,damp, label='damp')
    plt.xlabel('file number')
    plt.ylabel(r'$\gamma$ [s]')
    plt.show()
   
    plt.scatter(file_arr, A, label='A')
    plt.xlabel('file number')
    plt.ylabel('A')
    plt.show()

    plt.scatter(file_arr, chi_sq_arr, label=r'$\chi^{2}$')
    plt.xlabel('file number')
    plt.ylabel(r'$\chi^{2}$/DOF')
    plt.show()

def fit_many_in_sequence_psd(filenames, low, high):
    chi_sq_means = []
    chi_sq_stds = []
    damp_means = []
    damp_stds = []
    freq_means = []
    freq_stds = []
    a_means = []
    a_stds = []
    A_means = []
    A_stds = []
    b_means =[]
    b_stds = []

    for i in range(len(filenames)):
        data = np.load(filenames[i], allow_pickle=True)

        print(data.files)
        freqs = data['freqs']
        avg = data['avgs']
        Ns = data['Ns'][()]
        Fs = data['Fs']
        stds = data['std']
        psds = data['psds']
        print(data)
        print(Ns/Fs, Ns)

        meas_name = filenames[i].split('/')[-1].split('.')[0]

        mask = (freqs > low) & (freqs < high)

        chi_sq_arr = []
        damp = []
        freq_ = []
        a= []
        A= []
        b= []   
        
        if len(psds) == 0:
            print('skip', i)
            A_means.append(0)
            A_stds.append(0)
            freq_means.append(0)
            freq_stds.append(0)
            damp_means.append(0)
            damp_stds.append(0)
            #a_means.append(0)
            #a_stds.append(0)
            b_means.append(0)
            b_stds.append(0)
            chi_sq_means.append(0)
            chi_sq_stds.append(0)
            continue

        for j, psd in enumerate(psds[:]):
            max_ind = np.argmax(psd[mask])
            freq_guess = freqs[mask][max_ind]

            #plt.loglog(freqs, avg)
            #plt.show()


            res = (freqs[1]-freqs[0])

            #p0 = [2e5, freq_guess, 0.4, 9.8e-7]
            #bounds = ([1e5, freq_guess-res, res,  9e-7], [5e5, freq_guess+res, 1e0, 9e-6])

            p0 = [2e4, freq_guess, 8, 9.8e-7]
            bounds = ([1e0, freq_guess-res, res,  9e-7], [5e5, freq_guess+res, 1e2, 9e-6])

            sigma = stds[mask]

            try:
                popt, pcov = curve_fit(psd_lorentzian, freqs[mask], psd[mask], p0=p0, bounds=bounds, sigma=sigma)
            except:
                print('failed fit')
                continue

            
            #fit = psd_lorentzian(freqs[mask], *popt)
            #
            #fit_max = np.amax(fit)
            #fit_max_ind = np.argmax(fit)

            #plt.plot(freqs[mask],fit)
            #plt.scatter(freqs[mask][fit_max_ind], fit_max)
            #plt.show()

            #cut = ((fit_max * 0.5 + 0.5*fit_max) < fit) #& ((fit_max * 0.5 + .5*fit_max) > fit) 
            #points = fit[cut]

            #plt.plot((freqs[mask])[cut], fit[cut])
            #plt.scatter(freqs[mask][fit_max_ind], fit_max)
            #
            #plt.show()

            #print(points)
            #raw_input()
            
            f_arr = []
            chi_sq = 0
            for j, y in enumerate(psd[mask]):
                f = psd_lorentzian(freqs[mask][j], *popt)
                f_arr.append(f)
                chi_sq += chi_sq_single(y, f, sigma[j])


            print(chi_sq/(len(psd[mask])-5), 'chi_sq')
            print(popt[:])

            chi_sq_dof = chi_sq/(len(psd[mask])-5)
            if chi_sq_dof > 2:
                continue
            A.append(popt[0])
            chi_sq_arr.append(chi_sq_dof)
            freq_.append(popt[1])
            damp.append(popt[2])
            #a.append(popt[3])
            b.append(popt[3])

            label1 = r'$\tau$' +' = {} s'.format(popt[2].round(2))
            #label2 = r'a' + ' = {} Hz'.format(popt[3].round(2))
            label2 = ''
            label3 = r'$f_{peak}$' + ' = {} Hz'.format(popt[1].round(2))

            full = label1 + ', ' + label2 + ', ' + label3

            x = np.linspace(freqs[mask][0], freqs[mask][-1], 10000)
            #leg = plt.legend()
            #leg.get_frame().set_linewidth(0.0)

            meas_name = filenames[i].split('/')[-1].split('.')[0]
            
            if False:#popt[2] > 100 or chi_sq_dof > 3:
                fig, ax = plt.subplots(2, 1, sharex=True)
                ax[0].semilogy(freqs[mask], psd[mask], label=meas_name)
                ax[0].semilogy(x,psd_lorentzian(x, *popt),label=full)
                #ax[0].semilogy(x,lorentzian(x, *popt[:-1]), label='lorentz') 

                ax[1].semilogy(freqs[mask], ((psd[mask]-f_arr)/sigma)**2 ) 
                plt.legend()

                plt.xlabel('Frequency [Hz]')
                ax[0].set_ylabel(r'PSD $[rad^{2}/Hz]$')
                ax[1].set_ylabel(r'$(\frac{y-f}{\sigma})^{2}$')
                plt.show()
        if np.mean(chi_sq_arr) > 2:
            print(chi_sq_arr)
            raw_input()
        
        A_means.append(np.mean(A))
        A_stds.append(np.std(A))
        freq_means.append(np.mean(freq_))
        freq_stds.append(np.std(freq_))
        damp_means.append(np.mean(damp))
        damp_stds.append(np.std(damp))
        #a_means.append(np.mean(a))
        #a_stds.append(np.std(a))
        b_means.append(np.mean(b))
        b_stds.append(np.std(b))
        chi_sq_means.append(np.mean(chi_sq_arr))
        chi_sq_stds.append(np.std(chi_sq_arr))
        
        
    file_arr = np.arange(len(filenames))

    #plt.scatter(file_arr,a_means, label='a')
    #plt.xlabel('file number')
    #plt.ylabel('a [Hz]')
    #plt.show()
   
    fig, ax = plt.subplots(3,1,sharex=True)

    #plt.scatter(file_arr,freq_means, label='freq')
    #plt.errorbar(file_arr, freq_means, yerr=freq_stds, fmt='.')
    #plt.xlabel('file number')
    #plt.ylabel('Libration frequency [Hz]')
    #plt.show()
    
    #plt.scatter(file_arr, damp_means, label='damp')
    ax[0].errorbar(file_arr, damp_means, yerr=damp_stds, fmt='.')
    #plt.xlabel('file number')
    ax[0].set_ylabel(r'$\gamma$ [Hz]')
    
    #plt.scatter(file_arr, A_means, label='A')
    ax[1].errorbar(file_arr, A_means, yerr=A_stds, fmt='.')
    #plt.xlabel('file number')
    ax[1].set_ylabel('A')
    
    
    #plt.scatter(file_arr, chi_sq_means, label=r'$\chi^{2}$')
    ax[2].errorbar(file_arr, chi_sq_means, yerr=chi_sq_stds, fmt='.')
    #plt.xlabel('file number')
    ax[2].set_ylabel(r'$\chi^{2}/DOF$')
    plt.show()

    plt.errorbar(file_arr, b_means, yerr=b_stds, fmt='.')
    plt.xlabel('file number')
    plt.ylabel(r'b')
    plt.show()

    return damp_means, damp_stds
def plot_psds_and_avg(filename, low, high, num_avg=1, cmap='inferno'):
    data = np.load(filename, allow_pickle=True)
    freqs = data['freqs']
    psds = data['psds']
    num_files = data['num_files']
    #z_arr = data['z_arr']

    print(num_files)

    if num_avg == num_files:
        psds = psds
    else:
        psds = psds[:-((num_files-1)-(num_avg-1))]
        num_files = num_avg
    print(len(psds))
    colors = bu.get_color_map(len(psds), cmap=cmap)

    avg = np.mean(psds, axis=0)

    mask = (freqs < high) & (freqs > low)
   
    for i in range(len(psds)):
        print(i)
        plt.semilogy(freqs[mask], psds[i][mask],label='{}'.format(i),\
                color=colors[num_files-1-i])
        

    plt.semilogy(freqs[mask], avg[mask], label='avg')

    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'PSD [$rad^{2}$/Hz]')
    #plt.legend()
    plt.show()

def fit_two(filename, low, high):
    data = np.load(filename, allow_pickle=True)

    Ns = int(data['Ns'])
    Fs = data['Fs']
    avg = data['avgs']
    stds = data['std']
    
    freqs = np.fft.rfftfreq(Ns, 1./Fs)
    mask = (freqs > low) & (freqs < high)

    max_ind = np.argmax(avg[mask])
    freq_guess = freqs[mask][max_ind]


    print(freq_guess)


    res = (freqs[1]-freqs[0]) * (50)
    print(res) 
    p0 = [1e3, 1e3, freq_guess-0.4,freq_guess, 1e-3, 1e-3, 0, 0, 1.5e-6]
    bounds = ([1e2,1e2,freq_guess-res,freq_guess-res,0,0,0, 0, 1e-7], [1e5,1e5,freq_guess+res, freq_guess+res, 1e4, 1e4, 1e5, 1e5, 09e-6])

    sigma = stds[mask]
    popt, pcov = curve_fit(two_psd_lorentzian, freqs[mask], avg[mask], p0=p0, bounds=bounds, sigma=sigma)

    print(popt)
    x = np.linspace(freqs[mask][0], freqs[mask][-1], len(freqs[mask])*100)

    p1 = [popt[0], popt[2], popt[4], popt[6], popt[8]]
    p2 = [popt[1], popt[3], popt[4], popt[6], popt[8]]
    plt.loglog(freqs[mask], avg[mask])
    plt.loglog(x, two_psd_lorentzian(x, *popt))
    plt.loglog(x, psd_lorentzian(x, *p1))
    plt.loglog(x, psd_lorentzian(x, *p2))
    plt.show()
    

def plot_crossp_and_drive(filenames, low, high):
    
    for i in range(len(filenames)):
    
        data = np.load(filenames[i], allow_pickle=True)
        meas_name = filenames[i].split('/')[-2]

        if meas_name == 'crossp_psds':
            label = 'Cross p'
        elif meas_name == 'drive_psds':
            label = 'Drive'
    
        freqs = data['freqs']
        avg = data['avgs']
    
        mask = (freqs < high) & (freqs > low)

        plt.semilogy(freqs[mask], avg[mask], label=label)

    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'PSD')
    plt.legend()
    plt.show()

def plot_crossp_background_and_signal(background_file, signal_file):
    backg_freqs, backg_avg, backg_psds, b_Ns, b_Fs, b_num_files\
            = load_data(background_file)
    sig_freqs, sig_avg, sig_psds, s_Ns, s_Fs, s_num_files = \
            load_data(signal_file)

    plt.loglog(backg_freqs, backg_avg)
    plt.loglog(sig_freqs, sig_avg)
    plt.show()

    plt.loglog(sig_freqs, sig_avg-backg_avg)
    plt.show()


def plot_libration_sig(filename):
    data = np.load(filename, allow_pickle=True)

    Ns = int(data['Ns'])
    Fs = data['Fs']

    z = data['z_arr']
    z = z[0]

    t = np.arange(Ns)/Fs
    freqs = np.fft.rfftfreq(Ns, 1/Fs)
    
    amp = np.abs(z)
    phase = signal.detrend(np.unwrap(np.angle(z)))

   
    phase = bp_filt(phase, 664, Ns, Fs, 100)
    
    phase_fft = np.fft.rfft(phase)
   
    plt.loglog(freqs, np.abs(phase_fft))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'$\phi$')
    plt.show()

    z_phase = signal.hilbert(phase)
    phase = np.unwrap(np.angle(z_phase))

    dt = t[1]-t[0]
    inst_freq = np.gradient(phase, dt)/(2*np.pi)

    inst_freq_fft = np.fft.rfft(inst_freq)

    plt.loglog(freqs, np.abs(inst_freq_fft))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Instaneous Frequency of $\phi$')
    plt.show()

    plt.plot(t, inst_freq)
    plt.xlabel('Time [s]')
    plt.ylabel(r'Instaneous Frequency of $\phi$') 
    plt.show()

#plot_libration_sig(files[0])

#plot_sigs(directory, freq_window_low,freq_window_high, lib_freq=665, filt=True, bw=50, inst_freq=True)

#plot_crossp_background_and_signal(backg_file, sig_file)
#plot_crossp_and_drive(files, 530, 630)
#print(files[file_num])
#plot_psds_and_avg(files[-2], freq_window_low, freq_window_high, num_avg=15)
#plot_many(files[0:3])
#plot_freq_range(files[:], freq_window_low, freq_window_high)#, skip=False)
#fit_many_in_sequence(files[:], freq_window_low, freq_window_high)
#fit_many_in_sequence_psd(files[:], freq_window_low, freq_window_high)

damp_avg_arr = []
damp_std_arr = []
dg = [0, 0.8, 0.85, 0.9]
for i, f in enumerate(fils):
    files, length = bu.find_all_fnames(f, ext='.npz')

    damp_means, damp_stds = fit_many_in_sequence_psd(files[:], freq_window_low, freq_window_high)
    
    damp_avg_arr.append(damp_means)
    damp_std_arr.append(damp_stds)
    

damp_avg_arr = np.array(damp_avg_arr)
damp_std_arr = np.array(damp_std_arr)

print(damp_avg_arr)
print(damp_std_arr)
plt.errorbar(dg, damp_avg_arr, yerr=damp_std_arr, fmt='o')
#plt.scatter(dg, damp_avg_arr)
plt.ylabel(r'$\gamma$ [Hz]')
plt.xlabel('dg scale factor [arb.]')
plt.yscale('log')
plt.grid(b=True, which='minor', axis='both')
plt.grid(b=True ,which='major', axis='both')
plt.show()

#fit_many_in_sequence_psd_mult(files[:], freq_window_low, freq_window_high)
#fit_two(files[0], freq_window_low, freq_window_high)
