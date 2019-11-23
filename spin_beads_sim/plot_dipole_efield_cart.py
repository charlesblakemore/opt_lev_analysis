import matplotlib.pyplot as plt
import numpy as np

directory = '/home/dmartin/Desktop/simulations/added_thermtorque/'
base_filename = 'test4'
#base_filename = 'step_func_efield_start_0Hz_stop_50s_16e1Vm'

datafile = directory + base_filename + '_data.npy'
parameters = directory + base_filename + '_parameters.npy'

data = np.load(datafile)
params = np.load(parameters, allow_pickle=True)

time = data[:,0]
mask = time >= 0.#(time >= 20.) & (time <= 100.)

w_arr = {'units': 'rad/s', 'axes': ['$\omega_{x}$','$\omega_{y}$'\
        ,'$\omega_{z}$'],'data': data[:,1:4]}

p_arr = {'units': '$e$' + ' $\mu m$', 'axes': ['$p_{x}$','$p_{y}$','$p_{z}$'] , 'data': data[:,7:10] * (1./1.602e-19) * 1.e6}

Efield_arr = {'units': 'V/m', 'axes': ['$E_{x}$','$E_{y}$','$E_{z}$'],'data': data[:,10:14]}
    
cart_to_sphere = False
save_p = False
save_p_fft = False
p_linear = False
p_plot_axes = []
set_limits = False

w_plot_axes = []

def cart_to_sphere_coord(arr): 
    print arr.shape
    p_x = arr[:,0]
    p_y = arr[:,1]
    p_z = arr[:,2]
    

    p_r = np.sqrt(p_x**2 + p_y**2 + p_z**2)
    
    p_theta = np.arctan(np.sqrt(p_x**2+p_y**2)/p_z)
    
    p_phi = np.arctan(p_y/p_x)
    
    p_spher_coord = np.array([p_r, p_theta, p_phi])
    p_spher_coord = np.transpose(p_spher_coord)

    return p_spher_coord

def plot_data_fft(arr, tarr, Ns, Fs, axes=[], units='blah', save_fft=False,\
                  linear=False, plot_axes=[]):
   
   
    arr = np.delete(arr, plot_axes, 1)

    fig, ax = plt.subplots(arr.shape[1], 1, figsize=(7,6), sharex = True, dpi = 200)
    
    
    freqs = np.fft.rfftfreq(Ns, 1./Fs)
    mask = (tarr > 50) & (tarr < 100)
    

    for i in range(arr.shape[1]):
        fft = np.fft.rfft(arr[:,i])
    
        if arr.shape[1] != 1:
            ax[i].loglog(freqs, 2 * np.abs(fft)/(len(fft)))
            
            if axes:
                ax[i].set_ylabel(axes[i] + ' [{}]'.format(units))

            ax[i].set_xlabel('Frequency [Hz]')
            #ax[i].legend()
            ax[i].minorticks_on() 
        else:
            for j, axis in enumerate(plot_axes):
                if axis:
                    ax.loglog(freqs, 2 * np.abs(fft)/(len(fft)))
                    ax.set_xlabel('Frequency [Hz]')
                    ax.minorticks_on()
            if axes:
                ax.set_ylabel(axes[j] + ' [{}]'.format(units))
    
    
    fig.tight_layout()
    if save_fft:
        fig.savefig(directory + base_filename + '_plot_fft.png')

    plt.show()

    if linear:
        fig, ax = plt.subplots(arr.shape[1], 1, figsize=(7,6), sharex = True, dpi = 200)    
        
        for i in range(arr.shape[1]):
            fft = np.fft.rfft(arr[:,i])
            
            if arr.shape[1] != 1:
                ax[i].semilogy(freqs, 2 * np.abs(fft)/(len(fft)))
                
                ax[i].minorticks_on()

                if axes:
                    ax[i].set_ylabel(axes[i] + ' [{}]'.format(units))

                ax[i].set_xlabel('Frequency [Hz]')
                #ax[i].legend()
            else:
                ax.semilogy(freqs, 2 * np.abs(fft)/(len(fft)))
                ax.minorticks_on()

                if axes:
                    for j, axis in enumerate(plot_axes):
                        if axis:
                            ax.set_ylabel(axes[j] + '[{}]'.format(units))

        plt.show()

def plot_data(arr, tarr, axes=[], units='blah', set_limits=False, save=False):
    
    fig, ax = plt.subplots(arr.shape[1], 1, figsize=(7,6), sharex = True, dpi = 200)
    
    for i in range(len(ax)):
        y_bot_lim = -1.5*(np.amax(arr[:,i]))
        y_top_lim = 1.5*(np.amax(arr[:,i]))

        if axes:
            ax[i].plot(tarr, arr[:,i], label=axes[i])
        else:
            ax[i].plot(tarr, arr[:,i])
        ax[i].set_ylabel(axes[i] + ' [{}]'.format(units))
        ax[i].set_xlabel('Time [s]')
       
        ax[i].minorticks_on()
        if set_limits:
            ax[i].set_ylim(y_bot_lim,y_top_lim)
        #ax[i].legend()
    
    fig.tight_layout()
    if save:
        fig.savefig(directory + base_filename + '_plot.png')

    plt.show()


if cart_to_sphere:
    p_arr = cart_to_sphere_coord(p_arr['data'])
    E_arr = cart_to_sphere_coord(Efield_arr['data'])
    
    
    phase_diff = E_arr[:,2]-p_arr[:,2]
    
    
    #plt.plot(time, E_arr[:,2])
    #plt.plot(time[mask], p_arr[:,2][mask])
    #plt.plot(time, phase_diff)
    #plt.show()
    
    p_arr = {'units': 'blah', 'axes': ['$p_{r}$', r'$p_{\theta}$', r'$p_{\phi}$'], 'data': p_arr}
       
    Efield_arr = {'units': 'blah', 'axes': ['$E_{r}$', r'$E_{\theta}$', r'$E_{\phi}$'], 'data': E_arr}


Ns = len(time[mask])
Fs = 1/(time[1]-time[0])

print Ns, Fs


#plot_data_fft(phase_diff[mask], time[mask], Ns, Fs, ['','',''],['','','']) 


if cart_to_sphere:
    plot_data(p_arr['data'], time, p_arr['axes'], p_ang_arr['units'])

else:
    plot_data(p_arr['data'][mask], time[mask], p_arr['axes'], p_arr['units'],save = save_p, set_limits=set_limits)

    plot_data_fft(p_arr['data'][mask], time[mask],Ns, Fs, p_arr['axes'], p_arr['units'], save_fft=save_p_fft, linear=p_linear, plot_axes=p_plot_axes)

plot_data(w_arr['data'][mask], time[mask], w_arr['axes'], w_arr['units'])

plot_data_fft(w_arr['data'][mask], time[mask],Ns, Fs, w_arr['axes'], w_arr['units'], plot_axes=w_plot_axes)

plot_data(Efield_arr['data'], time, Efield_arr['axes'], Efield_arr['units'])

p_fft = np.fft.rfft(p_arr['data'][0])
E_fft = np.fft.rfft(Efield_arr['data'][0])

