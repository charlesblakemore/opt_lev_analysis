import matplotlib.pyplot as plt
import numpy as np

#datafile = '/home/dmartin/Desktop/simulations/libration_3/slow_rot_field/16.5e3Vm_data.npy'
datafile = '/home/dmartin/Desktop/simulations/efield_phase_mod/test_faster_phase_mod_data.npy'
parameters = '/home/dmartin/Desktop/simulations/libration_3/slow_rot_field/33e3Vm_5_4_parameters.npy'

data = np.load(datafile)
params = np.load(parameters, allow_pickle=True)

time = data[:,0]
mask = time >= 0. 

w_arr = {'units': 'blah', 'axes': ['$\omega_{x}$','$\omega_{y}$'\
        ,'$\omega_{z}$'],'data': data[:,1:4]}

p_arr = {'units': 'blah', 'axes': ['$p_{x}$','$p_{y}$','$p_{z}$'] , 'data': data[:,7:10]}

Efield_arr = {'units': 'blah', 'axes': ['$E_{x}$','$E_{y}$','$E_{z}$'],'data': data[:,10:14]}

cart_to_sphere = False

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

def plot_data_fft(arr, tarr, Ns, Fs, axes=[], units='blah'):
    fig, ax = plt.subplots(arr.shape[1], 1, figsize=(7,4), sharex = True, dpi = 100)
    
    
    freqs = np.fft.rfftfreq(Ns, 1./Fs)
    mask = (tarr > 50) & (tarr < 100)
    
    for i in range(len(ax)):
        fft = np.fft.rfft(arr[:,i])
        ax[i].loglog(freqs, 2 * np.abs(fft)/(len(fft)))
        
        if axes:
            ax[i].set_ylabel(axes[i] + ' [{}]'.format(units))

        ax[i].set_xlabel('Frequency [Hz]')
        ax[i].legend()
    
    plt.show()

def plot_data(arr, tarr, axes=[], units='blah'):
    fig, ax = plt.subplots(arr.shape[1], 1, figsize=(7,4), sharex = True, dpi = 100)
    
    
    freqs = np.fft.rfftfreq(Ns, 1./Fs)
    mask = (tarr > 50) & (tarr < 100)
    
    for i in range(len(ax)):
        fft = np.fft.rfft(arr[:,i])
        if axes:
            ax[i].plot(tarr,arr[:,i], label=axes[i])
        else:
            ax[i].plot(tarr,arr[:,i])
        ax[i].set_ylabel(axes[i] + ' [{}]'.format(units))
        ax[i].set_xlabel('Time [s]')
        ax[i].legend()
    
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
    plot_data(p_arr['data'][mask], time[mask], p_arr['axes'], p_arr['units'])

    plot_data_fft(p_arr['data'][mask], time[mask],Ns, Fs, p_arr['axes'], p_arr['units'])

plot_data(w_arr['data'][mask], time[mask], w_arr['axes'], w_arr['units'])
#plot_data_fft(w_arr['data'][mask], time[mask],Ns, Fs, w_arr['axes'], w_arr['units'])
#plot_data(Efield_arr['data'], time, Efield_arr['axes'], Efield_arr['units'])

p_fft = np.fft.rfft(p_arr['data'][0])
E_fft = np.fft.rfft(Efield_arr['data'][0])

