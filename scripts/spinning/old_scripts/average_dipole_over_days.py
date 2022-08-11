import numpy as np
import matplotlib.pyplot as plt
import hs_digitizer as hd 
import bead_util_funcs as buf
import string
import os

plt.rcParams.update({'font.size': 14})

out_dir = "/home/charles/analyzedData_dmartin/20190626/pramp/"
base_dir = "/daq2/20190626/bead1/spinning/"

#first data taken
first_file = "/daq2/20190626/bead1/spinning/pramp/N2/wobble_1/wobble_0000/turbombar_powfb_xyzcool_0.h5"

#dipole_dir = "/processed_data/spinning/wobble/20190626/"
dipole_dir = "/home/charles/analyzedData_dmartin/20190626/pramp/dipole/"

load_files = ['/home/charles/analyzedData_dmartin/20190626/pramp/dipole/SF6_dipole_data.npy', \
              '/home/charles/analyzedData_dmartin/20190626/pramp/dipole/Xe_dipole_data.npy', \
              '/home/charles/analyzedData_dmartin/20190626/pramp/dipole/Kr_dipole_data.npy', \
              '/home/charles/analyzedData_dmartin/20190626/pramp/dipole/Ar_dipole_data.npy', \
              '/home/charles/analyzedData_dmartin/20190626/pramp/dipole/N2_dipole_data.npy', \
              '/home/charles/analyzedData_dmartin/20190626/pramp/dipole/He_dipole_data.npy', \
              '/home/charles/analyzedData_dmartin/20190626/pramp/dipole/wobble_slow_after-highp_later_dipole_data.npy', \
              '/home/charles/analyzedData_dmartin/20190626/pramp/dipole/wobble_slow_after-highp_dipole_data.npy', \
              '/home/charles/analyzedData_dmartin/20190626/pramp/dipole/long_wobble_dipole_data.npy', \
              '/home/charles/analyzedData_dmartin/20190626/pramp/dipole/initial_dipole_data.npy']#,'/home/charles/analyzedData_dmartin/20190626/pramp/dipole/wobble_many_slow.npy']
load_files = load_files[::-1]

ext_ = ".npy"

#t0 = buf.get_hdf5_time(first_file)

gases = ['wobble']

wobble = ['/wobble_many_slow/']
#gas_ = ['after-highp_slow_later.npy']#,'after-highp_slow_later.npy'
colors = buf.get_colormap(len(gases),cmap='Set1')

def get_dipoles_and_times(gas,wobble_file):
    
    dipole_files = []
    dipoles = []

    #files, lengths = buf.find_all_fnames(dipole_dir, ext=ext_)
    
#   for i in range(len(files)):
#       d_file = files[i].split('/')[-1]
#       print(d_file)   
#       #if d_file.split('_')[0] == gas_:
#       if d_file == gas:
#           dipole_files.append(d_file)
#           dipoles.append(np.load(files[i]))

    fil = dipole_dir + wobble_file.split('/')[-2] + ext_ 
    dipole_files.append(fil)
    dipoles.append(np.load(fil))

    time_files = []
    times = []
    for i in range(len(wobble)):    
        files, lengths = buf.find_all_fnames(base_dir + gas + wobble[i])
        for j in range(len(files)):
            if 'wobble_0000' in files[j] and 'turbombar_powfb_xyzcool_0' in files[j]:
                
                time = buf.get_hdf5_time(files[j])
                time_files.append(files[j]) 
                times.append((time-t0)*1e-9*(1/86400.))
    if gas == 'N2':
        gas = 'N2_2'
        for i in range(len(wobble)):    
            files, lengths = buf.find_all_fnames(base_dir + gas + wobble[i])
            for j in range(len(files)):
                if 'wobble_0000' in files[j] and 'turbombar_powfb_xyzcool_0' in files[j]:
                
                    time = buf.get_hdf5_time(files[j])
                    time_files.append(files[j]) 
                    times.append((time-t0)*1e-9*(1/86400.))
        gas = 'N2'
        

    if len(times) != len(dipoles):
        print('times and dipoles are not the same length') 

    return {'gas': gas, 'dipole_files': dipole_files,'time_files': time_files, 'dipoles': dipoles, 'times':times}

def plot_dipole_v_time():
    colors = buf.get_colormap(len(load_files),cmap='plasma')
    colors = buf.get_colormap(7,cmap='plasma')
    print(len(colors))
    fig, ax = plt.subplots(1,1,figsize=(6,3),dpi=200)
    for i in range(len(load_files)):
        
        data = np.load(load_files[i])
        dipoles = np.array(data.item().get('dipoles'))
        times = data.item().get('times')
        gas = data.item().get('gas')
        time_files = data.item().get('time_files')
        print((load_files[i]))

        # if gas == 'initial':
        #     for j in range(len(time_files)):
        #         gas = time_files[j].split('/')[-3]
        #         ax.scatter(times,dipoles[:,0] *(1/1.602e-19) * (1e6),c=colors[i],label='{}'.format(gas))
        #     continue

                
        if i < 4:
            gas = load_files[i].split('/')[-1].split('.')[-2]
            if i == 0:
                ax.scatter(times,dipoles[:,0] *(1/1.602e-19) * (1e6),c=colors[0],label='$d$ characterization')
            else:
                ax.scatter(times,dipoles[:,0] *(1/1.602e-19) * (1e6),c=colors[0])
        else:
            print(i)
            ax.scatter(times,dipoles[:,0] *(1/1.602e-19) * (1e6), \
                        c=colors[i-3], label='{}'.format(gas))
        
    
    ax.set_ylabel('Dipole Moment $(e\cdot\mu m)$')
    ax.set_xlabel('Time [Days]')
    ax.legend(loc=0, fontsize=12, ncol=2)
    plt.tight_layout()

    fig.savefig('/home/charles/plots/20190626/wobble/all_dipole_v_time.png')
    fig.savefig('/home/charles/plots/20190626/wobble/all_dipole_v_time.svg')

    plt.show()

plot_dipole_v_time()

#for j in range(len(wobble)):
#   data = get_dipoles_and_times(gases[0],wobble[j])
#   
#   print(data) 
#   
#   w = wobble[j].split('/')[-2]    
#
#   if int(raw_input('Save ')) == 1:
#       #print(out_dir + gases[j] + '_dipole_data.npy')
#       #np.save(out_dir + gases[j] + '_dipole_data.npy', data)
#       np.save(out_dir + w + '_dipole_data.npy', data)
'''
for j in range(len(gases)):
    d_gas = []
    
    for i in range(len(files)):
        filename = files[i].split('/')[-1]
        if gases[j] == filename.split('_')[0]:  
            d = np.load(files[i])
            d *= (1/1.602e-19) * (1e6)
            print(d)
            d_gas.append(d[0])
                
            
    #avg_d = np.mean(d_gas) * (1/1.602e-19) * (1e6)
    
            plt.scatter(j,d[0], c = colors[j])
            plt.errorbar(j, d[0], yerr=d[1])
    #print(avg_d)


x = np.arange(len(gases))
plt.ylabel('Dipole Moment [$e\cdot\mu m$]')
plt.xticks(x,gases)
plt.show()
'''
