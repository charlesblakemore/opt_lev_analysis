import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.constants as constants
import scipy.signal as signal

import sys, time, os

import bead_util as bu


fac = 425 * constants.elementary_charge

base_path = '/processed_data/comsol_data/patch_potentials/'


fnames = bu.find_all_fnames(base_path, ext='')
names = []

for filname in fnames:
    parts = filname.split('/')
    name = parts[-1].split('.')[0]
    if name not in names:
        names.append(name)


names = [#'patch_pot_2um_1Vrms_50um-deep-patches', \
         #'patch_pot_2um_1Vrms_150um-deep-patches', \
         #'patch_pot_2um_1Vrms_250um-deep-patches', \
         #'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC', \
         #
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed0', \
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed10', \
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed20', \
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed30', \
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed40', \
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed50', \
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed60', \
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed70', \
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed80', \
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed90', \
         'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed100', \
        ]

base_name = 'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC_seed'
for seed in np.linspace(0,2000,21):
    names.append(base_name + str(int(seed)))


#names = []
#base_name = 'patch_pot_2um_1Vrms_150um-deep-patches_4mmBC'
#names.append(base_name + '_FINGER')
#names.append(base_name + '_PATCH')
#names.append(base_name + '_PATCH-FINGER')

#print names


#style_dict = {0: '--', 1: ':', 2: '-', 3: '-.'}
#label_dict = {0: '$\pm$500mV on Fingers', 1: '1Vrms Patches', 2: 'Sum'}
#color_dict = {0: 'C0', 1: 'C1', 2: 'C2',}

style_dict = {}
color_dict = {}
for nameind in range(len(names)):
    style_dict[nameind] = '-'
    color_dict[nameind] = 'C0'

all_rms = [[], [], []]
mean_rms = []
finger_rms = []
N = 0

for nameind, name in enumerate(names):
    print('Processing: ', name)
    xx = np.load(open(base_path + name + '.xx', 'rb'))
    yy = np.load(open(base_path + name + '.yy', 'rb'))
    zz = np.load(open(base_path + name + '.zz', 'rb'))

    field = np.load(open(base_path + name + '.field', 'rb'))
    potential = np.load(open(base_path + name + '.potential', 'rb')) 


    pot_func = interp.RegularGridInterpolator((xx, yy, zz), potential)

    field_func = []
    for resp in 0,1,2:
        field_func.append( interp.RegularGridInterpolator((xx, yy, zz), field[resp]) )


    posvec = np.linspace(-50e-6, 50e-6, 101)
    ones = np.ones_like(posvec)
    xval = 12.0e-6
    zval = 0.0e-6
    eval_pts = np.stack((xval*ones, posvec, zval*ones), axis=-1)

    ann_str = 'Sep: %0.2f um, Height: %0.2f um' % (xval*1e6, zval*1e6)

    
    plt.figure(0)
    plt.plot(posvec*1e6, pot_func(eval_pts), color='C0', ls=style_dict[nameind])

    plt.figure(1, figsize=(7,5))
    plt.title(name)
    if nameind == 0:
        plt.plot(posvec*1e6, field_func[0](eval_pts)*fac, label='fx', color='C0', \
                 ls=style_dict[nameind])
        plt.plot(posvec*1e6, field_func[1](eval_pts)*fac, label='fy', color='C1', \
                 ls=style_dict[nameind])
        plt.plot(posvec*1e6, field_func[2](eval_pts)*fac, label='fz', color='C2', \
                 ls=style_dict[nameind])
    else:
        plt.plot(posvec*1e6, field_func[0](eval_pts)*fac, color='C0', \
                 ls=style_dict[nameind])
        plt.plot(posvec*1e6, field_func[1](eval_pts)*fac, color='C1', \
                 ls=style_dict[nameind])
        plt.plot(posvec*1e6, field_func[2](eval_pts)*fac, color='C2', \
                 ls=style_dict[nameind])
    plt.legend()
    plt.xlabel('Displacement Along Cantilever Face [um]')
    plt.ylabel('Force on 425e$^-$ [N]')
    plt.annotate(ann_str, xy=(0.2, 0.9), xycoords='axes fraction')
    plt.tight_layout()
    if name == names[-1]:
        plt.grid()
    

    xx_plot = xx[xx > 0.9e-6]

    rms_force = [[], [], []]
    #rms_force = []
    for sepind, sep in enumerate(xx_plot):
        rms_val = 0.0
        eval_pts = np.stack((sep*ones, posvec, zval*ones), axis=-1)
        for resp in [0,1,2]:
            forcevec = field_func[resp](eval_pts) * fac
            rms_val += np.std(forcevec)
            rms_force[resp].append(rms_val)
        #rms_val *= 1.0 / np.sqrt(3)
        #rms_val *= 1.0 / 3
        #rms_force.append(np.sqrt(rms_val))
        
    for resp in [0,1,2]:
        all_rms[resp].append(rms_force[resp])

    if 'FINGER' in name:
        if 'PATCH' not in name:
            finger_rms = np.copy(rms_force)

    if not len(mean_rms):
        mean_rms = rms_force
    else:
        mean_rms += rms_force
    N += 1
    #for resp in [0,1,2]:
    #    if resp == 0:
    #        plt.loglog(xx_plot*1e6, rms_force[resp]*fudge_dict[nameind], label=name, \
    #                   ls=style_dict[resp], color=color_dict[nameind])
    #    else:
    #        plt.loglog(xx_plot*1e6, rms_force[resp]*fudge_dict[nameind], \
    #                   ls=style_dict[resp], color=color_dict[nameind])
        
    plt.figure(2)
    #plt.title('RMS Force vs. X: Different Patch Realizations')
    for resp in [0]:#,1,2]:
        plt.loglog(xx_plot*1e6, rms_force[resp], color=color_dict[nameind], \
                   ls=style_dict[nameind])#, \
                   #label=label_dict[nameind])
    plt.xlabel('X [um]')
    plt.ylabel('RMS force on 425e$^-$ [N]')
    plt.legend(loc=0)


    #if '250um' in name:
        


    '''
    xz_plane = np.meshgrid(xx, zz, indexing='ij')

    levels = np.linspace(np.min(potential), np.max(potential), 100) * 0.1
    levels = 100

    for i in range(10):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cont = ax.contourf(xz_plane[0]*1e6, xz_plane[1]*1e6, potential[:,-(i+1),:], levels)
        cbar = plt.colorbar(cont)
        ax.set_xlabel('Displaccment Along Cantilever [um]')
        ax.set_ylabel('Height [um]')
        plt.tight_layout()
    plt.show()
    '''

mean_rms = [[], [], []]
std_rms = [[], [], []]
for resp in [0,1,2]:
    mean_rms[resp] = np.array(all_rms[resp]).mean(axis=0)
    std_rms[resp] = np.array(all_rms[resp]).std(axis=0)


max_xrms = np.array(all_rms[0]).max(axis=0)
min_xrms = np.array(all_rms[0]).min(axis=0)
std_xrms = np.array(all_rms[0]).std(axis=0)

mean_xrms = np.array(all_rms[0]).mean(axis=0)
plt.figure(4)
plt.loglog(xx_plot*1e6, mean_xrms, color='C0', ls='-', label='X')
plt.fill_between(xx_plot*1e6, mean_xrms+std_xrms, mean_xrms-std_xrms, \
                 color='C0', alpha=0.2, edgecolor='C0')
plt.loglog(xx_plot*1e6, mean_xrms+std_xrms, color='C0', ls=':', label='X+', \
           alpha=0.4)
plt.loglog(xx_plot*1e6, mean_xrms-std_xrms, color='C0', ls=':', label='X-', \
           alpha=0.4)
#plt.loglog(xx_plot*1e6, min_xrms, color='C1', ls='--', label='Xmin')
#plt.loglog(xx_plot*1e6, max_xrms, color='C2', ls='--', label='Xmax')
#plt.loglog(xx_plot*1e6, mean_rms[1] / N, color='C1', ls='-', label='Y')
#plt.loglog(xx_plot*1e6, mean_rms[2] / N, color='C2', ls='-', label='Z')
plt.grid()
plt.legend()

outarr = [xx_plot, mean_rms[0], mean_rms[1], mean_rms[2], \
          std_rms[0], std_rms[1], std_rms[2]]
outarr = np.array(outarr)

np.save( open(base_path + '2um-1Vrms-patches_rms-force_vs_separation.npy', 'wb'), outarr)


#outarr2 = [xx_plot, finger_rms[0], finger_rms[1], finger_rms[2]]
#outarr2 = np.array(outarr2)

#np.save( open(base_path + 'bipolar-500mV-fingers_rms-force_vs_separation.npy', 'wb'), outarr2)



plt.show()
