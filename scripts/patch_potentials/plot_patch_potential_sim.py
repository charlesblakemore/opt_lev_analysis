import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.constants as constants

import sys, time, os

import bead_util as bu


fac = 500 * constants.elementary_charge

base_path = '/processed_data/comsol_data/patch_potentials/'


fnames = bu.find_all_fnames(base_path, ext='')
names = []

for filname in fnames:
    parts = filname.split('/')
    name = parts[-1].split('.')[0]
    if name not in names:
        names.append(name)


names = ['patch_pot_2um_1Vrms_50um-deep-patches' \
        ]

#print names


style_dict = {0: '-', 1: '--', 2: ':',}
color_dict = {0: 'C0', 1: 'C1', 2: 'C2',}
fudge_dict = {0: 1, 1: 2, 2: 5}
fudge_dict = {0: 0.2, 1: 0.5, 2: 1}


for nameind, name in enumerate(names):
    print 'Processing: ', name
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

    
    plt.figure()
    plt.plot(posvec*1e6, pot_func(eval_pts))

    plt.figure(figsize=(7,5))
    plt.title(name)
    plt.plot(posvec*1e6, field_func[0](eval_pts)*fac, label='fx')
    plt.plot(posvec*1e6, field_func[1](eval_pts)*fac, label='fy')
    plt.plot(posvec*1e6, field_func[2](eval_pts)*fac, label='fz')
    plt.legend()
    plt.xlabel('Displacement Along Attractor [um]')
    plt.ylabel('Force on 500e$^-$ [N]')
    plt.annotate(ann_str, xy=(0.2, 0.9), xycoords='axes fraction')
    plt.tight_layout()
    plt.grid()

    plt.show()
    

    xx_plot = xx[xx > 10.0e-6]

    #rms_force = [[], [], []]
    rms_force = []
    for sepind, sep in enumerate(xx_plot):
        rms_val = 0.0
        eval_pts = np.stack((sep*ones, posvec, zval*ones), axis=-1)
        for resp in [0,1,2]:
            forcevec = field_func[resp](eval_pts) * fac
            rms_val += np.std(forcevec)**2
            #rms_force[resp].append(rms_val)
        #rms_val *= 1.0 / np.sqrt(3)
        rms_val *= 1.0 / 3
        rms_force.append(np.sqrt(rms_val))
        
    rms_force = np.array(rms_force)
    #for resp in [0,1,2]:
    #    if resp == 0:
    #        plt.loglog(xx_plot*1e6, rms_force[resp]*fudge_dict[nameind], label=name, \
    #                   ls=style_dict[resp], color=color_dict[nameind])
    #    else:
    #        plt.loglog(xx_plot*1e6, rms_force[resp]*fudge_dict[nameind], \
    #                   ls=style_dict[resp], color=color_dict[nameind])
        
    Vstr = str(fudge_dict[nameind]) + 'Vrms'
    if nameind == 0:
        newname = name[:17] +  Vstr
    else:
        newname = name[:16] + Vstr

    plt.loglog(xx_plot*1e6, rms_force*fudge_dict[nameind], label=newname, color=color_dict[nameind])

    if name == names[2]:
        plt.xlabel('Separation [um]')
        plt.ylabel('RMS force [N]')
        plt.legend()
        plt.show()


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
