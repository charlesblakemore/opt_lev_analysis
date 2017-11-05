import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import Tkinter
import tkFileDialog
import os, sys
from scipy.optimize import curve_fit
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize 

dirs = [1,]
ddict = bu.load_dir_file( "/dirfiles/dir_file_july2017.txt" )
#print ddict

cant_axis = 2
step_axis = 1
bin_size = 1  # um
lpf = 150 # Hz

init_data = [0., 0., -40]
cal_drive_freq = 41.

maxfiles = 1000

fig_title = 'Force vs. Cantilever Position:'
xlab = 'Distance along Cantilever [um]'

tf_path = '/calibrations/transfer_funcs/Hout_20170718.p'
step_cal_path = '/calibrations/step_cals/step_cal_20170718.p'



#####################################################


def proc_dir(d):
    dv = ddict[d]

    init_data = [dv[0], [0,0,dv[-1]], dv[1]]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])
    dir_obj.load_dir(cu.simple_loader)
    
    return dir_obj

dir_objs = map(proc_dir, dirs)

pos_dict = {}
for obj in dir_objs:
    dirlabel = obj.label
    for fobj in obj.fobjs:
        cpos = fobj.get_stage_settings(axis=step_axis)[0]
        cpos = cpos * 80. / 10.   # 80um travel per 10V control
        if cpos not in pos_dict:
            pos_dict[cpos] = []
        pos_dict[cpos].append(fobj.fname)

colors = bu.get_color_map(len(pos_dict.keys()))

pos_keys = pos_dict.keys()
pos_keys.sort()


f, axarr = plt.subplots(3,2,sharex='all',sharey='all',figsize=(7,8),dpi=100)

for i, pos in enumerate(pos_keys):
    newobj = cu.Data_dir(0, init_data, pos)
    newobj.files = pos_dict[pos]
    newobj.load_dir(cu.diag_loader, maxfiles=maxfiles)

    
    newobj.load_H(tf_path)
    newobj.load_step_cal(step_cal_path)
    newobj.calibrate_H()

    newobj.filter_files_by_cantdrive(cant_axis=cant_axis, nharmonics=10, noise=True, width=1.)

    newobj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=lpf, #plot_Happ=True, \
                             build_conv_facs=True, drive_freq=cal_drive_freq, cantfilt=True,\
                             close_dat=False)

    newobj.get_avg_force_v_pos(cant_axis=cant_axis, bin_size = bin_size, cantfilt=True)
    newobj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = bin_size, cantfilt=True)



    keys = newobj.avg_diag_force_v_pos.keys()
    cal_facs = newobj.conv_facs
    color = colors[i]

    lab = "Filtered"

    for key in keys:
        diagdat = newobj.avg_diag_force_v_pos[key]
        dat = newobj.avg_force_v_pos[key]

        for resp in [0,1,2]:
            #offset = - dat[resp,0][1][-1]
            offset = - np.mean(dat[resp,0][1])
            #diagoffset = - diagdat[resp,0][1][-1]
            diagoffset = - np.mean(diagdat[resp,0][1])
            axarr[resp,0].errorbar(dat[resp,0][0], \
                                   (dat[resp,0][1]+offset)*cal_facs[resp]*1e15, \
                                   dat[resp,0][2]*cal_facs[resp]*1e15, \
                                   fmt='.-', ms=20, color = color, label=lab)
            axarr[resp,1].errorbar(diagdat[resp,0][0], \
                                   (diagdat[resp,0][1]+diagoffset)*1e15, \
                                   diagdat[resp,0][2]*1e15, \
                                   fmt='.-', ms=20, color = color, label=lab)




    newobj.get_avg_force_v_pos(cant_axis=cant_axis, bin_size = bin_size, cantfilt=False)
    newobj.get_avg_diag_force_v_pos(cant_axis = cant_axis, bin_size = bin_size, cantfilt=False)

    lab = "Un-Filtered"

    for key in keys:
        diagdat = newobj.avg_diag_force_v_pos[key]
        dat = newobj.avg_force_v_pos[key]

        for resp in [0,1,2]:
            #offset = - dat[resp,0][1][-1]
            offset = - np.mean(dat[resp,0][1])
            #diagoffset = - diagdat[resp,0][1][-1]
            diagoffset = - np.mean(diagdat[resp,0][1])

            axarr[resp,0].errorbar(dat[resp,0][0], \
                                   (dat[resp,0][1]+offset)*cal_facs[resp]*1e15, \
                                   dat[resp,0][2]*cal_facs[resp]*1e15, \
                                   fmt='.-', ms=10, color = 'r', label=lab)
            axarr[resp,1].errorbar(diagdat[resp,0][0], \
                                   (diagdat[resp,0][1]+diagoffset)*1e15, \
                                   diagdat[resp,0][2]*1e15, \
                                   fmt='.-', ms=10, color = 'r', label=lab)




axarr[0,0].set_title('Raw Imaging Response')
axarr[0,1].set_title('Diagonalized Forces')

for col in [0,1]:
    axarr[2,col].set_xlabel(xlab)

axarr[0,0].set_ylabel('X-direction Force [fN]')
axarr[1,0].set_ylabel('Y-direction Force [fN]')
axarr[2,0].set_ylabel('Z-direction Force [fN]')

axarr[0,1].legend(loc=0, numpoints=1, ncol=2, fontsize=9)

if len(fig_title):
    f.suptitle(fig_title + ' ' + dirlabel, fontsize=18)

plt.show()
