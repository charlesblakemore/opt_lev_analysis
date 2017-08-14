import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import cant_util as cu
import grav_util as gu
import scipy.signal as signal
import scipy.optimize as optimize 

show_final_stitching = True
matchmeans = False
minmsq = True
detrend = True

init_data = [0,0,0]

derp_ind = 0
minval = -170.
stepval = 40.


save = True #False
savepath = '/force_v_pos/20170718_grav_background.p'

plot = True


### Load data directories

dirs = [2,3,4,5,6,7,8,9]

ddict = bu.load_dir_file( "/dirfiles/dir_file_aug2017.txt" )
maxfiles = 1   # Maximum number of files to load from a directory

SWEEP_AX = 2     # Cantilever sweep axis, 1 for Y, 2 for Z
RESP_AX = 1      # Response Axis
bin_size = 1     # um, Binning for final force v. pos

lpf = 150        # Hz, acausal top-hat filter at this freq
cantfilt = False # Notch filter at cantilever drive

# Locate Calibration files
tf_path = '/calibrations/transfer_funcs/Hout_20170707.p'
step_cal_path = '/calibrations/step_cals/step_cal_20170707.p'
cal_drive_freq = 41.




### Process individual directories


def proc_dir(d):
    # simple directory processing function to load data and find
    # different cantilever positions
    global derp_ind, minval, stepval

    dv = ddict[d]
    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-1]], dv[1])

    newfils = []
    for fil in dir_obj.files:
        if 'Y20um' in fil:
            newfils.append(fil)
    dir_obj.files = newfils

    dir_obj.load_dir(cu.diag_loader, maxfiles=maxfiles)

    # Load the calibrations
    dir_obj.load_H(tf_path)
    dir_obj.load_step_cal(step_cal_path)
    dir_obj.calibrate_H()
    
    if cantfilt:
        dir_obj.filter_files_by_cantdrive(cant_axis=SWEEP_AX, nharmonics=10, noise=True, width=1.)

    dir_obj.diagonalize_files(reconstruct_lowf=True,lowf_thresh=lpf,  #plot_Happ=True, \
                              build_conv_facs=True, drive_freq=cal_drive_freq, cantfilt=cantfilt)

    dir_obj.get_avg_force_v_pos(cant_axis=SWEEP_AX, bin_size = bin_size, cantfilt=cantfilt)
    dir_obj.get_avg_force_v_pos(cant_axis=SWEEP_AX, bin_size = bin_size, diag=True, cantfilt=cantfilt)

    dir_obj.cant_corner_img = (0, 0, minval + stepval * derp_ind)
    print dir_obj.cant_corner_img
    derp_ind += 1

    return dir_obj


# Do processing
dir_objs = map(proc_dir, dirs)




### Stitch together final data and save

final_dat = cu.Force_v_pos()
final_dat.load_dir_objs(dir_objs)
final_dat.stitch_data(bin_size=1.2*bin_size, cant_axis=SWEEP_AX, resp_axis=RESP_AX, showstitch=show_final_stitching, \
                      matchmeans=matchmeans, detrend=detrend, minmsq=minmsq)

if save:
    final_dat.save(savepath)


if plot:
    f, axarr = plt.subplots(1,2,sharex='all',sharey='all',figsize=(10,5),dpi=100)

    axarr[0].errorbar(final_dat.bins, final_dat.force*1e15, final_dat.errs*1e15, \
                   fmt='.-', ms=10, color = 'r') #, label=lab)

    axarr[0].set_ylabel('Force [fN]')
    axarr[0].set_xlabel('Position along cantilever [um]')

    axarr[1].errorbar(final_dat.diagbins, final_dat.diagforce*1e15, final_dat.diagerrs*1e15, \
                   fmt='.-', ms=10, color = 'r') #, label=lab)

    axarr[1].set_xlabel('Position along cantilever [um]')

    plt.show()



