import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import cant_util as cu
import grav_util as gu
import scipy.signal as signal
import scipy.optimize as optimize 

show_final_stitching = True


### Load theoretical modified gravity curves

gpath = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/'
filname = 'attractorv2_1um_manysep_fullthrow_force_curves.p'

fcurve_obj = gu.Grav_force_curve(path=gpath+filname, make_splines=True, spline_order=3)

SEP = 5.0e-6
ALPHA = 1.e8
YUKLAMBDA = 10.0e-6
RBEAD = 2.43e-6



limitdata_path = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160928_datathief_nodecca2.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160914_datathief.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'




init_data = [0,0,0]

derp_ind = 0
minval = -170.
stepval = 40.



### Load data directories

dirs = [2,3,4,5,6,7,8,9]

ddict = bu.load_dir_file( "/dirfiles/dir_file_aug2017.txt" )
maxfiles = 3   # Maximum number of files to load from a directory

SWEEP_AX = 2     # Cantilever sweep axis, 1 for Y, 2 for Z
RESP_AX = 1      # Response Axis
bin_size = 1     # um, Binning for final force v. pos

lpf = 150        # Hz, acausal top-hat filter at this freq
cantfilt = True #True  # Notch filter at cantilever drive

fig_title = 'Force vs. Cantilever Position:'
xlab = 'Distance along Cantilever [um]'

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

    #for obj in dir_obj.fobjs:
    #    print obj.pos_data

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




### Stitch together final data and compare to modified gravity

final_dat = cu.Force_v_pos()
final_dat.load_dir_objs(dir_objs)
final_dat.stitch_data(numbins=200, cant_axis=SWEEP_AX, resp_axis=RESP_AX, showstitch=show_final_stitching, \
                      matchmeans=True, detrend=True, minmsq=True)



grav_dat = fcurve_obj.mod_grav_force(final_dat.bins*1e-6, sep=SEP, \
                                     alpha=ALPHA, yuklambda=YUKLAMBDA, rbead=RBEAD)
diag_grav_dat = fcurve_obj.mod_grav_force(final_dat.diagbins*1e-6, sep=SEP, \
                                     alpha=ALPHA, yuklambda=YUKLAMBDA, rbead=RBEAD)

grav_dat = signal.detrend(grav_dat)
diag_grav_dat = signal.detrend(diag_grav_dat)

wvnum = np.fft.rfftfreq(len(final_dat.force), d=(final_dat.bins[1]-final_dat.bins[0]))
datfft = np.fft.rfft(final_dat.force)
datasd = np.sqrt(datfft.conj() * datfft)
gravfft = np.fft.rfft(grav_dat)

diagwvnum = np.fft.rfftfreq(len(final_dat.diagforce), d=(final_dat.diagbins[1]-final_dat.diagbins[0]))
diagdatfft = np.fft.rfft(final_dat.diagforce)
diagdatasd = np.sqrt(diagdatfft.conj() * diagdatfft)
diaggravfft = np.fft.rfft(diag_grav_dat)




lambdas = fcurve_obj.lambdas
alphas = np.zeros_like(lambdas)
diagalphas = np.zeros_like(lambdas)

for ind, yuklambda in enumerate(lambdas):
    fcurve = fcurve_obj.mod_grav_force(final_dat.bins*1e-6, sep=SEP, alpha=1., \
                                       yuklambda=yuklambda, rbead=RBEAD, nograv=True)
    diagfcurve = fcurve_obj.mod_grav_force(final_dat.diagbins*1e-6, sep=SEP, alpha=1., \
                                           yuklambda=yuklambda, rbead=RBEAD, nograv=True)

    fcurve = signal.detrend(fcurve)
    diagfcurve = signal.detrend(diagfcurve)

    fft = np.fft.rfft(fcurve)
    asd = np.sqrt(fft.conj() * fft)
    diagfft = np.fft.rfft(diagfcurve)
    diagasd = np.sqrt(diagfft.conj() * diagfft)

    maxind = np.argmax(asd)
    diagmaxind = np.argmax(diagasd)

    alpha = datasd[maxind] / asd[maxind]
    diagalpha = diagdatasd[diagmaxind] / diagasd[maxind]

    alphas[ind] = np.abs(alpha)
    diagalphas[ind] = np.abs(diagalpha)

    







f, axarr = plt.subplots(1,2,sharex='all',sharey='all',figsize=(10,5),dpi=100)

#print final_dat.bins
#print final_dat.force

axarr[0].errorbar(final_dat.bins, final_dat.force*1e15, final_dat.errs*1e15, \
               fmt='.-', ms=10, color = 'r') #, label=lab)
axarr[0].plot(final_dat.bins, grav_dat*1e15, '-', color='k')

axarr[0].set_ylabel('Force [fN]')
axarr[0].set_xlabel('Position along cantilever [um]')

axarr[1].errorbar(final_dat.diagbins, final_dat.diagforce*1e15, final_dat.diagerrs*1e15, \
               fmt='.-', ms=10, color = 'r') #, label=lab)
axarr[1].plot(final_dat.diagbins, diag_grav_dat*1e15, '-', color='k')

axarr[1].set_xlabel('Position along cantilever [um]')






f2, axarr2 = plt.subplots(1,2,sharex='all',sharey='all',figsize=(10,5),dpi=100)

axarr2[0].loglog(wvnum, np.sqrt(datfft.conj() * datfft) )
axarr2[0].loglog(wvnum, np.sqrt(gravfft.conj() * gravfft) )

axarr2[0].set_ylabel('ASD [N / rt(um$^{-1}$)]')
axarr2[0].set_xlabel('Wavenumber [um$^{-1}$]')

axarr2[1].loglog(diagwvnum, np.sqrt(diagdatfft.conj() * diagdatfft) )
axarr2[1].loglog(diagwvnum, np.sqrt(diaggravfft.conj() * diaggravfft) )

axarr2[1].set_xlabel('Wavenumber [um$^{-1}$]')






f3, axarr3 = plt.subplots(1,2,sharex='all',sharey='all',figsize=(10,5),dpi=100)

axarr3[0].loglog(lambdas, alphas, label='Raw Data: Sig = Noise')
axarr3[0].loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
axarr3[0].loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')

axarr3[0].set_ylabel('alpha [arb]')
axarr3[0].set_xlabel('lambda [m]')

axarr3[1].loglog(lambdas, diagalphas, label='Diagonalized Data: Sig = Noise')
axarr3[1].loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
axarr3[1].loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')

axarr3[1].set_xlabel('lambda [m]')

axarr3[0].legend(numpoints=1)
axarr3[1].legend(numpoints=1)






plt.show()
