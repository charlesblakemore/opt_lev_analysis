import os, fnmatch

import numpy as np

import scipy.optimize as opti

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

import chopper_profiler_funcs as chopfuncs




#### SETUP FOR DATA SAMPLED AT 50 kHz

fontsize = 16
plot_title = 'Obj. Output - 92cm'
plot_title = ''


'''
xfile = '/daq2/20190308/profiling/epoxyed_fiber_3/10cm_xprof-1_no-light.h5'
yfile = '/daq2/20190308/profiling/epoxyed_fiber_3/10cm_xprof-2_no-light.h5'

xfilobj = bu.DataFile()
xfilobj.load(xfile, skip_fpga=True)
xfilobj.load_other_data()

yfilobj = bu.DataFile()
yfilobj.load(yfile, skip_fpga=True)
yfilobj.load_other_data()
'''


# xprof_dir = '/daq2/20190311/profiling/tele_out_coll/xprof_150cm_init2'
# yprof_dir = '/daq2/20190311/profiling/tele_out_coll/yprof_150cm_init2'

xprof_dir = '/data/old_trap/20171024/alignment/objective_coll_x_dummy'
yprof_dir = '/data/old_trap/20171024/alignment/objective_coll_y_dummy'

# xprof_dir = '/data/old_trap/20171025/chopper_profiling/xprof_output'
# yprof_dir = '/data/old_trap/20171025/chopper_profiling/yprof_output'
raw_dat_col=7

# xprof_dir = '/data/old_trap/20201204/chopper_profiling/no_telescope_near_xprof'
# yprof_dir = '/data/old_trap/20201204/chopper_profiling/no_telescope_near_yprof'


# plot_raw_dat = True
plot_raw_dat = False
# plot_result = True
plot_result = False
# plot_rebin = True
plot_rebin = False




x_d, x_prof, x_popt = chopfuncs.profile_directory(\
                            xprof_dir, raw_dat_col=raw_dat_col, \
                            plot_peaks=False, return_pos=True, \
                            guess=3.0e-3, plot_raw_dat=plot_raw_dat, \
                            plot_result=plot_result)
y_d, y_prof, y_popt = chopfuncs.profile_directory(\
                            yprof_dir, raw_dat_col=raw_dat_col, \
                            plot_peaks=False, return_pos=True, \
                            guess=3.0e-3, plot_raw_dat=plot_raw_dat, \
                            plot_result=plot_result)


#x_d, x_prof, x_popt = chopfuncs.profile(xfilobj, raw_dat_col = 0, \
#                                        return_pos = True, numbins = 500, \
#                                        fit_intensity = True, plot_peaks=False)
#y_d, y_prof, y_popt = chopfuncs.profile(yfilobj, raw_dat_col = 0, \
#                                        return_pos = True, numbins = 500, \
#                                        fit_intensity = True, plot_peaks=False)



x_prof = x_prof / x_popt[0]
y_prof = y_prof / y_popt[0]

x_popt[0] = 1.0
y_popt[0] = 1.0


binned_x_d, binned_x_prof, x_errs = \
    bu.rebin(x_d, x_prof, nbin=1000, plot=plot_rebin)
binned_y_d, binned_y_prof, y_errs = \
    bu.rebin(y_d, y_prof, nbin=1000, plot=plot_rebin)




print("X diam (2 * waist): ", x_popt[-1] * 1e3 * 2)
print("Y diam (2 * waist): ", y_popt[-1] * 1e3 * 2)

print() 

print("X waist: ", x_popt[-1] * 1e3) 
print("Y waist: ", y_popt[-1] * 1e3) 

print()

#LP_p0 = [1, 0.001]
#final_x_LP_popt, final_x_LP_pcov = \
#        opti.curve_fit(chopfuncs.bessel_intensity, \
#                       binned_x_d, binned_x_prof, p0=LP_p0)





fig1, axarr1 = plt.subplots(2,1,sharex=True,sharey=True,figsize=(8,6),dpi=100)
axarr1[0].plot(x_d * 1e3, x_prof, label="All Data")
axarr1[0].plot(binned_x_d * 1e3, binned_x_prof, label="Avg'd Data", color='k')
axarr1[0].plot(binned_x_d * 1e3, \
               chopfuncs.gauss_intensity(binned_x_d, *x_popt), \
               '--', color = 'r', linewidth=1.5, label="Gaussian Fit")
axarr1[0].set_xlabel("Displacement [mm]", fontsize=fontsize)
axarr1[0].set_ylabel("X Intensity [arb]", fontsize=fontsize)
axarr1[0].legend(fontsize=fontsize-4, loc=1)
plt.setp(axarr1[0].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr1[0].get_yticklabels(), fontsize=fontsize, visible=True)

axarr1[1].plot(y_d * 1e3, y_prof)
axarr1[1].plot(binned_y_d * 1e3, binned_y_prof, color='k')
axarr1[1].plot(binned_y_d * 1e3, \
               chopfuncs.gauss_intensity(binned_y_d, *y_popt), \
               '--', color = 'r', linewidth=1.5)
axarr1[1].set_xlabel("Displacement [mm]", fontsize=fontsize)
axarr1[1].set_ylabel("Y Intensity [arb]", fontsize=fontsize)
plt.setp(axarr1[1].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr1[1].get_yticklabels(), fontsize=fontsize, visible=True)

fig1.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

if plot_title:
    plt.suptitle(plot_title, fontsize=20)
    plt.subplots_adjust(top=0.9)





fig2, axarr2 = plt.subplots(2,1,sharex=True,sharey=True,figsize=(8,6),dpi=100)
axarr2[0].semilogy(binned_x_d * 1e3, np.abs(binned_x_prof), \
                   label="Avg'd Data", color='k')
axarr2[0].semilogy(binned_x_d * 1e3, \
                   chopfuncs.gauss_intensity(binned_x_d, *x_popt),\
                   '--', color = 'r', linewidth=1.5, label="Gaussian Fit")
#axarr2[0].semilogy(binned_x_d * 1e3, LP01_mode(binned_x_d, *final_x_LP_popt),\
#                   '--', color = 'g', linewidth=1.5, label="LP Fit")
axarr2[0].set_xlabel("Displacement [mm]", fontsize=fontsize)
axarr2[0].set_ylabel("X Intensity [arb]", fontsize=fontsize)
axarr2[0].set_ylim(1e-4,3)
axarr2[0].legend(fontsize=fontsize-4, loc=1)
plt.setp(axarr2[0].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr2[0].get_yticklabels(), fontsize=fontsize, visible=True)

axarr2[1].semilogy(binned_y_d * 1e3, np.abs(binned_y_prof), color='k')
axarr2[1].semilogy(binned_y_d * 1e3, \
                   chopfuncs.gauss_intensity(binned_y_d, *y_popt),\
                   '--', color = 'r', linewidth=1.5)
axarr2[1].set_xlabel("Displacement [mm]", fontsize=fontsize)
axarr2[1].set_ylabel("Y Intensity [arb]", fontsize=fontsize)
axarr2[0].set_ylim(1e-4,3)
plt.setp(axarr2[1].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr2[1].get_yticklabels(), fontsize=fontsize, visible=True)

fig2.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

if plot_title:
    plt.suptitle(plot_title, fontsize=20)
    plt.subplots_adjust(top=0.9)




fig3, axarr3 = plt.subplots(2,1,sharex=True,sharey=True,figsize=(8,6),dpi=100)
axarr3[0].plot(binned_x_d * 1e3, binned_x_prof - \
                   chopfuncs.gauss_intensity(binned_x_d, *x_popt), \
                   label="Residuals", color='k')
axarr3[0].set_xlabel("Displacement [mm]", fontsize=fontsize)
axarr3[0].set_ylabel("X Intensity [arb]", fontsize=fontsize)
axarr3[0].legend(fontsize=fontsize-4, loc=1)
plt.setp(axarr3[0].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr3[0].get_yticklabels(), fontsize=fontsize, visible=True)

axarr3[1].plot(binned_y_d * 1e3, binned_y_prof - \
               chopfuncs.gauss_intensity(binned_y_d, *y_popt), color='k')
axarr3[1].set_xlabel("Displacement [mm]", fontsize=fontsize)
axarr3[1].set_ylabel("Y Intensity [arb]", fontsize=fontsize)
plt.setp(axarr3[1].get_xticklabels(), fontsize=fontsize, visible=True)
plt.setp(axarr3[1].get_yticklabels(), fontsize=fontsize, visible=True)

fig3.tight_layout(pad=1.0, w_pad=1.0, h_pad=0.5)

if plot_title:
    plt.suptitle(plot_title, fontsize=20)
    plt.subplots_adjust(top=0.9)

plt.show()





