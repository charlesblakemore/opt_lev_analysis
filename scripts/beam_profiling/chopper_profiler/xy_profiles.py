import os, fnmatch

import numpy as np

import scipy.optimize as opti

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import bead_util as bu
import configuration as config

import chopper_profiler_funcs as chopfuncs



fontsize = 16
plot_title = 'Centered'


#xfile = '/data/20171023/alignment/objective_coll/xprof_pos6.h5'
#yfile = '/data/20171023/alignment/objective_coll/yprof_pos6.h5'

#xfile = '/data/20171024/alignment/lens_tube/xprof_pos8.h5'
#yfile = '/data/20171024/alignment/lens_tube/yprof_pos8.h5'

xfile = '/data/20171025/chopper_profiling/xprof_output_centered.h5'
yfile = '/data/20171025/chopper_profiling/yprof_output_centered.h5'
#xfile = '/data/20171025/chopper_profiling/45degprof_output.h5'






#xfile = '/data/20171020/beam_profiling/xprof_init.h5'
#xfile = '/data/20171020/beam_profiling/xprof_noclamp.h5'
#xfile = '/data/20171020/beam_profiling/xprof_clamp_gel.h5'
#xfile = '/data/20171020/beam_profiling/xprof_mintape.h5'

#yfile = '/data/20171020/beam_profiling/yprof_init.h5'
#yfile = '/data/20171020/beam_profiling/yprof_noclamp.h5'
#yfile = '/data/20171020/beam_profiling/yprof_clamp_gel.h5'
#yfile = '/data/20171020/beam_profiling/yprof_mintape.h5'


xfilobj = bu.DataFile()
xfilobj.load(xfile)
xfilobj.load_other_data()

yfilobj = bu.DataFile()
yfilobj.load(yfile)
yfilobj.load_other_data()


xprof_dir = '/data/20171020/beam_profiling/many_xprofs_mintape'




x_d, x_prof, x_popt = chopfuncs.profile(xfilobj, raw_dat_col = 4, \
                                        return_pos = True, numbins = 500, \
                                        fit_intensity = True)
y_d, y_prof, y_popt = chopfuncs.profile(yfilobj, raw_dat_col = 4, \
                                        return_pos = True, numbins = 500, \
                                        fit_intensity = True)

x_prof = x_prof / x_popt[0]
y_prof = y_prof / y_popt[0]

x_popt[0] = 1.0
y_popt[0] = 1.0


binned_x_d, binned_x_prof, x_errs = chopfuncs.rebin(x_d, x_prof, numbins=500)
binned_y_d, binned_y_prof, y_errs = chopfuncs.rebin(y_d, y_prof, numbins=500)




print "X diam (2 * waist): ", x_popt[-1] * 1e3 * 2
print "Y diam (2 * waist): ", y_popt[-1] * 1e3 * 2

print 

print "X waist: ", x_popt[-1] * 1e3 
print "Y waist: ", y_popt[-1] * 1e3 

print

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

plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

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

plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)

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

plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0.5)

if plot_title:
    plt.suptitle(plot_title, fontsize=20)
    plt.subplots_adjust(top=0.9)

plt.show()





