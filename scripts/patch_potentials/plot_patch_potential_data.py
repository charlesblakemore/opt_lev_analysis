import time, sys, os
import dill as pickle

import numpy as np
import scipy.constants as constants
import scipy.interpolate as interp
import scipy.optimize as opti

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import (FormatStrFormatter, MultipleLocator)

import grav_util_3 as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")




theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'

patches_base_path = '/processed_data/comsol_data/patch_potentials/'
patches_name = 'patch_pot_2um_1Vrms_150um-deep-patches'


rms_curve_path = patches_base_path + '2um-1Vrms-patches_rms-force_vs_separation.npy'
rms_curves = np.load( open(rms_curve_path, 'rb') )
rms_curve_funcs = []
rms_std_curve_funcs = []
for resp in [0,1,2]:
    rms_curve_funcs.append( interp.interp1d(rms_curves[0], rms_curves[resp+1]) )
    rms_std_curve_funcs.append( interp.interp1d(rms_curves[0], rms_curves[resp+4]) )


finger_rms_curve_path = patches_base_path + 'bipolar-500mV-fingers_rms-force_vs_separation.npy'
finger_rms_curves = np.load( open(finger_rms_curve_path, 'rb') )
finger_rms_curve_funcs = []
for resp in [0,1,2]:
    finger_rms_curve_funcs.append( interp.interp1d(finger_rms_curves[0], finger_rms_curves[resp+1]) )

#plt.loglog(rms_curves[0], rms_curves[1])
#plt.loglog(finger_rms_curves[0], finger_rms_curves[1])
#plt.show()


# Include some legacy grav data to compare to later
data_dirs = [#'/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz', \
             #\
             #'/data/20180704/bead1/grav_data/shield', \
             #\
             #'/data/20180808/bead4/grav_data/shield1' \
             #\
             #'/data/20180827/bead2/500e_data/dipole_v_height_ac', \
             '/data/20180827/bead2/500e_data/dipole_v_height_no_v_ysweep', \
             #'/data/20180927/bead1/electric_measurements_for_chas'
             ]

load_files = False
save_plots = True
annotate_fit = False

tfdate = ''
#tfdate = '20180827'

p0_bead_dict = {'20180625': [19.0, 40.0, 20.0], \
                '20180704': [18.7, 40.0, 20.0], \
                '20180808': [18.0, 40.0, 20.0], \
                '20180827': [55.0 ,40.0, 28.6]
                }

p0_bead_dict = {'20180625': [0.0, 0.0,30.0], \
                '20180704': [0.0, 0.0, 0.0], \
                '20180808': [0.0, 0.0, 0.0], \
                '20180827': [12.0, 40.0, 27.0], \
                '20180927': [19.5, 40.0, 27.0]
                }

opt_ext = '_electrostatics'

figpath = '/home/charles/plots/20180829/20180827_rms-force_vs_sep_fit.pdf'
figpath2 = '/home/charles/plots/20180829/20180827_rms-force_vs_sep_fit.png'

harms = [1,2,3,4,5,6,7,8,9]
#harms = [1,2,3,4,5]

charge = 425 * constants.elementary_charge * (-1.0)
plot_field_test = False

############################################################
xx = np.load(open(patches_base_path + patches_name + '.xx', 'rb'))
yy = np.load(open(patches_base_path + patches_name + '.yy', 'rb'))
zz = np.load(open(patches_base_path + patches_name + '.zz', 'rb'))

field = np.load(open(patches_base_path + patches_name + '.field', 'rb'))
potential = np.load(open(patches_base_path + patches_name + '.potential', 'rb')) 


pot_func = interp.RegularGridInterpolator((xx, yy, zz), potential, \
                                          bounds_error=False, fill_value=None)

field_func = []
for resp in 0,1,2:
    field_func.append( interp.RegularGridInterpolator((xx, yy, zz), field[resp], \
                                          bounds_error=False, fill_value=None) )


if plot_field_test:
    posvec = np.linspace(-40e-6, 40e-6, 101)
    ones = np.ones_like(posvec)
    xval = 20.0e-6
    yval = 0.0e-6
    zval = 0.0e-6
    eval_pts = np.stack((xval*ones, posvec, zval*ones), axis=-1)
    #eval_pts = np.stack((xval*ones, yval*ones, posvec), axis=-1)

    ann_str = r'Sep: %0.2f um, Height: %0.2f um' % (xval*1e6, zval*1e6)

    
    plt.figure()
    plt.plot(posvec*1e6, pot_func(eval_pts))

    plt.figure(figsize=(7,5))
    #plt.title(name)
    plt.plot(posvec*1e6, field_func[0](eval_pts)*charge, label='fx')
    plt.plot(posvec*1e6, field_func[1](eval_pts)*charge, label='fy')
    plt.plot(posvec*1e6, field_func[2](eval_pts)*charge, label='fz')
    plt.legend()
    plt.xlabel(r'Displacement Along Attractor [$\mu$m]')
    plt.ylabel(r'Force on 500e$^-$ [N]')
    plt.annotate(ann_str, xy=(0.2, 0.9), xycoords='axes fraction')
    plt.tight_layout()
    plt.grid()

    plt.show()
    





for ddir in data_dirs:

    paths = gu.build_paths(ddir, opt_ext=opt_ext)

    datafiles = bu.find_all_fnames(ddir)
    p0_bead = p0_bead_dict[paths['date']]

    if load_files:
        agg_dat = gu.AggregateData(datafiles, p0_bead=p0_bead, harms=harms, \
                                   elec_drive=False, elec_ind=0, plot_harm_extraction=False, \
                                   tfdate=tfdate)       

        agg_dat.save(paths['agg_path'])

    else:
        agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms)
        agg_dat.load(paths['agg_path'])
        agg_dat.p0_bead = p0_bead

    agg_dat.bin_rough_stage_positions(ax_disc=1.0)
    #agg_dat.plot_force_plane(resp=0, fig_ind=1, show=False)
    #agg_dat.plot_force_plane(resp=1, fig_ind=2, show=False)
    #agg_dat.plot_force_plane(resp=2, fig_ind=3, show=False)

    #force_plane_dict = agg_dat.get_vector_force_plane(resp=(0,2), fig_ind=1, \
    #                                                  plot=True, show=True, \
    #                                                  sign=[-1.0, -1.0, -1.0])


    if True: #'chas' in ddir:
        rms_grid = np.zeros((3, len(agg_dat.ax0vec), len(agg_dat.ax1vec)))
        err_grid = np.zeros((3, len(agg_dat.ax0vec), len(agg_dat.ax1vec)))

        N_grid = np.zeros_like(rms_grid[0])
        for ax0ind, ax0 in enumerate(agg_dat.ax0vec):
            for ax1ind, ax1 in enumerate(agg_dat.ax1vec):
                fd_objs = agg_dat.agg_dict[agg_dat.biasvec[0]][ax0][ax1]
                for fd_obj in fd_objs:
                    N_grid[ax0ind,ax1ind] += 1
                    for resp in [0,1,2]:
                        rms_grid[resp,ax0ind,ax1ind] += np.std(fd_obj.binned[resp][1])
                        err_grid[resp,ax0ind,ax1ind] += np.mean(fd_obj.binned[resp][2])

        for resp in [0,1,2]:
            rms_grid[resp] *= 1.0 / N_grid
            err_grid[resp] *= 1.0 / N_grid

        pos_dict = agg_dat.make_ax_arrs()
        seps = pos_dict['seps']
        print(agg_dat.p0_bead)
        print(seps)
        sep_sort = pos_dict['sep_sort']
        heights = pos_dict['heights']
        height_sort = pos_dict['height_sort']

        seps_g, heights_g = np.meshgrid(seps, heights, indexing='ij')
    
        for resp in [0,1,2]:
            rms_grid[resp][:,:] = rms_grid[resp][sep_sort,:]
            rms_grid[resp][:,:] = rms_grid[resp][:,height_sort]
            err_grid[resp][:,:] = err_grid[resp][sep_sort,:]
            err_grid[resp][:,:] = err_grid[resp][:,height_sort]

        height_ind = np.argmin(np.abs(heights - 0.0))
        
        xvec = seps
        yvec = rms_grid[0,:,height_ind] * 1e15
    
    
        comsol_rms = []
        for resp in [0,1,2]:
            comsol_rms.append(rms_curve_funcs[resp](seps*1e-6))
    
        const_scale = 1.0e-14
    
        def cost_finger(params, axes=[0]):
            finger_amp, const = params
            N = 0
            cost = 0.0
            for resp in [0,1,2]:
                if resp not in axes:
                    continue
                func_vals = finger_amp * finger_rms_curve_funcs[resp]((seps)*1e-6) + \
                            const * const_scale
                #plt.loglog(func_vals*0.05)
                #plt.show()
                diffsq = (rms_grid[resp,:,height_ind] - func_vals)**2
                var = (rms_grid[resp,:,height_ind])**2
                var = np.ones_like(diffsq) * np.mean(var)
                cost += np.sum( diffsq / var )
                N += diffsq.size
            return 0.5 * cost
    
    
        def cost_patch(params, axes=[0]):
            patch_amp, const = params
            N = 0
            cost = 0.0
            for resp in [0,1,2]:
                if resp not in axes:
                    continue
                func_vals = patch_amp * rms_curve_funcs[resp]((seps)*1e-6) + \
                            const * const_scale
                #plt.loglog(func_vals*0.05)
                #plt.show()
                diffsq = (rms_grid[resp,:,height_ind] - func_vals)**2
                var = (rms_grid[resp,:,height_ind])**2
                var = np.ones_like(diffsq) * np.mean(var)
                cost += np.sum( diffsq / var )
                N += diffsq.size
            return 0.5 * cost
    
    
        def cost_both(params, axes=[0]):
            patch_amp, finger_amp, const = params
            N = 0
            cost = 0.0
            for resp in [0,1,2]:
                if resp not in axes:
                    continue
                func_vals = patch_amp * rms_curve_funcs[resp]((seps)*1e-6) + \
                            finger_amp * finger_rms_curve_funcs[resp]((seps)*1e-6) + \
                            const * const_scale
                #plt.loglog(func_vals*0.05)
                #plt.show()
                diffsq = (rms_grid[resp,:,height_ind] - func_vals)**2
                var = (rms_grid[resp,:,height_ind])**2
               # var = np.ones_like(diffsq) * np.mean(var)
                cost += np.sum( diffsq / var )
                N += diffsq.size
            return 0.5 * cost
    

        init = [1.0, 1.0, 0.0]
        patch_init = [1.0, 0.0]
        print(cost_both(init))
        res_both_x = opti.minimize(cost_both, init, method='L-BFGS-B', \
                                   bounds=((0, None), (0, None), (0, None)), args=([0]) )
        res_both_y = opti.minimize(cost_both, init, method='L-BFGS-B', \
                                   bounds=((0, None), (0, None), (0, None)), args=([1]) )
                                   
        res_patch_x = opti.minimize(cost_patch, patch_init, method='L-BFGS-B', \
                                    bounds=((0, None), (0, None)), args=([0]) )
        res_patch_y = opti.minimize(cost_patch, patch_init, method='L-BFGS-B', \
                                    bounds=((0, None), (0, None)), args=([1]) )


        
        #res_finger = opti.minimize(cost_finger, init[0], args=([0]) )
        #x_finger = res_finger.x

        #print x_both


        #x = res_patch_x.x
        x = res_both_x.x[0]
        xconst = res_both_x.x[-1] * const_scale
        #y = res_patch_y.x
        y = res_both_y.x[0]
        yconst = res_both_y.x[-1] * const_scale

        print("RESULT: ", x, y)


        print(res_both_x.x)
        print(res_both_y.x)


        #Hfun = ndt.Hessian(cost_function, full_output=True)
        #hessian_ndt, info = Hfun(x)
        #pcov = 4.0 * np.linalg.inv(hessian_ndt)



        def fitfun(x, A, pow, const):
            return A / (x**pow) + const

        def fitfunx(x, A, pow):
            return A / (x**pow) + xconst * 1e15
        
        xvec = seps
        yvec_x = rms_grid[0,:,height_ind] * 1e15
        yvec_y = rms_grid[1,:,height_ind] * 1e15

        def cost_pow(params, axes=[0]):
            A, pow, const = params
            A *= 1e-15
            const *= 1e-15
            N = 0
            cost = 0.0
            penalty = 0
            for resp in [0,1,2]:
                if resp not in axes:
                    continue
                #if resp == 0:
                #    const = xconst
                func_vals = fitfun(xvec, A, pow, const)
                diffsq = (rms_grid[resp,:,height_ind] - func_vals)**2
                var = (rms_grid[resp,:,height_ind])**2
                #var = (np.ones_like(rms_grid[resp,:,height_ind]))*1e-28
                cost += np.sum( diffsq / var )
                N += diffsq.size
                #if resp == 0:
                #    penalty += 1e14 * (const - xconst)**2 / xconst
            return 0.5 * cost + penalty


        init = [1, 2, 0]
        res_pow_x = opti.minimize(cost_pow, init, method='L-BFGS-B', \
                                  bounds=((0, None), (0, None), (0, None)), args=([0]) )
        res_pow_y = opti.minimize(cost_pow, init, method='L-BFGS-B', \
                                  bounds=((0, None), (0, None), (0, None)), args=([1]) )


        print(cost_pow(init, axes=[0]))
        input()

        #xvec = seps
        #yvec = rms_grid[0,:,height_ind] * 1e15
        #poptx, pcovx = opti.curve_fit(fitfunx, xvec, yvec, p0=[1, 2])

        #yvec = rms_grid[1,:,height_ind] * 1e15
        #popty, pcovy = opti.curve_fit(fitfun, xvec, yvec, p0=[1, 2, 0.0])
    
        #print poptx
        #print popty

        #print poptx[-1] * 1e-15, popty[-1] * 1e-15
    
        xerr = 2.5
        yerrx = err_grid[0,:,height_ind]
        yerry = err_grid[1,:,height_ind]
    
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111)
    
        p1x = ax.errorbar(seps, rms_grid[0,:,height_ind], yerrx, xerr, ms=5, fmt='o', \
                          label=r'Data $F_{x,{\rm rms}}$', color='C0')
        p1y = ax.errorbar(seps, rms_grid[1,:,height_ind], yerry, xerr, ms=6, fmt='^', \
                          label=r'Data $F_{y,{\rm rms}}$', color='C1')
        
        func_vals_x = x * rms_curve_funcs[0]((seps)*1e-6) + xconst 
        func_std_x = x * rms_std_curve_funcs[0]((seps)*1e-6)

        func_vals_y = y * rms_curve_funcs[1]((seps)*1e-6) + yconst 
        func_std_y = y * rms_std_curve_funcs[1]((seps)*1e-6)


        p2x = ax.loglog(seps, func_vals_x, \
                   label=r'FEA Fit: $F_{x, \rm{rms}}$', color='k', ls=':')
        ax.fill_between(seps, func_vals_x+func_std_x, func_vals_x-func_std_x, \
                        color='k', alpha=0.2)

        p2y = ax.loglog(seps, func_vals_y, \
                   label=r'FEA Fit: $F_{y, \rm{rms}}$', color='r', ls=':')
        ax.fill_between(seps, func_vals_y+func_std_y, func_vals_y-func_std_y, \
                        color='r', alpha=0.2)

        
        #x_powlaw = []
        #y_powlaw = []
        #for sep in seps:
        #    x_powlaw.append(fitfun(sep, res_pow_x.x[0], res_pow_x[1], res_pow_x[2]))
        #    y_powlaw.append(fitfun(sep, res_pow_y.x[0], res_pow_y[1], res_pow_y[2]))
    
        #p4x = ax.loglog(seps, fitfun(seps, *res_pow_x.x)*1e-15,# + xconst, \
        #                color='k', ls='--', lw=1.5, \
        #                label=r'$F_{x,{\rm rms}} \propto X^{-%0.1f}$' % res_pow_x.x[1])
    
        #p4y = ax.loglog(seps, fitfun(seps, *res_pow_y.x)*1e-15, \
        #                color='r', ls='--', lw=1.5, \
        #                label=r'$F_{y,{\rm rms}} \propto X^{-%0.1f}$' % res_pow_y.x[1])
    
        ax.set_xlabel(r'$x$ [$\mu$m]', fontsize=16)
        #ax.set_ylabel(r'$\sqrt{\langle F_x^2 \rangle}$ $(-40~\mu $m$ < Y < 40~\mu $m)  [N]')
        ax.set_ylabel(r'$F_{\rm rms} \,\,\, (-40 \, \mu {\rm m} < y < 40 \, \mu {\rm m})$  [N]', \
                      fontsize=16)
    
        handles, labels = ax.get_legend_handles_labels()
        newhandles = handles[2:] + handles[0:2]
        newlabels = labels[2:] + labels[0:2]
        #ax.legend(handles[::-1], labels[::-1] )
        ax.legend(newhandles, newlabels, loc=3, fontsize=12)
        
        #x = res_pow_x.x[0]
        #y = res_pow_y.x[0]

        ann_str_x = r'Xfit: $V_{\rm patch,rms}$~100 mV$_{\rm rms}$ for $l_{\rm patch}$~%0.1f $\mu$m' % (x * 20.0)
        ann_str_y = r'Yfit:               ~100 mV$_{\rm rms}$              ~%0.1f $\mu$m' % (y * 20.0)

        print("X Size: %0.1f" % (x*20.0))
        print("Y Size: %0.1f" % (y*20.0))
        
        if annotate_fit:
            ax.annotate(ann_str_x, xy=(0.29, 0.9), xycoords='axes fraction')
            ax.annotate(ann_str_y, xy=(0.29, 0.85), xycoords='axes fraction')

        plt.setp(ax.get_xticklabels(which='minor'), fontsize=16)
        plt.setp(ax.get_xticklabels(which='major'), fontsize=0)

        minorLocator = MultipleLocator(20)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%i'))

        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_minor_formatter(FormatStrFormatter('%i'))

        plt.setp(ax.get_yticklabels(), fontsize=16)

        plt.tight_layout()

        if save_plots:
            fig.savefig(figpath, dpi='figure')
            fig.savefig(figpath2, dpi='figure')
        else:
            plt.show()


