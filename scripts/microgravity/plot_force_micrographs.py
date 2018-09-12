import time, sys, os
import dill as pickle

import numpy as np
import scipy.constants as constants
import scipy.interpolate as interp
import scipy.optimize as opti
import scipy.signal as signal

import numdifftools as ndt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import grav_util_3 as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")



theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'

patches_base_path = '/processed_data/comsol_data/patch_potentials/'
patches_name = 'patch_pot_2um_0Vrms_bias-1Vdc'


# Include some legacy grav data to compare to later
data_dirs = [#'/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz', \
             #\
             #'/data/20180704/bead1/grav_data/shield', \
             #\
             #'/data/20180808/bead4/grav_data/shield1' \
             #\
             '/data/20180827/bead2/500e_data/dipole_v_height_ac' \
             #'/data/20180904/bead1/cant_force/ac_cant_elec_force_10s', \
             #'/data/20180904/bead1/recharged_20180909/cant_force/acgrid_3freqs_10s'
             ]

load_files = True

p0_bead_dict = {'20180625': [19.0, 40.0, 20.0], \
                '20180704': [18.7, 40.0, 20.0], \
                '20180808': [18.0, 40.0, 20.0], \
                '20180827': [35.0 ,40.0, 15.0]
                }


p0_bead_dict = {'20180827': [0.0 ,0.0, 0.0], \
                '20180904': [0.0 ,0.0, 0.0]
                }


opt_ext = '_electrostatics'

harms = [1]
#harms = [1,2,3,4,5]

charge = 430 * constants.elementary_charge * (-1.0)
plot_field_test = False
plot_cost_function = False
plot_keyscale = 1.0e-13

maxfreq = 200
dim3 = False


init = [15.0, 0.0, 28.0, -350]
init_rot = [15.0, 0.0, 28.0, -350, 0.0, 0.0, 0.0]


############################################################
############################################################
xx = np.load(open(patches_base_path + patches_name + '.xx', 'rb'))
yy = np.load(open(patches_base_path + patches_name + '.yy', 'rb'))
zz = np.load(open(patches_base_path + patches_name + '.zz', 'rb'))
dx = xx[1] - xx[0]
dy = yy[1] - yy[0]
dz = zz[1] - zz[0]

field = np.load(open(patches_base_path + patches_name + '.field', 'rb'))
potential = np.load(open(patches_base_path + patches_name + '.potential', 'rb')) 

print field[0].shape

gradE = []
gradE_func = []
for resp in [0,1,2]:
    gradE.append( np.gradient(field[resp], dx, dy, dz)[resp] )
    gradE_func.append( interp.RegularGridInterpolator((xx, yy, zz), gradE[resp], \
                                                      bounds_error=False, \
                                                      fill_value=None) )

pot_func = interp.RegularGridInterpolator((xx, yy, zz), potential, \
                                          bounds_error=False, fill_value=None)

field_func = []
for resp in [0,1,2]:
    field_func.append( interp.RegularGridInterpolator((xx, yy, zz), field[resp], \
                                          bounds_error=False, fill_value=None) )


if plot_field_test:
    posvec = np.linspace(-20e-6, 20e-6, 101)
    ones = np.ones_like(posvec)
    xval = 20.0e-6
    yval = 0.0e-6
    zval = 0.0e-6
    eval_pts = np.stack((xval*ones, posvec, zval*ones), axis=-1)
    eval_pts = np.stack((xval*ones, yval*ones, posvec), axis=-1)

    ann_str = 'Sep: %0.2f um, Height: %0.2f um' % (xval*1e6, zval*1e6)

    
    plt.figure()
    plt.plot(posvec*1e6, pot_func(eval_pts))

    plt.figure(figsize=(7,5))
    #plt.title(name)
    plt.plot(posvec*1e6, field_func[0](eval_pts)*charge, label='fx')
    plt.plot(posvec*1e6, field_func[1](eval_pts)*charge, label='fy')
    plt.plot(posvec*1e6, field_func[2](eval_pts)*charge, label='fz')
    plt.legend()
    plt.xlabel('Displacement Along Attractor [um]')
    plt.ylabel('Force on 500e$^-$ [N]')
    plt.annotate(ann_str, xy=(0.2, 0.9), xycoords='axes fraction')
    plt.tight_layout()
    plt.grid()

    plt.show()
############################################################
############################################################













############################################################
############################################################

for ddir in data_dirs:

    paths = gu.build_paths(ddir, opt_ext=opt_ext)

    datafiles = bu.find_all_fnames(ddir)
    p0_bead = p0_bead_dict[paths['date']]

    if load_files:
        agg_dat = gu.AggregateData(datafiles, p0_bead=p0_bead, harms=harms, \
                                   elec_drive=True, elec_ind=0, maxfreq=maxfreq, \
                                   plot_harm_extraction=False, dim3=dim3)        
        agg_dat.save(paths['agg_path'])
    else:
        agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms, dim3=dim3)
        agg_dat.load(paths['agg_path'])

    agg_dat.bin_rough_stage_positions(ax_disc=1.0, dim3=dim3)
    agg_dat.handle_sparse_binning(dim3=dim3, verbose=False)
    #print 'Testing sparse handling...',
    #agg_dat.handle_sparse_binning(dim3=dim3, verbose=True)
    #print 'Done!'
    #raw_input()

    #print agg_dat.ax0vec
    #print agg_dat.ax1vec
    #print agg_dat.ax2vec

    #print len(agg_dat.file_data_objs)

    force_plane_dict = agg_dat.get_vector_force_plane(plot_resp=(0,2), fig_ind=1, \
                                                      plot=True, show=True, \
                                                      sign=[1.0, 1.0, 1.0], dim3=dim3)

    err_dict = {0: 'xerr', 1: 'yerr', 2: 'zerr'}
    dat = [[], [], []]
    err = [[], [], []]
    for resp in [0,1,2]:
        dat[resp] = force_plane_dict[resp]
        err[resp] = force_plane_dict[err_dict[resp]]
    dat = np.array(dat)
    err = np.array(err)

    #for i in range(10):
    #    print force_plane_dict['drive'][i,:,:]
    #    raw_input()

    volt_drive = np.mean(force_plane_dict['drive'])
    print 'Voltage Drive: ', volt_drive
    #raw_input()
    
    #scale_fac = np.std(dat) * 0.1
    
    scale_fac = 1.0

    dat_sc = dat * (1.0 / scale_fac)
    err_sc = err * (1.0 / scale_fac)
    #err_sc = err_sc - err_sc + 1.0


    pos_dict = agg_dat.make_ax_arrs(dim3=dim3)
    seps = pos_dict['seps']
    heights = pos_dict['heights']
    if dim3:
        yposvec = pos_dict['ypos']
        seps_g, ypos_g, heights_g = np.meshgrid(seps, yposvec, heights, indexing='ij')
    else:
        yposvec = np.array([0.0])
        seps_g, heights_g = np.meshgrid(seps, heights, indexing='ij')


    rot_point = []


    def F_comsol_func(sep_off, ypos, height_off, charge, eval_resp=0, \
                      rot_angles=[0.0, 0.0, 0.0], rot_point=[], \
                      radians=False, plot_rot=False, \
                      add_dipole=False, dipole_moment=0.0, \
                      dim3=False):

        if dim3:
            interp_mesh = np.array([(seps_g + sep_off) * 1.0e-6, \
                                    (ypos_g + ypos) * 1.0e-6, \
                                    (heights_g + height_off) * 1.0e-6])
        else:
            interp_mesh = np.array(np.meshgrid((seps+sep_off)*1e-6, (yposvec+ypos)*1e-6, 
                                           (heights+height_off)*1e-6, indexing='ij'))

        interp_points = np.rollaxis(interp_mesh, 0, 4)
        interp_points = interp_points.reshape((interp_mesh.size // 3, 3))

        npts = interp_points.shape[0]

        rot_matrix = bu.euler_rotation_matrix(rot_angles, radians=radians)

        if not len(rot_point):
            p0 = []
            for resp in [0,1,2]:
                p0.append( np.mean(interp_points[:,resp]))
        else:
            p0 = rot_point
        p0 = np.array(p0)


        rot_pts = []
        for resp in [0,1,2]:
            rot_pts_vec = np.zeros(npts)
            for resp2 in [0,1,2]:
                rot_pts_vec += rot_matrix[resp,resp2] * (interp_points[:,resp2] - p0[resp2])
            rot_pts.append(rot_pts_vec)
            rot_pts_vec += p0[resp]
        rot_pts = np.array(rot_pts)
        rot_pts = rot_pts.T


        if plot_rot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(interp_points[:,0]*1e6, interp_points[:,1]*1e6, \
                       interp_points[:,2]*1e6, label='Original')
            ax.scatter(rot_pts[:,0]*1e6, rot_pts[:,1]*1e6, rot_pts[:,2]*1e6, \
                       label='Rot')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

        field_res = interp.interpn((xx, yy, zz), field[eval_resp], rot_pts, \
                                       bounds_error=False, fill_value=None)

        if add_dipole:
            grad_res = interp.interpn((xx, yy, zz), gradE[eval_resp], rot_pts, \
                                      bounds_error=False, fill_value=None)

        if dim3:
            shaped = np.reshape(field_res, (len(seps), len(yposvec), len(heights)))
            if add_dipole:
                shaped2 = np.reshape(grad_res, (len(seps), len(yposvec), len(heights)))
        else:
            shaped = np.reshape(field_res, (len(seps), len(heights)))
            if add_dipole:
                shaped2 = np.reshape(grad_res, (len(seps), len(heights)))

        if add_dipole:
            out = shaped * charge * constants.elementary_charge \
                    + shaped2 * dipole_moment * constants.elementary_charge * 1.0e-6
        else:
            out = shaped * charge * constants.elementary_charge

        return out #* volt_drive
    




    def cost_function(params, Ndof=True):
        delta_sep, ypos, delta_height, charge = params
        cost = 0.0
        N = 0
        for resp in [0,1,2]:
            func_vals = F_comsol_func(delta_sep, ypos, delta_height, charge, eval_resp=resp, \
                                      dim3=dim3) 
            func_vals *= (1.0 / scale_fac)
            diff_sq = np.abs( dat_sc[resp] - func_vals )**2
            #var = (np.ones_like(func_vals))**2
            var = (err_sc[resp])**2
            cost += np.sum( diff_sq / var )
            N += diff_sq.size
        if Ndof:
            cost *= (1.0 / float(N))
        return 0.5 * cost


    def cost_function_rot(params, Ndof=True):
        delta_sep, ypos, delta_height, charge, rotx, roty, rotz = params
        cost = 0.0
        N = 0
        for resp in [0,1,2]:
            func_vals = F_comsol_func(delta_sep, ypos, delta_height, charge, \
                                      rot_angles=[rotx, roty, rotz], rot_point=rot_point, \
                                      radians=False, eval_resp=resp, \
                                      dim3=dim3) 
            func_vals *= (1.0 / scale_fac)
            diff_sq = np.abs( dat_sc[resp] - func_vals )**2
            #var = (np.ones_like(func_vals))**2
            var = (err_sc[resp])**2
            cost += np.sum( diff_sq / var )
            N += diff_sq.size
        if Ndof:
            cost *= (1.0 / float(N))

        window = signal.tukey(91,0.8)
        angles = np.linspace(-45,45,91)
        penalty_box = interp.interp1d(angles, 1.0-window, bounds_error=False, fill_value=1.0)

        ang_penalty = 1.0e5 * ( penalty_box(rotx) + penalty_box(roty) + penalty_box(rotz))

        return 0.5 * cost #+ ang_penalty



    def diff_function(params, axes=[0,1,2]):
        delta_sep, ypos, delta_height, charge = params
        diff = []
        for resp in [0,1,2]:
            if resp not in axes:
                continue
            func_vals = F_comsol_func(delta_sep, ypos, delta_height, charge, eval_resp=resp, \
                                      dim3=dim3) 
            func_vals *= (1.0 / scale_fac)
            diff += list( ((dat_sc[resp] - func_vals) ).flatten() )#/ err_sc[resp]).flatten() )
        return np.array(diff)



    def diff_function_rot(params, axes=[0,1,2], no_penalty=False):
        delta_sep, ypos, delta_height, charge, rotx, roty, rotz = params
        #rotz = 0
        diff = []
        for resp in [0,1,2]:
            if resp not in axes:
                continue
            func_vals = F_comsol_func(delta_sep, ypos, delta_height, charge, \
                                      rot_angles=[rotx, roty, rotz], \
                                      rot_point=rot_point,\
                                      radians=False, eval_resp=resp, \
                                      dim3=dim3) 
            func_vals *= (1.0 / scale_fac)
            diff += list( ((func_vals - dat_sc[resp]) ).flatten() )#/ err_sc[resp]).flatten() )

        window = signal.tukey(91,0.8)
        angles = np.linspace(-45,45,91)
        penalty_box = interp.interp1d(angles, 2.5*(1.0-window) + 1, bounds_error=False, fill_value=1.0)

        ang_penalty = ( penalty_box(rotx) + penalty_box(roty) + penalty_box(rotz))

        ypenalty = 0
        if no_penalty:
            ang_penalty = 1.0
        return np.array(diff) #* ang_penalty



    if plot_cost_function:
        test = [np.linspace(5.0, 25.0, 101), \
                np.linspace(-10.0, 10.0, 101), \
                np.linspace(20.0, 40.0, 101)]

        cost_eval = [[], [], []]
        for resp in [0,1,2]:
            for ind, val in enumerate(test[resp]):
                params = [10.3, 0.0, 28.84, -331.57]
                params[resp] = val

                cost_eval[resp].append(cost_function(params))

            plt.plot(test[resp], cost_eval[resp])
        plt.show()



    ### Optimize the previously defined function(s)
    res = opti.minimize(cost_function, init)
    x = res.x
    #pcov = res.hess_inv
    #if pcov is None:
    #    print res

    Hfun = ndt.Hessian(cost_function, full_output=True)
    hessian_ndt, info = Hfun(x)
    pcov = 4.0 * np.linalg.inv(hessian_ndt)


    res2 = opti.leastsq(diff_function, init, full_output=True)
    x2 = res2[0]
    pcov2 = res2[1]
    if pcov2 is None:
        print res2

    res3 = opti.leastsq(diff_function_rot, init_rot, full_output=True)
    x3 = res3[0]
    pcov3 = res3[1]
    if pcov3 is None:
        print res3

    res4 = opti.minimize(cost_function_rot, init_rot)
    x4 = res4.x
    #pcov4 = res4.hess_inv
    #if pcov4 is None:
    #    print res4

    Hfun2 = ndt.Hessian(cost_function_rot, full_output=True)
    hessian_ndt2, info2 = Hfun2(x4)
    pcov4 = 4.0 * np.linalg.inv(hessian_ndt2)

    
    print 'PARAMETER ESTIMATES: sep, ypos, height, charge, (rotx, roty, rotz)'
    print "No rot 1 :  ", x
    print "No rot 2 :  ", x2
    print "Rot 1    :  ", x4
    print "Rot 2    :  ", x3


    var = np.diag(pcov)# * np.std(diff_function(x))**2)
    var2 = np.diag(pcov2 * np.std(diff_function(x2))**2)
    var3 = np.diag(pcov3 * np.std(diff_function_rot(x3,no_penalty=True))**2)
    var4 = np.diag(pcov4)# * np.std(diff_function_rot(x4,no_penalty=True))**2)


    print

    print 'PARAMETER ERRORS: sep, ypos, height, charge, (rotx, roty, rotz)'
    print 'Minimize     : ', np.sqrt(var)
    print 'Leastsq      : ', np.sqrt(var2)
    print 'Minimize rot : ', np.sqrt(var4)
    print 'Leastsq rot  : ', np.sqrt(var3)
    print

    raw_input()



    hist_fig, hist_axarr = plt.subplots(3,1,sharex=True)
    hist_axarr[0].set_title('Vector Force Plane Fit Residuals')
    hist_axarr[0].hist(diff_function(x2, axes=[0])*scale_fac, bins=30, \
                       range=(-5*scale_fac,5*scale_fac))
    hist_axarr[0].axvline(0, color='k')
    hist_axarr[0].set_ylabel('Fx')
    hist_axarr[1].hist(diff_function(x2, axes=[1])*scale_fac, bins=30, \
                       range=(-5*scale_fac,5*scale_fac))
    hist_axarr[1].axvline(0, color='k')
    hist_axarr[1].set_ylabel('Fy')
    hist_axarr[2].hist(diff_function(x2, axes=[2])*scale_fac, bins=30, \
                       range=(-10*scale_fac, 2.5*scale_fac))
    hist_axarr[2].axvline(0, color='k')
    hist_axarr[2].set_ylabel('Fz')
    hist_axarr[2].set_xlabel('Residual Force [N]')
    hist_axarr[2].set_xlim(-0.8e-14, 0.6e-14)
    plt.tight_layout()
    

    hist_fig_2, hist_axarr_2 = plt.subplots(3,1,sharex=True)
    hist_axarr_2[0].set_title('Vector Force Plane (With Rotations) Fit Residuals')
    hist_axarr_2[0].hist(diff_function_rot(x3, axes=[0])*scale_fac, bins=30, \
                         range=(-5*scale_fac,5*scale_fac))
    hist_axarr_2[0].axvline(0, color='k')
    hist_axarr_2[0].set_ylabel('Fx')
    hist_axarr_2[1].hist(diff_function_rot(x3, axes=[1])*scale_fac, bins=30, \
                         range=(-5*scale_fac,5*scale_fac))
    hist_axarr_2[1].axvline(0, color='k')
    hist_axarr_2[1].set_ylabel('Fy')
    hist_axarr_2[2].hist(diff_function_rot(x3, axes=[2])*scale_fac, bins=30, \
                         range=(-10*scale_fac, 2.5*scale_fac))
    hist_axarr_2[2].axvline(0, color='k')
    hist_axarr_2[2].set_ylabel('Fz')
    hist_axarr_2[2].set_xlabel('Residual Force [N]')
    hist_axarr_2[2].set_xlim(-0.8e-14, 0.6e-14)
    plt.tight_layout()
    #plt.show()



    soln = x3
    cov = pcov3



    if cov is None:
        err_mat = np.ones((len(soln), len(soln)))
    else:
        err_mat = np.abs(cov) * np.std(diff_function_rot(x3))**2


    err_mat_norm = np.copy(err_mat)
    for resp in range(err_mat.shape[0]):
        for resp2 in range(err_mat.shape[0]):
            err_mat_norm[resp,resp2] *= 1.0 / np.abs(soln[resp] * soln[resp2])

    print np.min(err_mat_norm)

    plt.figure()
    plt.imshow(err_mat_norm)
    plt.yticks([0, 1, 2, 3, 4, 5, 6], ['sep', 'ypos', 'height', 'charge', 'rotx', 'roty', 'rotz'])
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ['sep', 'ypos', 'height', 'charge', 'rotx', 'roty', 'rotz'])
    cbar = plt.colorbar()
    cbar.set_label('Relative Error')
    #plt.show()

       
    print "Separation offset from p0: %0.2f +- %0.3f" % (soln[0], np.sqrt(err_mat[0,0]))
    print "Ypos:                      %0.2f +- %0.3f" % (soln[1], np.sqrt(err_mat[1,1]))
    print "Height offset from p0:     %0.2f +- %0.3f" % (soln[2], np.sqrt(err_mat[2,2]))
    print "Qbead * Vcant product:     %0.2f +- %0.3f" % (soln[3], np.sqrt(err_mat[3,3]))

    print 
    print "Implied sep:    ", p0_bead[0] + soln[0]
    print "Implied height: ", p0_bead[2] + soln[2]

                       
    F_comsol = [np.zeros((len(seps), len(heights))), \
                np.zeros((len(seps), len(heights))), \
                np.zeros((len(seps), len(heights)))]

    for resp in [0,1,2]:

        if len(soln) > 4:
            F_comsol[resp] = F_comsol_func(*soln[:4], rot_angles=soln[4:], \
                                           eval_resp=resp, plot_rot=False, \
                                           rot_point=rot_point, dim3=dim3)
        else:
            F_comsol[resp] = F_comsol_func(*soln[:4], eval_resp=resp, dim3=dim3)
            
    keyscale = plot_keyscale #1.0e-14
    scale_pow = int(np.log10(keyscale))

    scale = keyscale * 4


    fig = plt.figure()
    ax = fig.add_subplot(111)

    yind = len(yposvec) / 2

    

    if dim3:
        qdat = ax.quiver(seps_g[:,yind,:]+soln[0], heights_g[:,yind,:]+soln[2], \
                         force_plane_dict[0][:,yind,:], force_plane_dict[2][:,yind,:], \
                         color='k', pivot='mid', label='Data', scale=scale)
        qfit = ax.quiver(seps_g[:,yind,:]+soln[0], heights_g[:,yind,:]+soln[2], \
                         F_comsol[0][:,yind,:], F_comsol[2][:,yind,:], \
                         color='r', pivot='mid', label='Fit', scale=scale)
    else:
        qdat = ax.quiver(seps_g+soln[0], heights_g+soln[2], force_plane_dict[0], force_plane_dict[2], \
                         color='k', pivot='mid', label='Data', scale=scale)
        qfit = ax.quiver(seps_g+soln[0], heights_g+soln[2], F_comsol[0], F_comsol[2], \
                         color='r', pivot='mid', label='Fit', scale=scale)
    
    ax.quiverkey(qdat, X=0.3, Y=1.05, U=keyscale, \
                 label='$10^{%i}~$N Force' % scale_pow, labelpos='N')
    ax.quiverkey(qfit, X=0.7, Y=1.05, U=keyscale, \
                 label='$10^{%i}~$N Force' % scale_pow, labelpos='N')
    ax.legend(loc=1)
    ax.set_xlabel('Separation [um]   |   Fx')
    ax.set_ylabel('Height [um]   |   Fz')


   



    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    if dim3:
        qdat2 = ax2.quiver(seps_g[:,yind,:]+soln[0], heights_g[:,yind,:]+soln[2], \
                          force_plane_dict[0][:,yind,:], force_plane_dict[1][:,yind,:], \
                         color='k', pivot='mid', label='Data', scale=scale)
        qfit2 = ax2.quiver(seps_g[:,yind,:]+soln[0], heights_g[:,yind,:]+soln[2], \
                          F_comsol[0][:,yind,:], F_comsol[1][:,yind,:], \
                         color='r', pivot='mid', label='Fit', scale=scale)
    else:
        qdat2 = ax2.quiver(seps_g+soln[0], heights_g+soln[2], force_plane_dict[0], force_plane_dict[1], \
                         color='k', pivot='mid', label='Data', scale=scale)
        qfit2 = ax2.quiver(seps_g+soln[0], heights_g+soln[2], F_comsol[0], F_comsol[1], \
                         color='r', pivot='mid', label='Fit', scale=scale)
    
    ax2.quiverkey(qdat2, X=0.3, Y=1.05, U=keyscale, \
                  label='$10^{%i}~$N Force' % scale_pow, labelpos='N')
    ax2.quiverkey(qfit2, X=0.7, Y=1.05, U=keyscale, \
                  label='$10^{%i}~$N Force' % scale_pow, labelpos='N')
    ax2.legend(loc=1)
    ax2.set_xlabel('Separation [um]   |   Fy')
    ax2.set_ylabel('Height [um]   |   Fz')
    plt.show()
