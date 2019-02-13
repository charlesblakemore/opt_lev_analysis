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
import matplotlib.patches as mpatches

import grav_util_3 as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")



theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'

patches_base_path = '/processed_data/comsol_data/patch_potentials/'
patches_name = 'patch_pot_2um_0Vrms_bias-1Vdc'
patches_name = 'patch_pot_2um_0Vrms_bias-1Vdc_8um-cant'


# Include some legacy grav data to compare to later
data_dirs = [#'/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz', \
             #\
             #'/data/20180704/bead1/grav_data/shield', \
             #\
             #'/data/20180808/bead4/grav_data/shield1' \
             #\
             #'/data/20180827/bead2/500e_data/dipole_v_height_ac' \
             #'/data/20180904/bead1/cant_force/ac_cant_elec_force_10s', \
             '/data/20180904/bead1/recharged_20180909/cant_force/acgrid_3freqs_10s', \
             #'/data/20180927/bead1/ac_force'
             ]

load_files = False

p0_bead_dict = {'20180625': [19.0, 40.0, 20.0], \
                '20180704': [18.7, 40.0, 20.0], \
                '20180808': [18.0, 40.0, 20.0], \
                '20180827': [35.0 ,40.0, 15.0]
                }


p0_bead_dict = {'20180625': [19.0, 40.0, 20.0], \
                '20180827': [0.0 ,0.0, 0.0], \
                '20180904': [0.0 ,0.0, 0.0], \
                '20180927': [0.0 ,0.0, 0.0] \
                }

tfdate = ''
#tfdate = '20180904'

opt_ext = '_electrostatics'

harms = [1]
#harms = [1,2,3,4,5]

charge = 430 * constants.elementary_charge * (-1.0)
plot_field_test = False
plot_cost_function = False

plot_keyscale = 3.0e-14
noise_plot_keyscale = 1.0e-16

maxfreq = 200
dim3 = True

unity_errors = False


#init = [15.0, 0.0, 28.0, -300]
init = [15.0, 40.0, 27.0, -380]


init_rot = [15.0, 40.0, 27.0, -300, 0.0, 0.0, 0.0]
#rot_point = []
rot_point = [0.0, 0.0, 0.0]


init_rot_2 = [15.0, 40.0, 27.0, -300, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


#ang_weight_fac = 5e6
ang_weight_fac = 10e-1
#ang_weight_fac = 10e1

#f_ang_weight_fac = 5e4
f_ang_weight_fac = 10e-1
#f_ang_weight_fac = 10e1



############################################################
############################################################

lamb = 1.064
w0 = 2.5
k = 2.0 * np.pi / lamb
zr = np.pi * w0**2 / lamb

def w(z):
    return w0 * np.sqrt(1 +(z / zr)**2)
def psi(x):
    return np.arctan(z / zr)
def R(z):
    if z == 0:
        return np.inf
    else:
        return z * (1 + (zr / z)**2)
def Efield_rz(A, r, z):
    exp1 = np.exp(-r**2 / w(z)**2)
    exp2 = np.exp(-1.0j * (k * z + k * r**2 / ( 2 * R(z)) - psi(z) ) )
    return np.real( A * (w0 / w(z)) * exp1 * exp2)


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
    posvec = np.linspace(-5e-6, 100e-6, 106)
    ones = np.ones_like(posvec)
    xval = 20.0e-6
    yval = 0.0e-6
    zval = 0.0e-6
    eval_pts = np.stack((posvec, yval*ones, zval*ones), axis=-1)
    #eval_pts = np.stack((xval*ones, posvec, zval*ones), axis=-1)
    #eval_pts = np.stack((xval*ones, yval*ones, posvec), axis=-1)

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
                                   plot_harm_extraction=False, dim3=dim3, \
                                   extract_resonant_freq=False,noiselim=(47.0,47.0), \
                                   tfdate=tfdate)        
        agg_dat.save(paths['agg_path'])
    else:
        agg_dat = gu.AggregateData([], p0_bead=p0_bead, harms=harms, dim3=dim3, \
                                   tfdate=tfdate)
        agg_dat.load(paths['agg_path'])

    agg_dat.bin_rough_stage_positions(ax_disc=1.0, dim3=dim3)
    agg_dat.handle_sparse_binning(dim3=dim3, verbose=False)

    force_plane_dict = agg_dat.get_vector_force_plane(plot_resp=(0,2), fig_ind=1, \
                                                      plot=True, show=False, \
                                                      sign=[1.0, 1.0, 1.0], dim3=dim3, \
                                                      keyscale=plot_keyscale)
    force_plane_dict_2 = agg_dat.get_vector_force_plane(plot_resp=(1,2), fig_ind=2, \
                                                        plot=True, show=False, \
                                                        sign=[1.0, 1.0, 1.0], dim3=dim3, \
                                                        keyscale=plot_keyscale)

    err_dict = {0: 'xerr', 1: 'yerr', 2: 'zerr'}
    noise_dict = {0: 'xnoise', 1: 'ynoise', 2: 'znoise'}
    dat = [[], [], []]
    err = [[], [], []]
    noise = [[], [], []]
    for resp in [0,1,2]:
        dat[resp] = force_plane_dict[resp]
        err[resp] = force_plane_dict[err_dict[resp]]
        noise[resp] = force_plane_dict[noise_dict[resp]]
    dat = np.array(dat)
    err = np.array(err)
    noise = np.array(noise)



    err_zero = (err == 0.0)
    mean_err = np.mean(err)
    err += err_zero * mean_err * 10

    err_small = (err < 0.1 * mean_err)
    err += err_small * mean_err * 10  # to deweight points with crazy bad errors

    for resp in [0,1,2]:
        replace_val = np.median(np.abs(noise[resp].flatten()))
        crazy_noise = (np.abs(noise[resp]) > 5.0 * replace_val)
        noise[resp] = (noise[resp] - (noise[resp] - replace_val)*crazy_noise) 




    #for i in range(10):
    #    print force_plane_dict['drive'][i,:,:]
    #    raw_input()

    volt_drive = np.mean(force_plane_dict['drive'])
    print 'Voltage Drive: ', volt_drive
    #raw_input()
    
    #hist_scale_fac = np.std(dat) * 1000
    scale_fac = 1.0

    dat_sc = dat * (1.0 / scale_fac)
    err_sc = np.abs(err) * (1.0 / scale_fac)
    if unity_errors:
        err_sc = err_sc - err_sc + 1.0

    dat_sc[1] = np.zeros_like(dat_sc[1])

    pos_dict = agg_dat.make_ax_arrs(dim3=dim3)
    seps = pos_dict['seps']
    heights = pos_dict['heights']
    if dim3:
        yposvec = pos_dict['ypos']
        seps_g, ypos_g, heights_g = np.meshgrid(seps, yposvec, heights, indexing='ij')
    else:
        yposvec = np.array([0.0])
        seps_g, heights_g = np.meshgrid(seps, heights, indexing='ij')


    sep_fit_inds = (seps > 0.0) * (seps < 50.0)
    y_fit_inds = (yposvec > -250.0) * (yposvec < 250.0)
    height_fit_inds = (heights > -50.0) * (heights < 50.0)

    print len(seps), len(sep_fit_inds)
    print len(yposvec), len(y_fit_inds)
    print len(heights), len(height_fit_inds)

    print dat_sc.shape
    #raw_input()

    def F_comsol_func(sep_off, ypos, height_off, charge, eval_resp=0, \
                      pos_rot_angles=[0.0, 0.0, 0.0], rot_point=[], \
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

        rot_matrix = bu.euler_rotation_matrix(pos_rot_angles, radians=radians)

        if not len(rot_point):
            p0 = []
            for resp in [0,1,2]:
                p0.append( np.mean(interp_points[:,resp]))
        else:
            p0 = rot_point
        p0 = np.array(p0)


        rot_pts = bu.rotate_points(interp_points, rot_matrix, p0, plot=plot_rot)
        field_res = interp.interpn((xx, yy, zz), field[eval_resp], rot_pts, \
                                       bounds_error=False, fill_value=None)

        if dim3:
            shaped = np.reshape(field_res, (len(seps), len(yposvec), len(heights)))
        else:
            shaped = np.reshape(field_res, (len(seps), len(heights)))
        

        out = shaped * charge * constants.elementary_charge
        return out * volt_drive
    





    '''
    Fcomsol = []
    for resp in [0,1,2]:
        Fcomsol.append(F_comsol_func(15, 0, 28, -50, eval_resp=resp, \
                      pos_rot_angles=[0.0, 0.0, 0.0], rot_point=[], \
                      radians=False, plot_rot=False, \
                      add_dipole=False, dipole_moment=0.0, \
                      dim3=False))
    

    fig = plt.figure(7)
    ax = fig.add_subplot(111)
        
    qdat = ax.quiver(seps_g, heights_g, dat[0], dat[2], \
                     color='k', pivot='mid', label='Force', scale=plot_keyscale*4)
    qcom = ax.quiver(seps_g, heights_g, Fcomsol[0], Fcomsol[2], \
                     color='r', pivot='mid', label='Error', scale=plot_keyscale*4)
    ax.set_xlabel('Separation [um]')
    ax.set_ylabel('Height [um]')
    plt.show()
    '''





    def cost_function(params, Ndof=False):
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



    def cost_function_rot_2(params, Ndof=False):
        delta_sep, ypos, delta_height, charge, rotx, roty, rotz, frotx, froty, frotz = params
        cost = 0.0
        N = 0
        
        frot_mat = bu.euler_rotation_matrix([frotx, froty, frotz], radians=False)
        if dim3:
            dat_sc_rot = np.einsum('ij,jabc->iabc', frot_mat, dat_sc)
        else:
            dat_sc_rot = np.einsum('ij,jab->iab', frot_mat, dat_sc)

        for resp in [0,1]:#,2]:
            func_vals = F_comsol_func(delta_sep, ypos, delta_height, charge, \
                                      pos_rot_angles=[rotx, roty, rotz], rot_point=rot_point, \
                                      radians=False, eval_resp=resp, \
                                      dim3=dim3) 
            func_vals *= (1.0 / scale_fac)
            # [sep_fit_inds][y_fit_inds][height_fit_inds] - \
            diff_sq = np.abs( dat_sc_rot[resp] - \
                              func_vals )**2
            #var = (np.ones_like(func_vals))**2
            var = (err_sc[resp])**2 #* 1000.0
            cost += np.sum( diff_sq / var )
            N += diff_sq.size
        if Ndof:
            cost *= (1.0 / float(N))

        window = signal.tukey(91,0.8)
        angles = np.linspace(-45,45,91)
        penalty_box = interp.interp1d(angles, 1.0-window, bounds_error=False, fill_value=1.0)
        penalty_box = interp.interp1d(angles, angles**2 / 1600., bounds_error=False, fill_value=1.0)

        #plt.plot(angles, penalty_box(angles))
        #plt.show()

        ang_penalty = ang_weight_fac * ( penalty_box(rotx) + penalty_box(roty) + penalty_box(rotz))
        ang_penalty_2 = f_ang_weight_fac * ( penalty_box(frotx) + penalty_box(froty) + penalty_box(frotz))

        return 0.5*cost + ang_penalty + ang_penalty_2
    




    def diff_function(params, axes=[0,1,2], no_weights=False, relative=False):
        delta_sep, ypos, delta_height, charge = params
        diff = []
        for resp in [0,1,2]:
            if resp not in axes:
                continue
            func_vals = F_comsol_func(delta_sep, ypos, delta_height, charge, eval_resp=resp, \
                                      dim3=dim3) 
            func_vals *= (1.0 / scale_fac)
            if no_weights:
                if relative:
                    diff += list( ((func_vals - dat_sc[resp]) / func_vals).flatten() )
                else:
                    diff += list( (func_vals - dat_sc[resp]).flatten() )
            else:
                diff += list( ((func_vals - dat_sc[resp]) / err_sc[resp]).flatten() )
        return np.array(diff)




    def diff_function_rot_2(params, axes=[0,1,2], no_penalty=False, no_weights=False, \
                            relative=False):
        delta_sep, ypos, delta_height, charge, rotx, roty, rotz, frotx, froty, frotz = params
        #rotz = 0
        diff = []
        
        frot_mat = bu.euler_rotation_matrix([frotx, froty, frotz], radians=False)
        if dim3:
            dat_sc_rot = np.einsum('ij,jabc->iabc', frot_mat, dat_sc)
            err_sc_rot = np.einsum('ij,jabc->iabc', frot_mat, err_sc)
        else:
            dat_sc_rot = np.einsum('ij,jab->iab', frot_mat, dat_sc)
            err_sc_rot = np.einsum('ij,jab->iab', frot_mat, err_sc)

        for resp in [0,1,2]:
            if resp not in axes:
                continue
            func_vals = F_comsol_func(delta_sep, ypos, delta_height, charge, \
                                      pos_rot_angles=[rotx, roty, rotz], \
                                      rot_point=rot_point,\
                                      radians=False, eval_resp=resp, \
                                      dim3=dim3) 
            func_vals *= (1.0 / scale_fac)
            if no_weights:
                if relative:
                    diff += list( ((func_vals - dat_sc_rot[resp]) / func_vals).flatten() )
                else:
                    diff += list( (func_vals - dat_sc_rot[resp]).flatten() )
            else:
                diff += list( ((func_vals - dat_sc_rot[resp]) / err_sc_rot[resp]).flatten() )

        window = signal.tukey(91,0.8)
        angles = np.linspace(-45,45,91)
        penalty_box = interp.interp1d(angles, 2.5*(1.0-window) + 1, bounds_error=False, fill_value=1.0)

        ang_penalty = ( penalty_box(rotx) + penalty_box(roty) + penalty_box(rotz))

        ypenalty = 0
        if no_penalty:
            ang_penalty = 1.0
        return np.array(diff) #* ang_penalty




    ### Optimize the previously defined function(s)
    no_rot_res = opti.minimize(cost_function, init, method='SLSQP')
    no_rot_soln = no_rot_res.x

    Hfun = ndt.Hessian(cost_function, full_output=True)
    hessian_ndt, info = Hfun(no_rot_soln)
    no_rot_pcov = np.linalg.inv(hessian_ndt)

    for i in [0,1,2,3]:
        init_rot_2[i] = no_rot_soln[i]

    rot_res = opti.minimize(cost_function_rot_2, init_rot_2, method='SLSQP')
    rot_soln = rot_res.x

    Hfun2 = ndt.Hessian(cost_function_rot_2, full_output=True)
    hessian_ndt2, info2 = Hfun2(rot_soln)
    rot_pcov = np.linalg.inv(hessian_ndt2)

    #test_angles = np.linspace(-45.0, 45.0, 200)
    #test_cost = []
    #for angle in test_angles:
    #    test_cost.append( cost_function_rot_2( [rot_soln[0], rot_soln[1], rot_soln[2], \
    #                                            rot_soln[3], rot_soln[4], rot_soln[5], \
    #                                            rot_soln[6], rot_soln[7], rot_soln[8], \
    #                                            angle] ) )
    #plt.plot(test_angles, test_cost)
    #plt.show()

    print "FIT RESULTS"
    print "            NO ROT      ROT "
    print "SEP:         %0.2f      %0.2f"  %  (no_rot_soln[0], rot_soln[0])
    print "YPOS:       %0.2f      %0.2f"  %  (no_rot_soln[1], rot_soln[1])
    print "HEIGHT:     %0.2f      %0.2f"  %  (no_rot_soln[2], rot_soln[2])
    print "CHARGE:   %0.2f     %0.2f"  %  (no_rot_soln[3], rot_soln[3])
    print "ROTX:                  %0.2f"  %  rot_soln[4]
    print "ROTY:                  %0.2f"  %  rot_soln[5]
    print "ROTZ:                  %0.2f"  %  rot_soln[6]
    print "FROTX:                 %0.2f"  %  rot_soln[7]
    print "FROTY:                 %0.2f"  %  rot_soln[8]
    print "FROTZ:                 %0.2f"  %  rot_soln[9]





    no_rot_var = np.diag(no_rot_pcov) #* np.sum(diff_function(no_rot_soln, no_weights=True)**2)
    rot_var = np.diag(rot_pcov) #* np.sum(diff_function_rot_2(rot_soln, no_weights=True)**2)

    print

    print 'PARAMETER ERRORS: sep, ypos, height, charge, (rotx, roty, rotz)'
    print 'No Rot     : ', np.sqrt(no_rot_var)
    print 'Rot        : ', np.sqrt(rot_var)
    print

    #raw_input()


    #uprange1 = 2.0 
    uprange1 = 0.5e-14
    #downrange1 = -2.0 
    downrange1 = -0.5e-14

    uprange2 = 2.0 
    uprange2 = 0.5e-14
    downrange2 = -2.0 
    downrange2 = -0.5e-14
    relative = False

    hist_fig, hist_axarr = plt.subplots(3,1,sharex=True)
    hist_axarr[0].set_title('3D Vector Field Fit Residuals')
    hist_axarr[0].hist(diff_function(no_rot_soln, axes=[0], no_weights=True, relative=relative), \
                       bins=30, range=(downrange1, uprange1))
    hist_axarr[0].axvline(0, color='k')
    hist_axarr[0].set_ylabel('Fx')
    hist_axarr[1].hist(diff_function(no_rot_soln, axes=[1], no_weights=True, relative=relative), \
                       bins=30, range=(downrange1, uprange1))
    hist_axarr[1].axvline(0, color='k')
    hist_axarr[1].set_ylabel('Fy')
    hist_axarr[2].hist(diff_function(no_rot_soln, axes=[2], no_weights=True, relative=relative), \
                       bins=30, range=(downrange1, uprange1))
    hist_axarr[2].axvline(0, color='k')
    hist_axarr[2].set_ylabel('Fz')
    if relative:
        hist_axarr[2].set_xlabel('Residual Force / Expected Force [abs]')
    else:
        hist_axarr[2].set_xlabel('Residual Force [N]')
    #hist_axarr[2].set_xlim(-0.8e-14, 0.6e-14)
    plt.tight_layout()




    hist_fig_2, hist_axarr_2 = plt.subplots(3,1,sharex=True)
    hist_axarr_2[0].set_title('3D Vector Field (With Rotations) Fit Residuals')
    hist_axarr_2[0].hist(diff_function_rot_2(rot_soln, axes=[0], no_weights=True, relative=relative), \
                         bins=30, range=(downrange2, uprange2))
    hist_axarr_2[0].axvline(0, color='k')
    hist_axarr_2[0].set_ylabel('Fx')
    hist_axarr_2[1].hist(diff_function_rot_2(rot_soln, axes=[1], no_weights=True, relative=relative), \
                         bins=30, range=(downrange2, uprange2))
    hist_axarr_2[1].axvline(0, color='k')
    hist_axarr_2[1].set_ylabel('Fy')
    hist_axarr_2[2].hist(diff_function_rot_2(rot_soln, axes=[2], no_weights=True, relative=relative), \
                         bins=30, range=(downrange2, uprange2))
    hist_axarr_2[2].axvline(0, color='k')
    hist_axarr_2[2].set_ylabel('Fz')
    if relative:
        hist_axarr_2[2].set_xlabel('Residual Force / Expected Force [abs]')
    else:
        hist_axarr_2[2].set_xlabel('Residual Force [N]')
    #hist_axarr_2[2].set_xlim(-0.8e-14, 0.6e-14)
    plt.tight_layout()
    #plt.show()




    F_comsol = [np.zeros((len(seps), len(heights))), \
                np.zeros((len(seps), len(heights))), \
                np.zeros((len(seps), len(heights)))]

    F_comsol_frot = [np.zeros((len(seps), len(heights))), \
                     np.zeros((len(seps), len(heights))), \
                     np.zeros((len(seps), len(heights)))]

    soln_angles = [rot_soln[4], rot_soln[5], rot_soln[6]]
    #soln_rot = [0.0, 0.0, 0.0]
    soln_f_angles = [rot_soln[7], rot_soln[8], rot_soln[9]]
    #soln_frot = [0.0, 0.0, 0.0]

    soln_rot_mat = bu.euler_rotation_matrix(soln_angles, radians=False)
    soln_frot_mat = bu.euler_rotation_matrix(soln_f_angles, radians=False)

    for resp in [0,1,2]:
        F_comsol[resp] = F_comsol_func(*no_rot_soln, eval_resp=resp, dim3=dim3)

        F_comsol_frot[resp] = F_comsol_func(*rot_soln[:4], pos_rot_angles=soln_angles, \
                                            eval_resp=resp, plot_rot=False, \
                                            rot_point=rot_point, dim3=dim3)

    rot_pos = bu.rotate_meshgrid(seps+rot_soln[0], yposvec+rot_soln[1], heights+rot_soln[2], \
                                 soln_rot_mat, rot_point, plot=False)

    force_grids = np.array([force_plane_dict[0], force_plane_dict[1], force_plane_dict[2]])
    if dim3:
        rot_force_grids = np.einsum('ij, jabc -> iabc', soln_frot_mat, force_grids)
    else:
        rot_force_grids = np.einsum('ij, jab -> iab', soln_frot_mat, force_grids)



    if rot_pcov is None:
        err_mat = np.ones((len(soln), len(soln)))
    else:
        err_mat = np.abs(rot_pcov) #* np.std(diff_function_rot_2(soln, no_penalty=True, \
                                   #                             no_weights=True))**2

    #print err_mat

    err_mat_norm = np.copy(err_mat)
    for resp in range(err_mat.shape[0]):
        for resp2 in range(err_mat.shape[0]):
            err_mat_norm[resp,resp2] *= 1.0 / np.abs(rot_soln[resp] * rot_soln[resp2])

    #print np.min(err_mat_norm)

    plt.figure()
    plt.imshow(err_mat_norm)
    plt.yticks([0, 1, 2, 3, 4, 5, 6], ['sep', 'ypos', 'height', 'charge', 'rotx', 'roty', 'rotz'])
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ['sep', 'ypos', 'height', 'charge', 'rotx', 'roty', 'rotz'])
    cbar = plt.colorbar()
    cbar.set_label('Relative Error')
    #plt.show()

       
    #print "Separation offset from p0: %0.2f +- %0.3f" % (soln[0], np.sqrt(err_mat[0,0]))
    #print "Ypos:                      %0.2f +- %0.3f" % (soln[1], np.sqrt(err_mat[1,1]))
    #print "Height offset from p0:     %0.2f +- %0.3f" % (soln[2], np.sqrt(err_mat[2,2]))
    #print "Qbead * Vcant product:     %0.2f +- %0.3f" % (soln[3], np.sqrt(err_mat[3,3]))

    #print 
    #print "Implied sep:    ", p0_bead[0] + soln[0]
    #print "Implied height: ", p0_bead[2] + soln[2]

                       

    yind = len(yposvec) / 2 #- 1

    err_rms = np.zeros_like(err[0])
    for resp in [0,1,2]:
        err_rms += err[resp]**2
    err_rms = np.sqrt(err_rms)

    if dim3:
        plot_err = err_rms[:,yind,:]
    else:
        plot_err = err_rms[:,:]
    mean_rms = np.mean(plot_err)
    std_rms = np.std(plot_err)

    plt.figure()
    plt.hist(plot_err.flatten(), bins=50)
    #plt.show()

    inds = np.abs(plot_err-mean_rms) > 3.0 * std_rms

    plot_err[inds] *= mean_rms / plot_err[inds]


            
    keyscale = plot_keyscale #1.0e-14
    scale_pow = int(np.floor(np.log10(keyscale)))
    val = plot_keyscale / 10.0**scale_pow


    scale = keyscale * 4

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)

    if dim3:
        qdat = ax.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                           rot_force_grids[0][:,yind,:], rot_force_grids[2][:,yind,:], \
                         color='k', pivot='middle', label='Data', scale=scale)
        qfit = ax.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                           F_comsol_frot[0][:,yind,:], F_comsol_frot[2][:,yind,:], \
                           facecolor='r', pivot='middle', label='Best Fit FEA', scale=scale, \
                           alpha=0.6, linewidth=20.0, linestyle='--')
    else:
        qdat = ax.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                           rot_force_grids[0], rot_force_grids[2], \
                         color='k', pivot='middle', label='Data', scale=scale)
        qfit = ax.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                           F_comsol_frot[0], F_comsol_frot[2], \
                           color='r', pivot='middle', label='Best Fit FEA', scale=scale)
    
    ax.add_patch(mpatches.Rectangle((-20.0, -5.0), 20, 10, fill=None))

    #ax.text(-7.5, 7.5, 'Cantilever')
    ax.annotate('ACSB', xy=(-1, 6), xytext=(-1,15), \
                arrowprops=dict(facecolor='black', shrink=0.05, width=3), \
                horizontalalignment='center', verticalalignment='bottom')

    ax.quiverkey(qdat, X=0.2, Y=1.05, U=keyscale, \
                 label=r'Data: $%0.1f \times 10^{%i}~$N' % (val, scale_pow), \
                 labelpos='N', fontproperties={'size': 14})
    ax.quiverkey(qfit, X=0.75, Y=1.05, U=keyscale, \
                 label=r'Best Fit FEA: $%0.1f \times 10^{%i}~$N' % (val, scale_pow),\
                 labelpos='N', fontproperties={'size': 14})


    #ax.legend(loc=1)
    #ax.set_xlabel('$x$ [$\mu$m]   |   $F_x$', fontsize=16)
    ax.set_xlabel('$x$ [$\mu$m]', fontsize=16)
    #ax.set_ylabel('$z$ [$\mu$m]   |   $F_z$', fontsize=16)
    ax.set_ylabel('$z$ [$\mu$m]', fontsize=16)

    ax.scatter(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], s=8, marker='o', \
                color='w', edgecolor='k', linewidth=0.75)

    ax.set_xlim(-10.0, np.max(seps_g+rot_soln[0]) * 1.1)

    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)
   




    fig2 = plt.figure(dpi=200)
    ax2 = fig2.add_subplot(111)


    if dim3:
        qdat2 = ax2.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                           rot_force_grids[1][:,yind,:], rot_force_grids[2][:,yind,:], \
                         color='k', pivot='middle', label='Data', scale=scale)
        qfit2 = ax2.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                           F_comsol_frot[1][:,yind,:], F_comsol_frot[2][:,yind,:], \
                           facecolor='r', pivot='middle', label='Best Fit FEA', scale=scale, \
                           alpha=0.6, linewidth=20.0, linestyle='--')
    else:
        qdat2 = ax2.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                           rot_force_grids[1], rot_force_grids[2], \
                         color='k', pivot='middle', label='Data', scale=scale)
        qfit2 = ax2.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                           F_comsol_frot[1], F_comsol_frot[2], \
                           color='r', pivot='middle', label='Best Fit FEA', scale=scale)
    
    ax2.add_patch(mpatches.Rectangle((-20.0, -5.0), 20, 10, fill=None))
    #ax2.text(-7.5, 7.5, 'Cantilever')
    ax2.annotate('ACSB', xy=(-1, 6), xytext=(-1,15), \
                arrowprops=dict(facecolor='black', shrink=0.05, width=3), \
                horizontalalignment='center', verticalalignment='bottom')


    ax2.quiverkey(qdat2, X=0.2, Y=1.05, U=keyscale, \
                 label=r'Data: $%0.1f \times 10^{%i}~$N' % (val, scale_pow), \
                  labelpos='N', fontproperties={'size': 14})
    ax2.quiverkey(qfit2, X=0.75, Y=1.05, U=keyscale, \
                 label=r'Best Fit FEA: $%0.1f \times 10^{%i}~$N' % (val, scale_pow), \
                  labelpos='N', fontproperties={'size': 14})
    #ax2.legend(loc=1)
    #ax2.set_xlabel('$x$ [$\mu$m]   |   $F_y$', fontsize=16)
    ax2.set_xlabel('$x$ [$\mu$m]', fontsize=16)
    #ax2.set_ylabel('$z$ [$\mu$m]   |   $F_z$', fontsize=16)
    ax2.set_ylabel('$z$ [$\mu$m]', fontsize=16)

    ax2.scatter(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], s=8, marker='o', \
                color='w', edgecolor='k', linewidth=0.75)

    ax2.set_xlim(-10.0, np.max(seps_g+rot_soln[0]) * 1.1)

    plt.setp(ax2.get_xticklabels(), fontsize=16)
    plt.setp(ax2.get_yticklabels(), fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)




    plot_inds = [1,3,5,7,9]
    #plot_inds = range(len(yposvec))

    #noise_colors = bu.get_color_map(len(yposvec), cmap='jet')
    noise_colors = bu.get_color_map(len(plot_inds), cmap='inferno')
    noise_colors = noise_colors[::-1]
    #noise_colors = []
    #for i in range(9):
    #    noise_colors.append( 'C' + str(i) )
    #noise_colors.append('k')

    strs = []
    #handles = []
    for yvecind, yval in enumerate(yposvec):
        if yvecind == -1:
            continue
        ystr = r'$y=%0.1f~\mu$m' % (yval + rot_soln[1])
        strs.append(ystr)
        #handles.append(mpatches.Patch(color=noise_colors[yvecind], label=ystr))
    #handles.append(mpatches.Patch(color='k', label= r'$y=%0.1f~\mu$m' % (yposvec[-1])))
    strs.append(r'$y=%0.1f~\mu$m' % (yposvec[-1] + rot_soln[1]))

    strs = []
    for yvecind in plot_inds:
        yval = yposvec[yvecind]
        ystr = r'$y=%0.1f~\mu$m' % (yval + rot_soln[1])
        strs.append(ystr)


    noise_keyscale = noise_plot_keyscale #1.0e-14
    noise_scale_pow = int(np.floor(np.log10(noise_keyscale)))
    
    noise_val = noise_keyscale / 10.0**noise_scale_pow

    noise_scale = noise_keyscale * 4
    noise_width = 5.0e-3


    fig3 = plt.figure(dpi=200)
    ax3 = fig3.add_subplot(111)

    if dim3:
        for derpind, yvecind in enumerate(plot_inds[:-1]):
            qdat3 = ax3.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                               noise[0][:,yvecind,:], noise[2][:,yvecind,:], \
                               pivot='tail',  scale=noise_scale, width=noise_width, \
                               color=noise_colors[derpind], label=str(derpind))
    
        qnoise3 = ax3.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                             noise[0][:,plot_inds[-1],:], noise[2][:,plot_inds[-1],:], \
                             color=noise_colors[-1], pivot='tail', \
                             scale=noise_scale, width=noise_width, label='-1')
        handles, labels = ax3.get_legend_handles_labels()
        
    else:
    
        qnoise3 = ax3.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                             noise[0], noise[2], \
                             color='k', pivot='tail', \
                             scale=noise_scale, width=noise_width)
        

    ax3.add_patch(mpatches.Rectangle((-20.0, -5.0), 20, 10, fill=None))
    #ax3.text(-7.5, 7.5, 'Cantilever')
    ax3.annotate('ACSB', xy=(-1, 6), xytext=(-1,15), \
                arrowprops=dict(facecolor='black', shrink=0.05, width=3), \
                horizontalalignment='center', verticalalignment='bottom')

    ax3.quiverkey(qnoise3, X=0.5, Y=1.05, U=noise_keyscale, \
                  label=r'Noise: $%0.1f \times 10^{%i}~$N' % (noise_val, noise_scale_pow), \
                  labelpos='N', fontproperties={'size': 14})
    #ax3.legend(loc=1)
    #ax3.set_xlabel('$x$ [$\mu$m]   |   $F_x$', fontsize=16)
    ax3.set_xlabel('$x$ [$\mu$m]', fontsize=16)
    #ax3.set_ylabel('$z$ [$\mu$m]   |   $F_z$', fontsize=16)
    ax3.set_ylabel('$z$ [$\mu$m]', fontsize=16)

    ax3.scatter(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], s=8, marker='o', \
                color='w', edgecolor='k', linewidth=0.75)

    ax3.set_xlim(-10.0, np.max(seps_g+rot_soln[0]) * 1.1)

    plt.setp(ax3.get_xticklabels(), fontsize=16)
    plt.setp(ax3.get_yticklabels(), fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)

    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width*0.81, box.height])

    ax3.legend(handles, ['%0.1f' % (yposvec[ind] + rot_soln[1]) for ind in plot_inds], \
               loc='center left', bbox_to_anchor=[1,0.5], fontsize=12)
    ax3.text(112, 15, '$y$ [$\mu$m]', fontsize=14)







    fig4 = plt.figure(dpi=200)
    ax4 = fig4.add_subplot(111)

    if dim3:
        for derpind, yvecind in enumerate(plot_inds[:-1]):
            qdat4 = ax4.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                               noise[1][:,yvecind,:], noise[2][:,yvecind,:], \
                               pivot='tail',  scale=noise_scale, width=noise_width, \
                               color=noise_colors[derpind], label=str(derpind))
    
        qnoise4 = ax4.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                             noise[1][:,plot_inds[-1],:], noise[2][:,plot_inds[-1],:], \
                             color=noise_colors[-1], pivot='tail', \
                             scale=noise_scale, width=noise_width, label='-1')
        handles, labels = ax4.get_legend_handles_labels()
    else:
    
        qnoise4 = ax4.quiver(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], \
                             noise[1], noise[2], \
                             color='k', pivot='tail', \
                             scale=noise_scale, width=noise_width)

    ax4.add_patch(mpatches.Rectangle((-20.0, -5.0), 20, 10, fill=None))
    #ax4.text(-7.5, 7.5, 'Cantilever')
    ax4.annotate('ACSB', xy=(-1, 6), xytext=(-1,15), \
                arrowprops=dict(facecolor='black', shrink=0.05, width=3), \
                horizontalalignment='center', verticalalignment='bottom')

    ax4.quiverkey(qnoise4, X=0.5, Y=1.05, U=noise_keyscale, \
                  label=r'Noise: $%0.1f \times 10^{%i}~$N' % (noise_val, noise_scale_pow), \
                  labelpos='N', fontproperties={'size': 14})
    #ax4.legend(loc=1)
    #ax4.set_xlabel('$x$ [$\mu$m]   |   $F_y$', fontsize=16)
    ax4.set_xlabel('$x$ [$\mu$m]', fontsize=16)
    #ax4.set_ylabel('$z$ [$\mu$m]   |   $F_z$', fontsize=16)
    ax4.set_ylabel('$z$ [$\mu$m]', fontsize=16)

    ax4.scatter(rot_pos[0][:,yind,:], rot_pos[2][:,yind,:], s=8, marker='o', \
                color='w', edgecolor='k', linewidth=0.75)

    ax4.set_xlim(-10.0, np.max(seps_g+rot_soln[0]) * 1.1)

    plt.setp(ax4.get_xticklabels(), fontsize=16)
    plt.setp(ax4.get_yticklabels(), fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.87)

    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0, box.width*0.81, box.height])

    ax4.legend(handles, ['%0.1f' % (yposvec[ind] + rot_soln[1]) for ind in plot_inds], \
               loc='center left', bbox_to_anchor=[1,0.5], fontsize=12)
    ax4.text(112, 15, '$y$ [$\mu$m]', fontsize=14)



    def gauss(x, A, mu, sigma):
        return (A / (2.0 * np.pi * sigma**2)) * np.exp( -1.0 * (x - mu)**2 / (2.0 * sigma**2) )


    bin_arr = [50, 50, 50]
    colors = ['C0', 'C1', 'C2']
    ax_arr = ['x', 'y', 'z']
    errhist_fig, errhist_ax = plt.subplots(3,1,sharex=True,dpi=200)
    for resp in [0,1,2]:
        n, bin_edge, patch = errhist_ax[resp].hist(noise[resp].flatten()*1e18, range=(-70, 70), \
                                                   bins=bin_arr[resp], color='w', \
                                                   edgecolor='k', linewidth=2)
        real_bins = bin_edge[:-1] + 0.5 * (bin_edge[1] - bin_edge[0])

        popt, pcov = opti.curve_fit(gauss, real_bins, n, p0=[100, 0, 20])

        lab = r'$\sigma_{%s}$=%0.1f aN' % (ax_arr[resp], np.abs(popt[2]))

        plot_vals = np.linspace(bin_edge[0], bin_edge[-1], 500)
        errhist_ax[resp].plot(plot_vals, gauss(plot_vals, *popt), color='r', lw=2, \
                              label=lab)
        errhist_ax[resp].legend(loc=1, fontsize=14)
        errhist_ax[resp].set_ylabel('Counts', fontsize=16)
        
        errhist_ax[resp].set_xlim(-70, 70)

        plt.setp(errhist_ax[resp].get_xticklabels(), fontsize=16)
        plt.setp(errhist_ax[resp].get_yticklabels(), fontsize=16)

    errhist_ax[-1].set_xlabel('Force noise [aN]', fontsize=16)
    plt.tight_layout()




    plt.show()








