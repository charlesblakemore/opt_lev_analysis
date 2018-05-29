import sys, time

import dill as pickle

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.stats as stats
import scipy.optimize as opti

import bead_util as bu
import calib_util as cal
import transfer_func_util as tf
import configuration as config





theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'



#########################################################

def build_mod_grav_funcs(theory_data_dir):
    '''Loads data from the output of /data/grav_sim_data/process_data.py
       which processes the raw simulation output from the farmshare code

       INPUTS: data_dir, path to the directory containing the data

       OUTPUTS: gfuncs, 3 element list with 3D interpolating functions
                        for regular gravity [fx, fy, fz]
                yukfuncs, 3 x Nlambda array with 3D interpolating function
                          for modified gravity with indexing: 
                          [[y0_fx, y1_fx, ...], [y0_fy, ...], [y0_fz, ...]]
                lambdas, np.array with all lambdas from the simulation
                lims, 3 element with tuples for (min, max) of coordinate
                      limits in interpolation
    '''

    # Load modified gravity curves from simulation output
    Gdata = np.load(theory_data_dir + 'Gravdata.npy')
    yukdata = np.load(theory_data_dir + 'yukdata.npy')
    lambdas = np.load(theory_data_dir + 'lambdas.npy')
    xpos = np.load(theory_data_dir + 'xpos.npy')
    ypos = np.load(theory_data_dir + 'ypos.npy')
    zpos = np.load(theory_data_dir + 'zpos.npy')
    
    lambdas = lambdas[::-1]
    yukdata = np.flip(yukdata, 0)

    # Find limits to avoid out of range erros in interpolation
    xlim = (np.min(xpos), np.max(xpos))
    ylim = (np.min(ypos), np.max(ypos))
    zlim = (np.min(zpos), np.max(zpos))

    # Build interpolating functions for regular gravity
    g_fx_func = interp.RegularGridInterpolator((xpos, ypos, zpos), Gdata[:,:,:,0])
    g_fy_func = interp.RegularGridInterpolator((xpos, ypos, zpos), Gdata[:,:,:,1])
    g_fz_func = interp.RegularGridInterpolator((xpos, ypos, zpos), Gdata[:,:,:,2])

    # Build interpolating functions for yukawa-modified gravity
    yuk_fx_funcs = []
    yuk_fy_funcs = []
    yuk_fz_funcs = []
    for lambind, yuklambda in enumerate(lambdas):
        fx_func = interp.RegularGridInterpolator((xpos, ypos, zpos), yukdata[lambind,:,:,:,0])
        fy_func = interp.RegularGridInterpolator((xpos, ypos, zpos), yukdata[lambind,:,:,:,1])
        fz_func = interp.RegularGridInterpolator((xpos, ypos, zpos), yukdata[lambind,:,:,:,2])
        yuk_fx_funcs.append(fx_func)
        yuk_fy_funcs.append(fy_func)
        yuk_fz_funcs.append(fz_func)

    gfuncs = [g_fx_func, g_fy_func, g_fz_func]
    yukfuncs = [yuk_fx_funcs, yuk_fy_funcs, yuk_fz_funcs]
    lims = [xlim, ylim, zlim]

    return np.array(gfuncs), np.array(yukfuncs), \
            np.array(lambdas), np.array(lims)

def ptarr(xarr, yarr, zarr):
    '''Generates the correctly shaped array to be taken by the interpolating 
       functions from the (n, 1) dimensional position arrays'''
    return np.stack((xarr, yarr, zarr), axis = -1)
    

gfuncs, yukfuncs, lambdas, lims = build_mod_grav_funcs(theory_data_dir)

lam25umind = np.argmin((lambdas-2.5e-5)**2)
 
xpltarr = np.arange(lims[0][0], lims[0][1], 1e-7)
ypltarr = np.arange(-4E-5, 4E-5, 1e-7)
zpltarr = np.arange(lims[2][0], lims[2][1], 1e-7)

ones = np.ones_like(ypltarr)
pts = ptarr(2.5e-5*ones, ypltarr, 0.*ones)

#plt.plot(ypltarr, yukfuncs[0][lam25umind](pts))
#plt.xlabel("displacement [m]")
#plt.ylabel("Fx[N]")
#plt.show()

