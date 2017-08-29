import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Image
import cv2
import os
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
import glob
import re
from scipy import stats
import bead_util as bu
from mpl_toolkits.mplot3d import Axes3D
import itertools

calib_image_path  = "/data/20170822/image_calibration/image_grid"
align_file = "/calibrations/image_alignments/stage_position_20170822.npy"
cal_out_file = "/calibrations/image_calibrations/stage_polynomial_20170822.npy"
imfile =  "/data/20170822/image_calibration/image_grid/trap_40um_40um_corner_stage-X9um-Y80um-Z0um.h5.npy"


def get_first_edge(row, l_ind, h_ind):
    #gets index of first non zero element in a vector
    if all(i==0 for i in row):
        return -1.
    else:
        edge_inds = np.nonzero(row)[0]
        b = (edge_inds>l_ind) & (edge_inds<h_ind)
        if len(edge_inds[b])>0: 
            return np.min(edge_inds[b])
        else:
            return -1.

def edge_in_range(edges, l_ind, h_ind):
    #given all of the edges finds the left most edge in a given range.
    s = np.shape(edges)
    l_edge = np.zeros(s[0])
    for i, e in enumerate(l_edge):
        l_edge[i] = get_first_edge(edges[i], l_ind, h_ind)

    return l_edge


def find_l_edge(edges, edge_width = 10., make_plt = False):
    #finds the coordinates of the left edge of a cantilever given a set of canny edges
    #plt.imshow(edges)
    #plt.show()
    shape = np.shape(edges)
    l_ind0 = 0.
    h_ind0 = shape[0]
    l_edge = edge_in_range(edges, l_ind0, h_ind0) 
    ed = np.median(l_edge[l_edge>-1.])
    l_edge = edge_in_range(edges, ed-edge_width/2., ed+edge_width/2.) 
    b = l_edge>0.
    x_edges = l_edge[b]
    y_edges = np.arange(shape[0])[b]
    if make_plt:
        plt.imshow(edges)
        plt.plot(x_edges, y_edges)
        plt.show()
    return x_edges, y_edges

def parab(x, a, b, c):
    #linear function for fitting
    return a*x**2 + b*x + c

def line(x, m, b):
    '''linear function for fitting.'''
    return m*x + b

def parab_int(pz, py):
    # finds intersection of 2 parabolas where z=parab(y, *pz) and y=parab(z, *py). Returns smallest real positive intersection.
    poly = [pz[0]*py[0]**2, 2.*pz[0]*py[0]*py[1], 2.*pz[0]*py[0]*py[2] + pz[0]*py[1]**2 + pz[1]*py[0], 2.*pz[0]*py[1]*py[2] + pz[1]*py[1] - 1., pz[0]*py[2]**2 + pz[1]*py[2] + pz[2]]
    zs = np.roots(poly)
    realb = np.isreal(zs) #only take real solutions
    zs = zs[realb]
    ys = parab(zs, *py)
    rs = ys**2 + zs**2
    minr = np.argmin(rs)
    return np.array([zs[minr], ys[minr]])

    

def measure_cantilever(fpath, fun = line, make_plot = False, plot_edges = False, thresh1 = 600, thresh2 = 700, app_width = 5, auto_thresh = True, nfit = 100, filt_size = 3):
    #measures pixel coordinates of the corner of the cantilever by fitting the edges of the cantilever to fun
    #import
    f, f_ext = os.path.splitext(fpath)
    #print f
    if f_ext == '.bmp':
    	img = cv2.imread(fpath, 0)
    if f_ext == '.npy':
        img = np.load(fpath)
    kern = np.ones((filt_size, filt_size), np.float32)/filt_size
    img_f = cv2.filter2D(img, -1, kern)
    shape = np.array(np.shape(img_f))
    #canny edge detect
    edges = np.zeros_like(img_f)
    cv2.Canny(img_f, thresh1, thresh2, edges, app_width)
    if plot_edges:
        plt.imshow(edges)
        plt.show()
    x_edges_z, y_edges_z = find_l_edge(edges)
    y_edges_y, x_edges_y = find_l_edge(np.transpose(edges))#transpose and flip output x->y, y->x to find top edge 

    #fit edge
    popt_z, pcov_z = curve_fit(fun, y_edges_z[:nfit], x_edges_z[:nfit])
    popt_y, pcov_y = curve_fit(fun, x_edges_y[:nfit], y_edges_y[:nfit])
    xplt = np.arange(shape[0])
    yplt = np.arange(shape[1])

    #find corner
    if fun == line:
        popt_z = np.hstack(([0.], popt_z))
        popt_y = np.hstack(([0], popt_y))
    	corn_coords = parab_int(popt_z, popt_y)#set x^2 coeff to 0
    else:
        corn_coords = parab_int(popt_z, popt_y)    


    if make_plot:
        plt.plot(parab(xplt, *popt_z), xplt, 'r')
        plt.plot(yplt, parab(yplt, *popt_y), 'k')
        plt.plot(x_edges_z[:nfit], y_edges_z[:nfit], 'w')
        plt.plot(x_edges_y[:nfit], y_edges_y[:nfit], 'w')
        plt.plot([corn_coords[0]], [corn_coords[1]], 'xy', ms = 20, mew = 2)
        plt.xlabel("x[pixels]")
        plt.ylabel("y[pixels]")
        plt.imshow(img_f)
        plt.show()
    
    return np.array(np.real(corn_coords))


    
def get_xydistance(im_fname, cols = {'x_stage':17, 'y_stage':18}, dat_ext = '.h5'):
    #extracts the mean stage position from the data associated with an image file name. returns a numpy array [x, y]
    fname, fext = os.path.splitext(im_fname)
    #print fname
    dat, attribs, f = bu.getdata(fname)#works with ext .h5.npy on im_fname
    f.close() 
    x =  np.mean(dat[:, cols['x_stage']])
    y = np.mean(dat[:, cols['y_stage']])
    return np.array([x, y])


def get_distances(files, align_file, cant_cal = 8.0, ext = '.npy'):
    #gets all of the distances associated with a list of image files from the associated data file.
    cal = np.load(align_file)
    ds = np.array(map(get_xydistance, files))*cant_cal
    ds = -1.*(np.array(ds) + cal)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return np.array(ds)

def ignore_point(vec, p):
    #broduces a boolian vector with ones for every point except points taking value p
    return np.array([i!=p for i in vec])

def quad_roots(popt):
    #the quadratic equation for a*x^2 + b*x + c = 0 when popt = [a, b, c]. Returns the smallest positive real root.
    rs = np.array([(-1.*popt[1] + np.sqrt(popt[1]**2-4.*popt[0]*popt[2]))/(2.*popt[0]), (-1.*popt[1] - np.sqrt(popt[1]**2-4.*popt[0]*popt[2]))/(2.*popt[0])])
    real = np.isreal(rs)
    return np.min(rs[rs[real]>= 0.])


def find_intercept(can_pos, ds, idx, make_plot = True, make_error_plot = False):
    #gets z and y intercept from fitting measured displacement to a parabola and calculating the intercept
    if idx == 0:
        o_idx = 1
        direction = 'z'
    else:
        o_idx = 0
        direction = 'y'
    ps = ds[:, idx]
    orth_pos = stats.mode(ds[:, idx])
    bo = ignore_point(ds[:, idx], orth_pos[0][0])
    if idx == 0:
        popt, pcov = curve_fit(parab, ds[bo, idx], can_pos[bo, idx])
        d0 =  quad_roots(popt)
    if make_plot:
        ds_plot = np.arange(np.min(ds[bo, idx]), np.max(ds[bo, idx]), .01)
        plt.plot(ds[bo, idx], can_pos[bo, idx], 'o')
        plt.plot(ds_plot, parab(ds_plot, *popt), label = 'intercept=' + str(np.round(d0, decimals = 2)))
        plt.xlabel("stage %s position [$\mu m$]" % direction)
        plt.ylabel("measured distance [pixels]")
        plt.legend()
        plt.show()
    if make_error_plot:
        plt.plot(ds[bo, idx], parab(ds[bo, idx], *popt)-can_pos[bo, idx], 'o')
        plt.xlabel("stage z position [$\mu m$]")
        plt.ylabel("error [pixels]")
        plt.show()

    return d0

def stage_pos_fun(can_pos, ds, cal_file, make_plot = True):
    #fits measured stage position in pixels vs 'true' stage position in microns to quadratic. Given a measured stage position this returns the 'true' stage position relative to the trap.
    popt_z, pcov_z = curve_fit(parab, can_pos[:, 0], ds[:, 0]) 
    popt_y, pcov_y = curve_fit(parab, can_pos[:, 1], ds[:, 1])
    out_poly = np.array([popt_z, popt_y])
    np.save(cal_file, out_poly)
    if make_plot:
        fit_z = np.linspace(np.min(can_pos[:, 0]), np.max(can_pos[:, 0]), 100)
        fit_y = np.linspace(np.min(can_pos[:, 1]), np.max(can_pos[:, 1]), 100)
        plt.plot(can_pos[:, 0], ds[:, 0], 'o', label = "z")
        plt.plot(can_pos[:, 1], ds[:, 1], 'o', label = "y")
        plt.plot(fit_z, parab(fit_z, *popt_z))
        plt.plot(fit_y, parab(fit_y, *popt_y))
        plt.xlabel("measured displacement from trap [pixels]")
        plt.ylabel("displacement from trap [$\mu m$]")
        plt.legend()
        plt.show()
 


def polyfit2d(x, y, z, order = 2):
    '''fits 2d surfaces to polynomial of order order'''
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i*y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    '''generates z vector from x, y and polynomial vector m'''
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def plot_3d_calib(can_pos, ds, axis, polyfit, stage_cal = 8.0):
    '''plots the pixel corner coordinates as a function of stage xy position'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(can_pos[:, 0], can_pos[:, 1], ds[:, axis], marker = 'o')  
    ax.set_xlabel('x pixel')
    ax.set_ylabel('y pixel')
    if axis==0:
        lab = 'stage x [um]'
    if axis==1:
        lab = 'stage y [um]'
    ax.set_zlabel(lab)
    ax.scatter(can_pos[:, 0], can_pos[:, 1], polyval2d(can_pos[:, 0], can_pos[:, 1], polyfit), marker = '^')
         

def get_calibration(img_cal_path, align_file, cal_out_file, image_ext = '.npy', cant_cal = 8.0):
    '''Does all of the steps to get the calibration of cantilever images and saves the result.'''
    fs = glob.glob(img_cal_path + '/*' + image_ext)
    ds = get_distances(fs, align_file)
    can_pos = np.array(map(measure_cantilever, fs))
    align = np.load(align_file)
    print ds
    print align
    ds[:, 0] += align[0]
    ds[:, 1] += align[1]
    mx = polyfit2d(can_pos[:, 0], can_pos[:, 1], ds[:, 0])
    my = polyfit2d(can_pos[:, 0], can_pos[:, 1], ds[:, 1])
    cal_arr = np.array([mx, my])
    directory = os.path.dirname(cal_out_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(cal_out_file, cal_arr)
    return ds, can_pos

def measure_image(im_file, cal_out_file, make_plot = True):
    #given an image file returns the calibrated coordinates of the cantileve edge
    pixels = measure_cantilever(im_file, make_plot = make_plot)
    cal = np.load(cal_out_file)
    
    px = polyval2d(pixels[0], pixels[1], cal[0])
    py = polyval2d(pixels[0], pixels[1], cal[1])
    return np.array([px, py])

get_calibration(calib_image_path, align_file, cal_out_file)    
