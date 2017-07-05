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


path  = "/data/20170704/profiling/images"
cal_file = "/home/arider/opt_lev_analysis/calibrations/20170704/stage_position.npy"
out_file = "/home/arider/opt_lev_analysis/calibrations/20170704/stage_polynomial.npy"

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


def find_l_edge(edges, edge_width = 6.):
    #finds the coordinates of the left edge of a cantilever given a set of canny edges
    shape = np.shape(edges)
    l_ind0 = 0.
    h_ind0 = shape[0]
    l_edge = edge_in_range(edges, l_ind0, h_ind0) 
    ed = np.median(l_edge)
    l_edge = edge_in_range(edges, ed-edge_width/2., ed+edge_width/2.) 
    b = l_edge>0.
    x_edges = l_edge[b]
    y_edges = np.arange(shape[0])[b]
    return x_edges, y_edges

def parab(x, a, b, c):
    #linear function for fitting
    return a*x**2 + b*x + c

def ex(arr, xinds, yinds):
    #finds the expected value of an array.
    s = sum(arr.flatten())
    ex_x = np.sum(np.einsum('i, ij->j', xinds, arr))/s
    ex_y = np.sum(np.einsum('j, ij->i', yinds, arr))/s
    return ex_x, ex_y

def find_beam(img, box = 10):
    #finds the center of the beam as the centroid about the maximum value within box.
    lim = int(box/2.)
    max_ind = np.argmax(img)
    max_coords =  np.unravel_index(max_ind, np.shape(img))
    xinds_r = (max_coords[0] - lim, max_coords[0] + lim)
    yinds_r = (max_coords[1] - lim, max_coords[1] + lim)
    arr = img[xinds_r[0]:xinds_r[1], yinds_r[0]:yinds_r[1]]
    xinds = np.arange(xinds_r[0], xinds_r[1])
    yinds = np.arange(yinds_r[0], yinds_r[1])
    return (ex(arr, xinds, yinds))

def closest_point(p, fun, popt, tol = 0.0001):
    #closest distance between function with parameters popt and point
    fun2 = lambda x: (p[0]-x)**2 + (p[1]-fun(x, *popt))**2
    res = minimize_scalar(fun2)
    return np.sqrt(fun2(res.x))

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
    

def measure_cantilever(fpath, fun = parab, make_plot = False, plot_edges = False, thresh1 = 150, thresh2 = 250, app_width = 5):
    #measures the position of the cantilever with respect to the beam by fitting the edges of the cantilever to fun
    #import
    img = cv2.imread(fpath, 0)
    shape = np.array(np.shape(img))
    #canny edge detect
    edges = np.zeros_like(img)
    cv2.Canny(img, thresh1, thresh2, edges, app_width)
    if plot_edges:
        plt.imshow(edges)
        plt.show()
    x_edges_z, y_edges_z = find_l_edge(edges)
    y_edges_y, x_edges_y = find_l_edge(np.transpose(edges))#transpose and flip output x->y, y->x to find top edge 

    #fit edge
    popt_z, pcov_z = curve_fit(parab, y_edges_z, x_edges_z)
    popt_y, pcov_y = curve_fit(parab, x_edges_y, y_edges_y)
    xplt = np.arange(shape[0])
    yplt = np.arange(shape[1])

    #find corner
    corn_coords = parab_int(popt_z, popt_y)
    
    if make_plot:
        plt.plot(parab(xplt, *popt_z), xplt, 'r')
        plt.plot(yplt, parab(yplt, *popt_y), 'k')
        plt.plot(x_edges_z, y_edges_z, 'w')
        plt.plot(x_edges_y, y_edges_y, 'w')
        plt.plot([corn_coords[0]], [corn_coords[1]], 'xy', ms = 20, mew = 2)
        plt.xlabel("z[pixels]")
        plt.ylabel("y[pixels]")
        plt.imshow(img)
        plt.show()
    
    return np.real(corn_coords)

def get_zydistance(string, unit = 'um'):
    #takes a string and gets the number following z before um as z and the number following y before um as y. returns a numpy array [z, y]
    z = float(re.findall(r'\d+', re.findall(r'z\d+um', string)[0])[0])
    y = float(re.findall(r'\d+', re.findall(r'y\d+um', string)[0])[0])
    return np.array([z, y])


def get_distances(path, cal_file, ext = '.bmp'):
    #gets all of the image  files and the stage settings cludgily encoded in the file name.
    cal = np.load(cal_file)
    files  = glob.glob(path + '/*' + ext)
    ds = map(get_zydistance, files)
    ds = -1.*(np.array(ds) + cal)
    return files, np.array(ds)

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

def stage_pos_fun(can_pos, ds, outfile, make_plot = True):
    #fits measured stage position in pixels vs 'true' stage position in microns to quadratic. Given a measured stage position this returns the 'true' stage position relative to the trap.
    popt_z, pcov_z = curve_fit(parab, can_pos[:, 0], ds[:, 0]) 
    popt_y, pcov_y = curve_fit(parab, can_pos[:, 1], ds[:, 1])
    out_poly = np.array([popt_z, popt_y])
    np.save(outfile, out_poly)
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
 
def do_calibration(path, cal_file, outfile):
    #Does all of the steps to get the calibration of cantilever images and saves the result.
    fs, ds = get_distances(path, cal_file)
    can_pos = np.array(map(measure_cantilever, fs))
    stage_pos_fun(can_pos, ds, outfile)


def measure_image(im_file, cal_file)
