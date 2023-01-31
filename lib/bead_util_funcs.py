import h5py, os, re, glob, time, sys, fnmatch, inspect, colorsys
import subprocess, math, xmltodict, traceback
import numpy as np
import datetime as dt
import dill as pickle 

import matplotlib
backends = ['MacOSX', 'Qt5Agg', 'GTK3Agg', 'TkAgg', 'Agg']
for backend in backends:
    try:
        matplotlib.use(backend)
        break
    except:
        continue

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.mlab as mlab

import scipy.interpolate as interpolate
import scipy.optimize as optimize
import scipy.signal as signal
import scipy.special as special
import scipy.stats as stats
import scipy.constants as constants
import scipy

from iminuit import Minuit, describe
from obspy.signal.detrend import polynomial
from tqdm import tqdm
from joblib import Parallel, delayed

import configuration

import warnings

from bead_data_funcs import get_hdf5_time

from stats_util import *

#######################################################
# This module has basic utility functions for analyzing bead
# data. In particular, this module has the basic data
# loading function, file finding/sorting, colormapping,
# FFT normalization, spatial binning etc.
# 
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
# ----------------------------------------------------
###
### The DataFile class is stored in a companion module
### bead_util, which imports these helper functions
### This module imports a number of functions from a 
### module 'fpga_data_funcs' which handles the raw data
### from the hdf5 files and ports them into python objects
###
# ----------------------------------------------------
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
#
# This version has been significantly trimmed from previous
# bead_util in an attempt to force modularization.
# Previous code for millicharge and chameleon data
# can be found by reverting opt_lev_analysis
#######################################################


my_path = os.path.abspath( os.path.dirname(__file__) )


#calib_path = '/data/old_trap_processed/calibrations/'
calib_path = os.path.abspath( os.path.join(my_path, '../data/') )

e_top_dat   = np.loadtxt(os.path.join(calib_path, 'e-top_1V_optical-axis.txt'), comments='%')
e_bot_dat   = np.loadtxt(os.path.join(calib_path, 'e-bot_1V_optical-axis.txt'), comments='%')
e_left_dat  = np.loadtxt(os.path.join(calib_path, 'e-left_1V_left-right-axis.txt'), comments='%')
e_right_dat = np.loadtxt(os.path.join(calib_path, 'e-right_1V_left-right-axis.txt'), comments='%')
e_front_dat = np.loadtxt(os.path.join(calib_path, 'e-front_1V_front-back-axis.txt'), comments='%')
e_back_dat  = np.loadtxt(os.path.join(calib_path, 'e-back_1V_front-back-axis.txt'), comments='%')

E_front  = interpolate.interp1d(e_front_dat[0], e_front_dat[-1])
E_back   = interpolate.interp1d(e_back_dat[0],  e_back_dat[-1])
E_right  = interpolate.interp1d(e_right_dat[1], e_right_dat[-1])
E_left   = interpolate.interp1d(e_left_dat[1],  e_left_dat[-1])
E_top    = interpolate.interp1d(e_top_dat[2],   e_top_dat[-1])
E_bot    = interpolate.interp1d(e_bot_dat[2],   e_bot_dat[-1])




e_xp_dat = np.loadtxt(os.path.join(calib_path, 'new-trap_efield-x_+x-elec-1V_x-axis.txt'), comments='%').transpose()
e_xn_dat = np.loadtxt(os.path.join(calib_path, 'new-trap_efield-x_-x-elec-1V_x-axis.txt'), comments='%').transpose()
e_yp_dat = np.loadtxt(os.path.join(calib_path, 'new-trap_efield-y_+y-elec-1V_y-axis.txt'), comments='%').transpose()
e_yn_dat = np.loadtxt(os.path.join(calib_path, 'new-trap_efield-y_-y-elec-1V_y-axis.txt'), comments='%').transpose()
e_zp_dat = np.loadtxt(os.path.join(calib_path, 'new-trap_efield-z_+z-elec-1V_z-axis.txt'), comments='%').transpose()
e_zn_dat = np.loadtxt(os.path.join(calib_path, 'new-trap_efield-z_-z-elec-1V_z-axis.txt'), comments='%').transpose()

E_xp  = interpolate.interp1d(e_xp_dat[0], e_xp_dat[-1])
E_xn  = interpolate.interp1d(e_xn_dat[0], e_xn_dat[-1])
E_yp  = interpolate.interp1d(e_yp_dat[1], e_yp_dat[-1])
E_yn  = interpolate.interp1d(e_yn_dat[1], e_yn_dat[-1])
E_zp  = interpolate.interp1d(e_zp_dat[2], e_zp_dat[-1])
E_zn  = interpolate.interp1d(e_zn_dat[2], e_zn_dat[-1])

# plt.figure()
# plt.plot(e_front_dat[0], e_front_dat[-1])
# plt.plot(e_back_dat[0], e_back_dat[-1])
# plt.figure()
# plt.plot(e_right_dat[1], e_right_dat[-1])
# plt.plot(e_left_dat[1], e_left_dat[-1])
# plt.figure()
# plt.plot(e_top_dat[2], e_top_dat[-1])
# plt.plot(e_bot_dat[2], e_bot_dat[-1])
# plt.show()



def gauss(x, A, mu, sigma, c):
    return A * np.exp( -1.0*(x-mu)**2 / (2 * sigma**2) ) + c

def line(x, a, b):
    return a*x + b





#### Generic Helper functions


def iterable(obj):
    '''
    Simple function to check if an object is iterable. Helps with
    exception handling when checking arguments in other functions.
    '''
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True



def progress_bar(count, total, suffix='', bar_len=50, newline=True):
    '''Prints a progress bar and current completion percentage.
       This is useful when processing many files and ensuring
       a script is actually running and going through each file

           INPUTS: count, current counting index
                   total, total number of iterations to complete
                   suffix, option string to add to progress bar
                   bar_len, length of the progress bar in the console

           OUTPUTS: none
    '''
    
    if len(suffix):
        max_bar_len = 80 - len(suffix) - 17
        if bar_len > max_bar_len:
            bar_len = max_bar_len

    if count == total - 1:
        percents = 100.0
        bar = '#' * bar_len
    else:
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '#' * filled_len + '-' * (bar_len - filled_len)
    
    # This next bit writes the current progress bar to stdout, changing
    # the string slightly depending on the value of percents (1, 2 or 3 digits), 
    # so the final length of the displayed string stays constant.
    if count == total - 1:
        sys.stdout.write('[%s] %s%s ... %s\r' % (bar, percents, '%', suffix))
    else:
        if percents < 10:
            sys.stdout.write('[%s]   %s%s ... %s\r' % (bar, percents, '%', suffix))
        else:
            sys.stdout.write('[%s]  %s%s ... %s\r' % (bar, percents, '%', suffix))

    sys.stdout.flush()
    
    if (count == total - 1) and newline:
        print()






def lighten_color(color, fac=1.0):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)

    From:
    https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
    """
    try:
        c = colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - fac * (1 - c[1]), c[2])





def round_sig(x, sig=2):
    '''Round a number to a certain number of sig figs

           INPUTS: x, number to be rounded
                   sig, number of sig figs

           OUTPUTS: num, rounded number'''

    neg = False
    if x == 0:
        return 0
    else:
        if x < 0:
            neg = True
            x = -1.0 * x
        num = round(x, sig-int(math.floor(math.log10(x)))-1)
        if neg:
            return -1.0 * num
        else:
            return num





def format_multiple_float_string(*args, sig_figs=3, extra=0):
    '''
    Takes an arbitrary number of floats as non-keyword arguments,
    rounds each to a certain number of significant figures, formats
    the result as a float (i.e. no scientific notation), then builds 
    a single string of the values as a comma-separated list, adding
    extra white space between values so that they take up the same
    horizontal space and multiple float strings can be aligned in 
    a sort of column like format.
    '''
    out_str = ''
    for arg_ind, arg in enumerate(args):
        if iterable(sig_figs):
            u_sig_figs = sig_figs[arg_ind]
        else:
            u_sig_figs = sig_figs

        length = int(u_sig_figs + 2 + extra)
        arg_str = np.format_float_positional( \
                    round_sig(arg, u_sig_figs), trim='-')
        arg_len = len(arg_str)
        for i in range( int(length-arg_len) ):
            out_str += ' '
        out_str += arg_str
        if arg != args[-1]:
            out_str += ', '
    return out_str



def weighted_mean(vals, errs, correct_dispersion=False):
    '''Compute the weighted mean, and the standard error on the weighted mean
       accounting for for over- or under-dispersion

           INPUTS: vals, numbers to be averaged
                   errs, nuncertainty on those numbers
                   correct_dispersion, scale variance by chi^2

           OUTPUTS: mean, mean_err'''
    vals = np.array(vals)
    errs = np.array(errs)

    if len(vals) != len(errs):
        raise ValueError

    if len(vals) == 1:
        return vals[0], errs[0]

    variance = errs**2
    weights = 1.0 / variance
    mean = np.sum(weights * vals) / np.sum(weights)
    mean_err = np.sqrt( 1.0 / np.sum(weights) )
    chi_sq = (1.0 / (len(vals) - 1)) * np.sum(weights * (vals - mean)**2)
    if correct_dispersion:
        mean_err *= chi_sq
    return mean, mean_err




def get_scivals(num, base=10.0):
    '''Return a tuple with factor and base X exponent of the input number.
       Useful for custom formatting of scientific numbers in labels.

           INPUTS: num, number to be decomposed
                   base, arithmetic base, assumed to be 10 for most

           OUTPUTS: tuple, (factor, base-X exponent)
                        e.g. get_scivals(6.32e11, base=10.0) -> (6.32, 11)
    '''
    exponent = np.floor(np.log10(num) / np.log10(base))
    return ( num / (base ** exponent), int(exponent) )




def fft_norm(N, fsamp):
    '''Factor to normalize FFT to ASD units'''
    return np.sqrt(2.0 / (N * fsamp))




def remove_outliers(data, m=2.0, return_mask=False):
    '''Take a data array and chop out the outliers. Alternatively returns
       a boolean array that can be used to index and/or mask.
    '''
    dist = np.abs(data - np.median(data))
    median_dist = np.median(dist)
    s = dist / (median_dist if median_dist else 1.)
    mask = s < m
    if return_mask:
        return mask
    else:
        return data[mask]







#### First define some functions to help with the DataFile object. 

def count_dirs(path):
    '''Counts the number of directories (and subdirectories)
       in a given path.

       INPUTS: path, directory name to loop over

       OUTPUTS: numdir, number of directories and subdirectories
                        in the given path'''

    count = 0
    for root, dirs, files in os.walk(path):
        count += len(dirs)

    return count
    

def make_all_pardirs(path, confirm=True):
    '''Function to help pickle from being shit. Takes a path
       and looks at all the parent directories etc and tries 
       making them if they don't exist.

       INPUTS: path, any path which needs a hierarchy already 
                     in the file system before being used

       OUTPUTS: none
       '''

    parts = path.split('/')
    parent_dir = '/'
    for ind, part in enumerate(parts):
        if ind == 0 or ind == len(parts) - 1:
            continue
        parent_dir += part
        parent_dir += '/'
        if not os.path.isdir(parent_dir):
            if confirm:
                print()
                print('Make this directory?')
                print(f'    {parent_dir}')
                print()
                answer = input('(Y/N): ')
            else:
                answer = 'Y'

            if ('y' in answer) or ('Y' in answer):
                os.mkdir(parent_dir)





def count_subdirectories(dirname):
    '''Simple function to count subdirectories.'''
    count = 0
    for root, dirs, files in os.walk(dirname):
        count += len(dirs)
    return count





def find_all_fnames(dirlist, ext='.h5', sort=False, exclude_fpga=True, \
                    verbose=True, substr='', subdir='', sort_time=False, \
                    sort_by_index=False, use_origin_timestamp=False, \
                    skip_subdirectories=False):
    '''Finds all the filenames matching a particular extension
       type in the directory and its subdirectories .

       INPUTS: 
            dirlist, list of directory names to loop over

            ext, file extension you're looking for
            
            sort, boolean specifying whether to do a simple sort
            
            exclude_fpga, boolean to ignore hdf5 files directly from
                the fpga given that the classes load these automatically
               
            verbose, print shit
            
            substr, string within the the child filename to match
            
            subdir, string within the full parent director to match
               
            sort_time, sort files by timestamp, trying to use the hdf5 
                timestamp first

            use_origin_timestamp, boolean to tell the time sorting to
                use the file creation timestamp itself

            skip_subdirectories, boolean to skip recursion into child
                directories when data is organized poorly

       OUTPUTS: 
            files, list of filenames as strings, or list of lists if 
                multiple input directories were given

            lengths, length of file lists found '''

    if verbose:
        print("Finding files in: ")
        print(dirlist)
        sys.stdout.flush()

    was_list = True

    lengths = []
    files = []

    if type(dirlist) == str:
        dirlist = [dirlist]
        was_list = False

    for dirname in dirlist:
        for root, dirnames, filenames in os.walk(dirname):
            if (root != dirname) and skip_subdirectories:
                continue
            for filename in fnmatch.filter(filenames, '*' + ext):
                if ('_fpga.h5' in filename) and exclude_fpga:
                    continue
                if substr and (substr not in filename):
                    continue
                if subdir and (subdir not in root):
                    continue
                files.append(os.path.join(root, filename))
        if was_list:
            if len(lengths) == 0:
                lengths.append(len(files))
            else:
                lengths.append(len(files) - np.sum(lengths)) 

    if len(files) == 0:
        print("DIDN'T FIND ANY FILES :(")
        return [], [0]

    if 'new_trap' in files[0]:
        new_trap = True
    else:
        new_trap = False

    if sort:
        # Sort files based on final index
        files.sort(key = find_str)

    if sort_by_index:
        files.sort(key = lambda x: int(re.findall(r'_([0-9]+)\.', x)[0]) )

    if sort_time:
        files = sort_files_by_timestamp(files, use_origin_timestamp=use_origin_timestamp, \
                                        new_trap=new_trap)

    if verbose:
        print("Found %i files..." % len(files))
    if was_list:
        return files, lengths
    else:
        return files, [len(files)]



def sort_files_by_timestamp(files, use_origin_timestamp=False, new_trap=False):
    '''Pretty self-explanatory function.'''

    if not use_origin_timestamp:
        try:
            files = [(get_hdf5_time(path, new_trap=new_trap), path) for path in files]
        except Exception:
            print('BAD HDF5 TIMESTAMPS, USING GENESIS TIMESTAMP')
            traceback.print_exc()
            use_origin_timestamp = True

    if use_origin_timestamp:
        files = [(os.stat(path), path) for path in files]
        files = [(stat.st_ctime, path) for stat, path in files]

    files.sort(key = lambda x: x[0])
    files = [obj[1] for obj in files]
    return files



def find_common_filnames(*lists):
    '''Takes multiple lists of files and determines their 
    intersection. This is useful when filtering a large number 
    of files by DC stage positions.'''

    intersection = []
    numlists = len(lists)
    
    lengths = []
    for listind, fillist in enumerate(lists):
        lengths.append(len(fillist))
    longind = np.argmax(np.array(lengths))
    newlists = []
    newlists.append(lists[longind])
    for n in range(numlists):
        if n == longind:
            continue
        newlists.append(lists[n])

    for filname in newlists[0]:
        present = True
        for n in range(numlists-1):
            if len(newlists[n+1]) == 0:
                continue
            if filname not in newlists[n+1]:
                present = False
        if present:
            intersection.append(filname)
    return intersection



def find_str(str):
    '''finds the index from the standard file name format'''
    idx_offset = 1e10 # Large number to ensure sorting by index first

    fname, _ = os.path.splitext(str)
    
    endstr = re.findall("\d+mV_[\d+Hz_]*[a-zA-Z]*[\d+]*", fname)
    if( len(endstr) != 1 ):
        # Couldn't find expected pattern, so return the 
        # second to last number in the string
        return int(re.findall('\d+', fname)[-1])

    # Check for an index number
    sparts = endstr[0].split("_")
    if ( len(sparts) >= 3 ):
        return idx_offset*int(sparts[2]) + int(sparts[0][:-2])
    else:
        return int(sparts[0][:-2])



def angle_between_vectors(vec1, vec2, coord='Cartesian', radians=True):
    '''Computes the angle between two 3d vectors, or two arrays of 3d
       vectors, in either cartesian or spherical-polar coordinates.
    '''

    axis = np.argmax( np.array( np.array(vec1).shape ) == 3 )
    if axis != 0:
        vec1 = np.transpose(vec1)
        vec2 = np.transpose(vec2)

    if (coord == 'Cartesian') or (coord == 'C') or (coord == 'c'):

        v1 = np.copy(vec1)
        v2 = np.copy(vec2)

    elif (coord == 'Spherical') or (coord == 'S') or (coord == 's'):

        v1 = np.array( [vec1[0] * np.sin(vec1[1]) * np.cos(vec1[2]), \
                        vec1[0] * np.sin(vec1[1]) * np.sin(vec1[2]), \
                        vec1[0] * np.cos(vec1[1])] )

        v2 = np.array( [vec2[0] * np.sin(vec2[1]) * np.cos(vec2[2]), \
                        vec2[0] * np.sin(vec2[1]) * np.sin(vec2[2]), \
                        vec2[0] * np.cos(vec2[1])] )

    else:
        print('Coordinate system not understood')
        return np.zeros(vec.shape[1])

    v1_mag = np.sqrt(np.sum(v1 * v1, axis=0))
    v2_mag = np.sqrt(np.sum(v2 * v2, axis=0))

    angle = np.arccos( np.sum(v1 * v2, axis=0) / (v1_mag * v2_mag))

    if not radians:
        angle *= 180 / np.pi

    return angle


def euler_rotation_matrix(rot_angles, radians=True):
    '''Returns a 3x3 euler-rotation matrix. Thus the rotation proceeds
       thetaX (about x-axis) -> thetaY -> thetaZ, with the result returned
       as a numpy ndarray.
    '''


    if not radians:
        rot_angles = (np.pi / 180.0) * np.array(rot_angles)

    rx = np.array([[1.0, 0.0, 0.0], \
                   [0.0, np.cos(rot_angles[0]), -1.0*np.sin(rot_angles[0])], \
                   [0.0, np.sin(rot_angles[0]), np.cos(rot_angles[0])]])

    ry = np.array([[np.cos(rot_angles[1]), 0.0, np.sin(rot_angles[1])], \
                   [0.0, 1.0, 0.0], \
                   [-1.0*np.sin(rot_angles[1]), 0.0, np.cos(rot_angles[1])]])

    rz = np.array([[np.cos(rot_angles[2]), -1.0*np.sin(rot_angles[2]), 0.0], \
                   [np.sin(rot_angles[2]), np.cos(rot_angles[2]), 0.0], \
                   [0.0, 0.0, 1.0]])


    rxy = np.matmul(ry, rx)
    rxyz = np.matmul(rz, rxy)

    return rxyz



def rotate_points(pts, rot_matrix, rot_point, plot=False):
    '''Takes an input of shape (Npts, 3) and applies the
       given rotation matrix to each 3D point. Order of rotations
       follow the Euler convention.
    '''
    npts = pts.shape[0]
    rot_pts = []
    for resp in [0,1,2]:
        rot_pts_vec = np.zeros(npts)
        for resp2 in [0,1,2]:
            rot_pts_vec += rot_matrix[resp,resp2] * (pts[:,resp2] - rot_point[resp2])
        rot_pts_vec += rot_point[resp]
        rot_pts.append(rot_pts_vec)
    rot_pts = np.array(rot_pts)
    rot_pts = rot_pts.T

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
            
        ax.scatter(pts[:,0]*1e6, pts[:,1]*1e6, \
                   pts[:,2]*1e6, label='Original')
        ax.scatter(rot_pts[:,0]*1e6, rot_pts[:,1]*1e6, rot_pts[:,2]*1e6, \
                       label='Rot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return rot_pts




def rotate_meshgrid(xvec, yvec, zvec, rot_matrix, rot_point, \
                    plot=False, microns=True):
    '''From an set of input vectors defining the xcoords, ycoords, and
       zcoords of a meshgrid, rotate the meshgrid about a given point 
       in 3D-Euclidean space. Output can be ploted'''
    xg, yg, zg = np.meshgrid(xvec, yvec, zvec, indexing='ij')
    init_mesh = np.array([xg, yg, zg])
    rot_grids = np.einsum('ij,jabc->iabc', rot_matrix, init_mesh)

    init_pts = np.rollaxis(init_mesh, 0, 4)
    init_pts = init_pts.reshape((init_mesh.size // 3, 3))

    rot_pts = np.rollaxis(rot_grids, 0,4)
    rot_pts = rot_pts.reshape((rot_grids.size // 3, 3))

    if microns:
        fac = 1.0
    else:
        fac = 1e6

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
            
        ax.scatter(init_pts[:,0]*fac, init_pts[:,2]*fac, label='Original')
        ax.scatter(rot_pts[:,0]*fac, rot_pts[:,1]*fac, rot_pts[:,2]*fac, \
                       label='Rot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return rot_grids


def sudo_call(fn, *args):
    with open("/home/charles/some_test.py", "wb") as f:
        f.write( inspect.getsource(fn) )
        f.write( "%s(*%r)" % (fn.__name__,args) )
    out = subprocess.check_output("sudo python /home/charles/some_test.py", shell=True)
    print(out)


def fix_time(fname, dattime):
    '''THIS SCRIPT ONLY WORKS AS ROOT OR A SUDOER. It usually runs
       via the script above, which creates a subroutine. Thus, this 
       function needs to be completely self-sufficient, which is why
       it reimports h5py.'''
    try:
        import h5py
        f = h5py.File(fname, 'r+')
        f['beads/data/pos_data'].attrs.create("time", dattime)
        f.close()
        print("Fixed time.")
    except Exception:
        print("Couldn't fix the time...")
        traceback.print_exc()


def labview_time_to_datetime(lt):
    '''Convert a labview timestamp (i.e. time since 1904) to a  
       more useful format (python datetime object)'''
    
    ## first get number of seconds between Unix time and Labview's
    ## arbitrary starting time
    lab_time = dt.datetime(1904, 1, 1, 0, 0, 0)
    nix_time = dt.datetime(1970, 1, 1, 0, 0, 0)
    delta_seconds = (nix_time-lab_time).total_seconds()

    lab_dt = dt.datetime.fromtimestamp( lt - delta_seconds)
    
    return lab_dt

def unpack_config_dict(dic, vec):
    '''takes vector containing data atributes and puts 
       it into a dictionary with key value pairs specified 
       by dict where the keys of dict give the labels and 
       the values specify the index in vec'''
    out_dict = {}
    for k in list(dic.keys()):
        out_dict[k] = vec[dic[k]]
    return out_dict 



def detrend_poly(arr, order=1.0, plot=False):
    xarr = np.arange( len(arr) )
    fit_model = np.polyfit(xarr, arr, order)
    fit_eval = np.polyval(fit_model, xarr)

    if plot:
        fig, axarr = plt.subplots(2,1,sharex=True, \
                        gridspec_kw={'height_ratios': [1,1]})
        axarr[0].plot(xarr, arr, color='k')
        axarr[0].plot(xarr, fit_eval, lw=2, color='r')
        axarr[1].plot(xarr, arr - fit_eval, color='k')
        fig.tight_layout()
        plt.show()

    return arr - fit_eval




def zerocross_pos2neg(data):
    '''Simple zero-crossing detector for 1D data. Returns indices of 
       crossings as a list.'''
    pos = data > 0
    return (pos[:-1] & ~pos[1:]).nonzero()[0]




def get_sine_amp_phase(sine, int_band=0, freq=0.0, fit=False, \
                       cosine_fit=True, \
                       ncycle_exclude=20, phase_fraction=0.9, \
                       half_width=np.pi/4, incoherent=False, \
                       plot=False, plot_nsamp=10000, verbose=True):
    '''
    I am a docstring.

    '''

    ncycle_exclude = int(ncycle_exclude)
    plot_nsamp = int(plot_nsamp)

    nsamp = len(sine)
    tvec = np.arange(nsamp)

    freqs = np.fft.rfftfreq(nsamp)
    fft = np.fft.rfft(sine) * fft_norm(nsamp, 1.0)

    drive_ind = np.argmax(np.abs(fft[1:])) + 1
    drive_freq_bin = freqs[drive_ind]

    if cosine_fit:
        fit_func = lambda t, a, f, phi, c: a * np.cos(2.0*np.pi*f*t + phi) + c
    else:
        fit_func = lambda t, a, f, phi, c: a * np.sin(2.0*np.pi*f*t + phi) + c

    if fit:
        amp_guess = np.std(sine)*np.sqrt(2)
        phase_guess = np.angle(fft[drive_ind])
        if not cosine_fit:
            phase_guess += np.pi/2.0

        if not freq:
            p0 = [amp_guess, drive_freq_bin, phase_guess, 0.0]
            popt, pcov = optimize.curve_fit(fit_func, tvec, sine, p0=p0)
            phase_unc = np.sqrt(pcov[2,2])
        else:
            p0 = [amp_guess, phase_guess, 0.0]
            popt, pcov = \
                optimize.curve_fit(lambda x,a,b,c: \
                                    fit_func(x,a,drive_freq_bin,b,c), \
                                   tvec, sine, p0=p0)
            popt = np.insert(popt, 1, drive_freq_bin)
            p0 = np.insert(p0, 1, drive_freq_bin)
            phase_unc = np.sqrt(pcov[1,1])

        amp_unc = np.sqrt(pcov[0,0])

    else:
        ### Try to choose an integration band that works for all sampling
        ### conditions, i.e. very large when we're lacking cycles, or 
        ### moderate in other conditions.
        if not int_band:
            if drive_freq_bin < 2.0 / nsamp:
                int_band = 2.0*drive_freq_bin
            else:
                int_band = 0.1*drive_freq_bin
        else:
            if int_band > 1.0:
                raise ValueError('Integration band must be in digital frequency'
                                 + 'units, where Fsamp = 1, so int_band < 0.5')

        int_inds = (freqs > drive_freq_bin - 0.5*int_band) \
                    * (freqs < drive_freq_bin + 0.5*int_band)
        int_freqs =  freqs[int_inds]
        int_fft = fft[int_inds]

        if np.sum(int_inds) <= 10 and verbose:
            print('RuntimeWarning: The number of Fourier bins being' \
                  + ' integrated for amplitude estimation is less than' \
                  + ' ten (< 10). Be wary of your result!... This can' \
                  + ' be due to your bin spacing being large compared to' \
                  + ' the signal frequency, or an overly narrow' \
                  + ' integration band. Maybe it is okay if you know' \
                  + ' what you are doing.')

        weights = np.abs(int_fft)**2 / np.sum(np.abs(int_fft)**2)
        if not freq:
            freq = np.sum(int_freqs * weights)

        amp_int = \
            np.sqrt(2.0 * np.sum((int_fft * int_fft.conj()).real) / nsamp)

        if incoherent:
            npoints_exclude = int(ncycle_exclude / freq)
            tvec_phased = (tvec % (1.0 / freq)) * (2.0 * np.pi * freq)
            tvec_phased_cut = tvec_phased[npoints_exclude:-npoints_exclude]
            sine_cut = sine[npoints_exclude:-npoints_exclude]
            weights = np.abs(sine_cut)

            ### Get phase from weighted mean over some portion of the signal
            ### near the max value
            max_ind = np.argmax(sine_cut)
            max_phase = tvec_phased_cut[max_ind]



            ### Hard-coded number to try to avoid stupid edge effects and 
            ### outliers skewing data
            buffer = 0.1  

            if not half_width:
                half_width = np.arcsin(phase_fraction)

            if ((max_phase - half_width) <= buffer) or \
                        ((max_phase + half_width) >= 2.0*np.pi - buffer):

                ### Move the data away from the edge so averages make sense
                ###    tpcr = tvec_phased_cut_rolled
                tpcr = (tvec_phased_cut + np.pi) % (2.0*np.pi)

                max_inds = np.abs(tpcr - tpcr[max_ind]) < half_width
                phase_at_max = np.sum(tpcr[max_inds] * weights[max_inds] ) \
                                    / np.sum(weights[max_inds])
                phase_at_max = (phase_at_max - np.pi) % (2.0*np.pi)

                if plot:
                    plot_inds = max_inds * (np.arange(len(sine_cut)) < plot_nsamp)
                    fig, ax = plt.subplots(1,1)
                    ax.set_title('Phase Estimation', fontsize=16)
                    ax.scatter(tpcr[:plot_nsamp], sine_cut[:plot_nsamp])
                    ax.scatter(tpcr[plot_inds], sine_cut[plot_inds])
                    ax.axvline(tpcr[max_ind])
                    fig.tight_layout()

            else:
                max_inds = np.abs(tvec_phased_cut - tvec_phased_cut[max_ind]) \
                                < half_width
                phase_at_max = np.sum(tvec_phased_cut[max_inds] * weights[max_inds]) \
                                    / np.sum(weights[max_inds])

                if plot:
                    plot_inds = max_inds * (np.arange(len(sine_cut)) < plot_nsamp)
                    fig, ax = plt.subplots(1,1)
                    ax.set_title('Phase Estimation', fontsize=16)
                    ax.scatter(tvec_phased_cut[:plot_nsamp], sine_cut[:plot_nsamp])
                    ax.scatter(tvec_phased_cut[plot_inds], sine_cut[plot_inds])
                    ax.axvline(tpcr[max_ind])
                    fig.tight_layout()

            ### Compute the actual value of the waveform phase based on the 
            ### location of the max in the modular 2pi*radian window                        
            phase = 2.0 * np.pi - phase_at_max

        else:
            phase = np.angle(fft[drive_ind])

        if not cosine_fit:
            phase += np.pi /2

        popt = [amp_int, freq, phase, np.mean(sine)]

        amp_unc = np.std(np.fft.irfft(fft - fft*int_inds))

        ### Fix this later. Pretty derpy calculation assuming our ability to 
        ### estimate phase scales purely with FFT bin spacing and the number
        ### waveform cycles present in the data. Probably a reasonable lower
        ### limit, but doesn't account for added uncertainty from amplitude
        ### fluctuations.
        # phase_unc = 0.0
        Ncycles = np.max([int(tvec[-1] * freq), 1.0])
        phase_unc = 2.0 * np.pi * freq / np.sqrt(Ncycles)

    if plot:
        fig2, ax2 = plt.subplots(1,1)
        ax2.set_title('Recorded Data vs. Inferred Sine', fontsize=16)
        ax2.plot(tvec[:plot_nsamp], sine[:plot_nsamp])
        ax2.plot(tvec[:plot_nsamp], fit_func(tvec[:plot_nsamp], *popt), \
                 ls='--', lw=2, color='r')
        if fit:
            ax2.plot(tvec[:plot_nsamp], fit_func(tvec[:plot_nsamp], *p0), \
                     ls='--', lw=2, color='r', alpha=0.5)
        fig2.tight_layout()

        plt.show()

        input()

    return popt[0], amp_unc, popt[2], phase_unc





def spatial_bin(drive, resp, dt, nbins=100, nharmonics=10, harms=[], \
                width=0, sg_filter=False, sg_params=[3,1], verbose=True, \
                maxfreq=2500, add_mean=False, correct_phase_shift=False, \
                grad_sign=0, plot=False):
    '''Given two waveforms drive(t) and resp(t), this function generates
       resp(drive) with a fourier method. drive(t) should be a pure tone,
       such as a single frequency cantilever drive (although the 
       existence of harmonics is fine). Ideally, the frequency with
       the dominant power should lie in a single DTFT bin.
       Behavior of this function is somewhat indeterminant when there
       is significant spectral leakage into neighboring bins.

       INPUT:   drive, single frequency drive signal, sampled with some dt
       	        resp, arbitrary response to be 'binned'
       	        dt, sample spacing in seconds [s]
                nbins, number of samples in the final resp(drive)
       	        nharmonics, number of harmonics to include in filter
                harms, list of desired harmonics (overrides nharmonics)
       	        width, filter width in Hertz [Hz]
                sg_filter, boolean value indicating use of a Savitsky-Golay 
                            filter for final smoothing of resp(drive)
                sg_params, parameters of the savgol filter 
                            (see scipy.signal.savgol_filter for explanation)
                verbose, usual boolean switch for printing
                maxfreq, top-hat filter cutoff
                add_mean, boolean switch toadd back the mean of each signal
                correct_phase_shift, boolean switch to adjust the phase of 
                                      of the response to match the drive
                grad_sign, -1, 0 or 1 to indicate the sign of the drive's 
                            derivative to include in order to select either
                            'forward-going' or 'backward-going' data

       OUTPUT:  drivevec, vector of drive values, monotonically increasing
                respvec, resp as a function of drivevec'''


    nsamp = len(drive)
    if len(resp) != nsamp:
        if verbose:
            print("Data Error: x(t) and f(t) don't have the same length")
            sys.stdout.flush()
        return

    ### Generate t array
    t = np.linspace(0, len(drive) - 1, len(drive)) * dt

    ### Generate FFTs for filtering
    drivefft = np.fft.rfft(drive)
    respfft = np.fft.rfft(resp)
    freqs = np.fft.rfftfreq(len(drive), d=dt)

    ### Find the drive frequency, ignoring the DC bin
    maxind = np.argmin( np.abs(freqs - maxfreq) )

    fund_ind = np.argmax( np.abs(drivefft[1:maxind]) ) + 1
    drive_freq = freqs[fund_ind]

    mindrive = np.min(drive)
    maxdrive = np.max(drive)

    meanresp = np.mean(resp)

    ### Build the notch filter
    drivefilt = np.zeros_like(drivefft) #+ np.random.randn(len(drivefft))*1.0e-3
    drivefilt[fund_ind] = 1.0 + 0.0j

    errfilt = np.zeros_like(drivefilt)
    noise_bins = (freqs > 10.0) * (freqs < 100.0)
    errfilt[noise_bins] = 1.0+0.0j
    errfilt[fund_ind] = 0.0+0.0j

    #plt.loglog(freqs, np.abs(respfft))
    #plt.loglog(freqs, np.abs(respfft)*errfilt)
    #plt.show()

    ### Error message triggered by verbose option
    if verbose:
        if ( (np.abs(drivefft[fund_ind-1]) > 0.03 * np.abs(drivefft[fund_ind])) or \
             (np.abs(drivefft[fund_ind+1]) > 0.03 * np.abs(drivefft[fund_ind])) ):
            print("More than 3% power in neighboring bins: spatial binning may be suboptimal")
            sys.stdout.flush()
            plt.loglog(freqs, np.abs(drivefft))
            plt.loglog(freqs[fund_ind], np.abs(drivefft[fund_ind]), '.', ms=20)
            plt.show()
    

    ### Expand the filter to more than a single bin. This can introduce artifacts
    ### that appear like lissajous figures in the resp vs. drive final result
    if width:
        lower_ind = np.argmin(np.abs(drive_freq - 0.5 * width - freqs))
        upper_ind = np.argmin(np.abs(drive_freq + 0.5 * width - freqs))
        drivefilt[lower_ind:upper_ind+1] = drivefilt[fund_ind]

    ### Generate an array of harmonics
    if not len(harms):
        harms = np.array([x+2 for x in range(nharmonics)])

    ### Loop over harmonics and add them to the filter
    for n in harms:
        harm_ind = np.argmin( np.abs(n * drive_freq - freqs) )
        drivefilt[harm_ind] = 1.0+0.0j
        errfilt[harm_ind] = 0.0+0.0j
        if width:
            h_lower_ind = harm_ind - (fund_ind - lower_ind)
            h_upper_ind = harm_ind + (upper_ind - fund_ind)
            drivefilt[h_lower_ind:h_upper_ind+1] = drivefilt[harm_ind]

    if correct_phase_shift:
        phase_shift = np.angle(respfft[fund_ind]) - np.angle(drivefft[fund_ind])
        drivefilt2 = drivefilt * np.exp(-1.0j * phase_shift)
    else:
        drivefilt2 = np.copy(drivefilt)

    if add_mean:
        drivefilt[0] = 1.0+0.0j
        drivefilt2[0] = 1.0+0.0j

    ### Apply the filter to both drive and response
    drivefft_filt = drivefilt * drivefft
    respfft_filt = drivefilt2 * respfft
    errfft_filt = errfilt * respfft

    ### Reconstruct the filtered data
    drive_r = np.fft.irfft(drivefft_filt) 
    resp_r = np.fft.irfft(respfft_filt)
    err_r = np.fft.irfft(errfft_filt)

    ### Sort reconstructed data, interpolate and resample
    mindrive = np.min(drive_r)
    maxdrive = np.max(drive_r)
    grad = np.gradient(drive_r)

    sortinds = drive_r.argsort()
    drive_r = drive_r[sortinds]
    resp_r = resp_r[sortinds]
    err_r = err_r[sortinds]

    if grad_sign < 0:
        ginds = grad[sortinds] < 0
    elif grad_sign > 0:
        ginds = grad[sortinds] > 0
    elif grad_sign == 0.0:
        ginds = np.ones(len(grad[sortinds]), dtype=np.bool)

    bin_spacing = (maxdrive - mindrive) * (1.0 / nbins)
    drivevec = np.linspace(mindrive+0.5*bin_spacing, \
                           maxdrive-0.5*bin_spacing, nbins)
    
    ### This part is slow, don't really know the best way to fix that....
    respvec = []
    errvec = []
    for bin_loc in drivevec:
        inds = (drive_r[ginds] >= bin_loc - 0.5*bin_spacing) * \
               (drive_r[ginds] < bin_loc + 0.5*bin_spacing)
        val = np.mean( resp_r[ginds][inds] )
        err_val = np.mean( err_r[ginds][inds] )
        respvec.append(val)
        errvec.append(err_val)

    respvec = np.array(respvec)
    errvec = np.array(errvec)

    #plt.plot(drive_r, resp_r)
    #plt.plot(drive_r[ginds], resp_r[ginds], linewidth=2)
    #plt.plot(drive_r[np.invert(ginds)], resp_r[np.invert(ginds)], linewidth=2)
    #plt.plot(drivevec, respvec, linewidth=5)
    #plt.show()

    if sg_filter:
        respvec = signal.savgol_filter(respvec, sg_params[0], sg_params[1])


    if plot:
        plt.figure()
        drive_asd = np.abs(drivefft)
        resp_asd = np.abs(respfft)
        plt.loglog(freqs, drive_asd / np.max(drive_asd), label='Drive')
        plt.loglog(freqs, resp_asd / np.max(resp_asd), label='Response')
        plt.loglog(freqs[np.abs(drivefilt)>0], \
                   resp_asd[np.abs(drivefilt)>0] / np.max(resp_asd), 
                   'X', label='Filter', ms=10)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('ASD [arb.]')
        plt.legend()

        plt.figure()
        plt.errorbar(drivevec, respvec,  yerr=errvec, ls='', marker='o', ms=6)
        plt.xlabel('Drive units')
        plt.ylabel('Response units')

        plt.show()


    return drivevec, respvec, errvec




def rebin(xvec, yvec, errs=[], nbin=500, plot=False, correlated_errs=False, \
          correlation_fac=1.0):
    '''Slow and derpy function to re-bin based on averaging. Works
       with any value of nbins, but can be slow since it's a for loop.'''

    # print(len(xvec), nbin)

    if len(errs):
        assert len(errs) == len(yvec), 'error vec is not the right length'

    if nbin > 0.25 * len(xvec):
        nbin = int(0.25 * len(xvec))

    lenx = np.max(xvec) - np.min(xvec)
    dx = lenx / nbin

    xvec_new = np.linspace(np.min(xvec)+0.5*dx, np.max(xvec)-0.5*dx, nbin)
    yvec_new = np.zeros_like(xvec_new)
    errs_new = np.zeros_like(xvec_new)
    errs_new_2 = np.zeros_like(xvec_new)

    for xind, x in enumerate(xvec_new):
        if x != xvec_new[-1]:
            inds = (xvec >= x - 0.5*dx) * (xvec < x + 0.5*dx)
        else:
            inds = (xvec >= x - 0.5*dx) * (xvec <= x + 0.5*dx)

        if len(errs):
            weights = 1.0 / errs[inds]**2
            errs_new[xind] = np.sqrt(1.0/np.sum(weights))
            yvec_new[xind] = np.average(yvec[inds], weights=weights)
        else:
            npts = np.sum(inds)
            yvec_new[xind] = np.mean(yvec[inds])
            errs_new[xind] = np.std(yvec[inds]) / np.sqrt(npts)
            if correlated_errs:
                # ecdf = ECDF(yvec[inds])
                # errs_new[xind] = np.sqrt(ecdf.get_variance(npts=npts))
                errs_new[xind] *= correlation_fac * np.sqrt(npts)

    if plot:
        fig, ax = plt.subplots(1,1)
        ax.scatter(xvec, yvec, color='C0', label='Original')
        ax.errorbar(xvec_new, yvec_new, yerr=errs_new, fmt='o', color='C1',\
                     label='Re-binned, with uncertainties')
        ax.set_title('Derpy Re-binning')

        fig.tight_layout()
        plt.show()
        input()

    return xvec_new, yvec_new, errs_new



def rebin_mean(a, *args):
    '''Uses a technique based on a scipy cookbook to do vectorized
       rebinning with the "evList" technique, which some consider
       'ugly' in implementation:
       https://scipy-cookbook.readthedocs.io.items/Rebinning.html

       An arbitrarily shaped array a can be rebinned into a shape
       given by *args. Output will have shape (args[0], args[1], ....),
       with the caveat the ratio (points in the oversampled array) / 
       (points in the rebinned array) has to be an integer, much like
       a downsampling factor. Elements of the rebinned array are the 
       mean of points within the appropriate window.

       Needs to be applied separately for xvec and yvec'''
    shape = a.shape
    lenShape = len(shape)
    factor = (np.asarray(shape)/np.asarray(args)).astype(int)

    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.mean(%d)'%(i+1) for i in range(lenShape)]
    #print ''.join(evList)

    return eval(''.join(evList))



def rebin_std(a, *args):
    '''Refer to rebin_mean() docstring. Not sure why, but this one
       seems to have trouble with more than 1D input arrays.'''
    shape = a.shape
    lenShape = len(shape)
    factor = (np.asarray(shape)/np.asarray(args)).astype(int)

    evList = ['a.reshape('] + \
              ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
              [')'] + ['.std(%d)/np.sqrt(factor[%d])'%(i+1,i) for i in range(lenShape)]

    return eval(''.join(evList))



def rebin_vectorized(xvec, yvec, nbin, model=None):
    '''Takes a vector (1D numpy array) a and rebins it to size nbin,
       with the caveats stated in rebin_mean() and rebin_std() docstrings.
       If the underlying data should follow a model, this first fits the data
       to said model and rebins the residuals to determine the appropriate
       rebinned error array.'''
    nbin_int = int(nbin)
    xvec_rb = rebin_mean(xvec, nbin_int)
    yvec_rb = rebin_mean(yvec, nbin_int)
    if model is not None:
        popt, pcov = opti.curve_fit(model, np.arange(nbin_int), yvec_rb)
        resid = yvec - model(np.linspace(0, nbin_int-1, len(yvec)), *popt)
        yvec_err_rb = rebin_std(resid, nbin_int)
    else:
        yvec_err_rb = rebin_std(yvec, nbin_int)
    return xvec_rb, yvec_rb, yvec_err_rb




def correlation(drive, response, fsamp, fdrive, filt = False, band_width = 1):
    '''Compute the full correlation between drive and response,
       correctly normalized for use in step-calibration.

       INPUTS:   drive, drive signal as a function of time
                 response, resposne signal as a function of time
                 fsamp, sampling frequency
                 fdrive, predetermined drive frequency
                 filt, boolean switch for bandpass filtering
                 band_width, bandwidth in [Hz] of filter

       OUTPUTS:  corr_full, full and correctly normalized correlation'''

    ### First subtract of mean of signals to avoid correlating dc
    drive = drive-np.mean(drive)
    response = response-np.mean(response)

    ### bandpass filter around drive frequency if desired.
    if filt:
        b, a = signal.butter(3, [2.*(fdrive-band_width/2.)/fsamp, \
                             2.*(fdrive+band_width/2.)/fsamp ], btype = 'bandpass')
        drive = signal.filtfilt(b, a, drive)
        response = signal.filtfilt(b, a, response)
    
    ### Compute the number of points and drive amplitude to normalize correlation
    lentrace = len(drive)
    drive_amp = np.sqrt(2)*np.std(drive)

    ### Define the correlation vector which will be populated later
    corr = np.zeros(int(fsamp/fdrive))

    ### Zero-pad the response
    response = np.append(response, np.zeros(int(fsamp / fdrive) - 1) )

    ### Build the correlation
    for i in range(len(corr)):
        ### Correct for loss of points at end
        correct_fac = 2.0*lentrace/(lentrace-i) ### x2 from empirical test
        corr[i] = np.sum(drive*response[i:i+lentrace])*correct_fac

    return corr * (1.0 / (lentrace * drive_amp))





def detrend_linalg(input_sig, xvec=None, input_unc=None, coeffs=False, \
                   plot=False):
    '''
    Function for rapid detrending using the MLE of a least-squared
    linear regression. Uses linear algebra to compute the MLE so as
    not to incur the computational cost of minimization.

    INPUTS:

        input_sig : array-like, signal to be detrended. If 'xvec' is not
            provided, this assumes the values of input_sig are equally 
            spaced along the axis of the independent variable

        xvec : array-like with same dimensions as 'input_sig'. Gvies the
            locations of teh values in input_sig, useful for uneven spacing

        input_unc : array-like uncertainties on the input_sig. Assumes 
            uniform uncertainties if this is left as None

        coeffs : boolean to just return the slope and offset instead of 
            the detrended array

        plot : boolean for plotting the result of the detrending

    OUTPUTS:

        output_sig : array-like with same size as 'input_sig'. Contains
            'input_sig' without the best fit linear component

        coeffs : optional output configuration. tuple with slope and offset

    '''

    if xvec is None:
        xvec = np.arange(len(input_sig))
    if input_unc is None:
        input_unc = np.ones_like(input_sig)

    xmean = np.mean(xvec)
    ymean = np.average(input_sig, weights=1.0/input_unc**2)

    ### Do linear regression and maximum likelihood estimation with 
    ### matrices since it's a trivial problem and we don't care about
    ### uncertainties in this particular operation
    slope = np.sum((xvec-xmean)*(input_sig-ymean)) / np.sum((xvec-xmean)**2)
    offset = ymean - slope * xmean

    if plot:
        fig, axarr = plt.subplots(2,1,figsize=(8,6),sharex=True,sharey=True)
        axarr[0].set_title('Signal detrending')
        axarr[0].plot(xvec, input_sig, color='k')
        axarr[0].plot(xvec, slope*xvec + offset, color='r', lw=2)
        axarr[1].plot(xvec, input_sig - (xvec * slope + offset), color='k')
        axarr[1].set_xlabel('Samples')
        fig.tight_layout()
        plt.show()

    if coeffs:
        return (slope, offset)
    else:
        return input_sig - (xvec * slope + offset)

        


def demod(input_sig, fsig, fsamp, harmind=1.0, filt=False, \
          bandwidth=1000.0, filt_band=[], pad=True, npad=1.0, \
          pad_mode='constant', optimize_frequency=False, \
          notch_freqs=[], notch_qs=[], \
          force_2pi_wrap=False, detrend=False, keep_mean=False, \
          tukey=False, tukey_alpha=1e-3, debug=False, plot=False):
    '''
    Sub-routine to perform a hilbert transformation on a given 
    signal, filtering it if requested and plotting throughout.
    Includes a tukey window to remove artifacts at the endpoint
    of the demodulated phase.

    Tukey windowing has been disabled by default as new padding
    routines significantly reduce edge effects.

    INPUTS:

        input_sig : 1D array-like, the signal to be demodulated

        fsig : float, demodulation frequency, fixed to this value
            by default (see option 'optimize_frequency')

        fsamp : float, signal sampling frequency

        harmind : int or float (but should be an integer unless
            you're wilding), default=1.0, harmonic of the carrier 
            to demodulate, useful when demodulating correlated
            signals from multiple measurements

        filt : boolean, default=False, flag to apply a 3rd order
            butterworth bandpass filter around the frequency to 
            be demodulated

        bandwidth : float, default=1000.0, bandwidth of the 
            butterworth filter optionally applied to the siganl

        filt_band : length 2 list of floats, default=[], explicit 
            upper and lower cutoffs of the optionally applied
            filter, overwrites the bandwidth option

        pad : boolean, default=True, whether to pad the signal

        npad : int or float, default=1.0, length of padding in units
            of the full signal length (i.e. default option triples
            the length of the signal)

        pad_mode : str, default='constant', padding method. See
            documentation of the numpy.pad() function.

        optimize_frequency : boolean, default=False, option to 
            adjust the given signal frequency in the event that it
            is not precisely known a priori. useful for incoherent 
            signals. implemented by taking the mean of the 
            derivative of the initially demodulationed phase of the 
            input signal in order to better compute the phase
            modulations of the carrier

        notch_freqs : list, default=[], list of frequencies to 
            notch prior to demodulation, useful when interfering
            signals are known. needs to be the same length as the
            'notch_qs' option

        notch_qs : list, default=[], list of quality factors (Q)
            of the notch filters applied. indexing and length 
            should match the 'notch_freqs' option

        force_2pi_wrap : boolean, default=False, option to enable
            a forced wrapping of the demodulated phase, implented
            with a naive modular operation (phase=phase%2pi)

        detrend : boolean, default=False, option to enable 
            detrending of the demodulated phase, again useful for
            incoherently sampled signals

        keep_mean : boolean, default=False, option to add the mean
            back after detrending, so only the linear component is
            removed

        tukey : boolean, default=False, option to apply a tukey 
            window to the demodulated amplitued and phase in 
            order to reduce edge effects (deprecated in favor
            of padding)

        tukey_alpha : float, default=1e-3, alpha value of the 
            half cosine tukey window, the length of the cosine
            tapering on the edges in units of the full signal
            length

        debug : boolean, default=False, if enabled, function 
            returns an additional dictionary with some numbers
            that may be useful for external operations

        plot : boolean, default=False, option to turn on some
            explanative plotting using matplotlib


    OUTPUTS:
        
        amp : 1D array-like of floats, demodulated amplitude 
            of the input signal

        phase_mod : 1D array-like of floats, demodulated phase
            of the input signal, with the carrier frequency
            component (i.e. phi_carrier = 2 pi fsig tvec) 
            removed and thus yielding the phase modulations
            of the carrier

        debug_dict : optional dictionary returned if the 
            'debug' option is True. read the action function
            to figure this one out, you may want to add things
            in your local version

    '''

    if debug:
        debug_dict = {}

    nsamp = len(input_sig)
    tvec = np.arange(nsamp) * (1.0 / fsamp)
    freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)

    fc = float(harmind) * fsig

    ### Hilbert transform doesn't work when there is an offset
    sig_mean = np.mean(input_sig)
    sig = input_sig - sig_mean

    ### Pad the waveform to be demodulated
    if pad:
        if not npad:
            npad_actual = int(nsamp)
        else:
            npad_actual = int(npad * nsamp)  ### Force cast, just in case

        sig = np.pad(sig, npad_actual, mode=pad_mode)

    if filt:
        if len(filt_band):
            lower, upper = filt_band
        else:
            lower = fc - 0.5 * bandwidth
            upper = fc + 0.5 * bandwidth

        sos = signal.butter(3, [lower, upper], btype='bandpass', fs=fsamp, output='sos')

        sig_filt = signal.sosfiltfilt(sos, sig, padlen=int(0.1*nsamp))
        if debug:
            debug_dict['sig_filt'] = sig_filt[npad_actual:-npad_actual]

        if len(notch_freqs):
            for i, notch_freq in enumerate(notch_freqs):
                notch_q = notch_qs[i]
                bn, an = signal.iirnotch(notch_freq, notch_q, fs=fsamp)
                sig_filt = signal.lfilter(bn, an, sig_filt)

        hilbert = signal.hilbert(sig_filt)

    else:
        hilbert = signal.hilbert(sig)

    if pad:
        sig = sig[npad_actual:-npad_actual]
        hilbert = hilbert[npad_actual:-npad_actual]
        if filt:
            sig_filt = sig_filt[npad_actual:-npad_actual]

    amp = np.abs(hilbert)
    phase = np.unwrap(np.angle(hilbert)) 

    if optimize_frequency:
        old_fc = fc
        wc, _ = detrend_linalg(phase, xvec=tvec, coeffs=True)
        fc = wc / (2.0*np.pi)
        if debug:
            debug_dict['optimized_frequency'] = fc

    phase_mod = phase - 2.0*np.pi*fc*tvec

    phase_mod = (phase_mod + np.pi) % (2.0*np.pi) - np.pi
    phase_mod_unwrap = np.unwrap(phase_mod)

    if detrend:
        good_inds = (phase_mod - phase_mod_unwrap) == 0
        inds = np.arange(len(phase_mod))
        slope, offset = \
            detrend_linalg(phase_mod[good_inds], xvec=inds[good_inds], coeffs=True)

        if debug:
            debug_dict['residual_freq'] = slope * fsamp

            print(fsig*float(harmind), fc, fc+debug_dict['residual_freq'])
            sys.stdout.flush()

        phase_mod_unwrap -= inds*slope + offset

        if plot:
            fig, axarr = plt.subplots(2,1,figsize=(8,6),sharex=True,sharey=True)
            axarr[0].set_title('Phase detrending')
            axarr[0].plot(inds, phase_mod, color='k')
            axarr[0].plot(inds, slope*inds + offset, color='r', lw=2)
            axarr[1].plot(inds, phase_mod_unwrap, color='k')
            axarr[1].set_xlabel('Samples')
            fig.tight_layout()

        if keep_mean:
            phase_mod_unwrap += offset + slope*np.mean(inds)

    if force_2pi_wrap:
        phase_mod = (phase_mod_unwrap + np.pi) % (2.0*np.pi) - np.pi
    else:
        phase_mod = np.copy(phase_mod_unwrap)

    if tukey:
        window = signal.tukey(len(phase), alpha=tukey_alpha)
        phase_mod *= window
        amp *= window

    phase_mod *= (1.0 / float(harmind))

    if plot:
        plt.figure()
        plt.title('Carrier signal')
        plt.plot(tvec, sig, label='signal')
        if filt:
            plt.plot(tvec, sig_filt, label='signal_filt')
        plt.plot(tvec, amp, label='amplitude')
        plt.xlabel('Time [s]')
        plt.ylabel('Signal [sig units]')
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.title('Carrier signal FFT')
        plt.loglog(freqs[1:], np.abs(np.fft.rfft(sig)[1:]), \
                   label='signal')
        if filt:
            plt.loglog(freqs[1:], np.abs(np.fft.rfft(sig_filt)[1:]), \
                       label='signal_filt')
        if optimize_frequency:
            plt.axvline(old_fc, ls='--', lw=3, color='r', alpha=0.5)
        plt.axvline(fc, ls='--', lw=3, color='r')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Signal ASD [Arb]')
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.title('Carrier phase modulation')
        plt.plot(tvec, phase_mod)
        plt.xlabel('Time [s]')
        plt.ylabel('Phase Modulation [rad]')
        plt.tight_layout()

        plt.figure()
        plt.title('Carrier phase modulation FFT')
        plt.loglog(freqs[1:], np.abs(np.fft.rfft(phase_mod)[1:]))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase ASD [Arb]')
        plt.tight_layout()

        plt.show()
        input()

    if debug:
        return amp, phase_mod, debug_dict
    else:
        return amp, phase_mod





def minimize_nll(nll_func, param_arr, confidence_level=0.9, plot=False):
    # 90% confidence level for 1sigma errors

    def parabola(x, a, b, c):
        # Just in case there is some globally defined 'parabola' hanging
        # around from other minimizations, this defines the standard 
        # form for the 1d minimization performed here
        return a * x**2 + b * x + c

    chi2dist = stats.chi2(1)
    # factor of 0.5 from Wilks's theorem: -2 log (Liklihood) ~ chi^2(1)
    con_val = 0.5 * chi2dist.ppf(confidence_level)

    nll_arr = []
    for param in param_arr:
        nll_arr.append(nll_func(param))
    nll_arr = np.array(nll_arr)

    popt_chi, pcov_chi = optimize.curve_fit(parabola, param_arr, nll_arr)

    minparam = - popt_chi[1] / (2. * popt_chi[0])
    minval = (4. * popt_chi[0] * popt_chi[2] - popt_chi[1]**2) / (4. * popt_chi[0])

    data_con_val = con_val - 1 + minval

    # Select the positive root for the non-diagonalized data
    soln1 = ( -1.0 * popt_chi[1] + np.sqrt( popt_chi[1]**2 - \
                    4 * popt_chi[0] * (popt_chi[2] - data_con_val)) ) / (2 * popt_chi[0])
    soln2 = ( -1.0 * popt_chi[1] - np.sqrt( popt_chi[1]**2 - \
                    4 * popt_chi[0] * (popt_chi[2] - data_con_val)) ) / (2 * popt_chi[0])

    err =  np.mean([np.abs(soln1 - minparam), np.abs(soln2 - minparam)])

    if plot:
        lab = ('{:0.2e}$\\pm${:0.2e}\n'.format(minparam, err)) + \
                'min$(\\chi^2/N_{\\mathrm{DOF}})=$' + '{:0.2f}'.format(minval)
        plt.plot(param_arr, nll_arr)
        plt.plot(param_arr, parabola(param_arr, *popt_chi), '--', lw=2, color='r', \
                    label=lab)
        plt.xlabel('Fit Parameter')
        plt.ylabel('$\\chi^2 / N_{\\mathrm{DOF}}$')
        plt.legend(fontsize=12, loc=0)
        plt.tight_layout()
        plt.show()


    return minparam, err, minval






def get_limit_from_general_profile(profx, profy, ss=False, confidence_level=0.95, \
                                   centered=False, no_discovery=False):

    chi2dist = stats.chi2(1)
    con_val = chi2dist.ppf(confidence_level)
    if not ss:
        con_val *= 0.5

    profx = np.array(profx)
    profy = np.array(profy)

    profy -= np.min(profy)

    if no_discovery:
        minval = 0.0
    else:
        minval = profx[np.argmin(profy)]

    profx_pos = profx[profx >= minval]
    profy_pos = profy[profx >= minval]

    profx_neg = profx[profx < minval]
    profy_neg = profy[profx < minval]

    try:
        ind_pos = np.argmin(np.abs(profy_pos - con_val))

        func_pos = interpolate.interp1d(profx_pos[ind_pos-5:ind_pos+5], \
                                        profy_pos[ind_pos-5:ind_pos+5], \
                                        fill_value='extrapolate')
        wrapper_pos = lambda x: (func_pos(x) - con_val)**2.0
        res_pos = optimize.minimize(wrapper_pos, x0=profx_pos[ind_pos]).x[0]
    except:
        res_pos = 0.0

    try:
        ind_neg = np.argmin(np.abs(profy_neg - con_val))
        func_neg = interpolate.interp1d(profx_neg[ind_neg-5:ind_neg+5], \
                                        profy_neg[ind_neg-5:ind_neg+5], \
                                        fill_value='extrapolate')
        wrapper_neg = lambda x: (func_neg(x) - con_val)**2.0
        res_neg = optimize.minimize(wrapper_neg, x0=profx_neg[ind_neg]).x[0]
    except:
        res_neg = 0.0

    if centered and not no_discovery:
        minval = np.mean([res_pos, res_neg])

    upper_unc = np.abs(res_pos - minval)
    lower_unc = -1.0 * np.abs(res_neg - minval)

    if no_discovery:
        return {'min': 0.0, 'upper_unc': res_pos, 'lower_unc': res_neg}

    else:
        return {'min': minval, 'upper_unc': upper_unc, 'lower_unc': lower_unc}







def trap_efield(voltages, nsamp=0, only_x=False, only_y=False, only_z=False, \
                n_elec=8, new_trap=False, plot_voltages=False):
    '''Using output of 4/2/19 COMSOL simulation, return
       the value of the electric field at the trap based
       on the applied voltages on each electrode and the
       principle of superposition.'''
    if nsamp == 0:
        nsamp = len(voltages[0])
    if len(voltages) != n_elec:
        print("There are ({:d}) electrodes.".format(n_elec))
        print("   len(volt arr. passed to 'trap_efield') != {:d}".format(n_elec))
    else:
        if plot_voltages:
            for i in range(n_elec):
                plt.plot(voltages[i], label='{:d}'.format(i))
            plt.xlabel('Sample index')
            plt.ylabel('Voltage [V]')
            plt.legend(fontsize=10, loc=0, ncol=2)
            plt.tight_layout()
            plt.show()

        if only_y or only_z:
            Ex = np.zeros(nsamp)
        else:
            if new_trap:
                Ex = voltages[3] * E_xp(0.0) + voltages[4] * E_xn(0.0)
            else:   
                Ex = voltages[3] * E_front(0.0) + voltages[4] * E_back(0.0)

        if only_x or only_z:
            Ey = np.zeros(nsamp)
        else:
            if new_trap:
                Ey = voltages[5] * E_yp(0.0) + voltages[6] * E_yn(0.0)
            else:
                Ey = voltages[5] * E_right(0.0) + voltages[6] * E_left(0.0)

        if only_y or only_z:
            Ez = np.zeros(nsamp)
        else:
            if new_trap:
                Ez = voltages[1] * E_zp(0.0)   + voltages[2] * E_zn(0.0)
            else:
                Ez = voltages[1] * E_top(0.0)   + voltages[2] * E_bot(0.0)

        return np.array([Ex, Ey, Ez])   








def thermal_psd_spec(f, A, f0, g):
    #The position power spectrum of a microsphere normalized so that A = (volts/meter)^2*2kb*t/M
    w = 2.*np.pi*f #Convert to angular frequency.
    w0 = 2.*np.pi*f0
    num = g * w0**2
    denom = ((w0**2 - w**2)**2 + w**2*g**2)
    return 2 * (A * num / denom) # Extra factor of 2 from single-sided PSD




def damped_osc_amp(f, A, f0, g):
    '''Fitting function for AMPLITUDE of a damped harmonic oscillator
           INPUTS: f [Hz], frequency 
                   A, amplitude
                   f0 [Hz], resonant frequency
                   g [Hz], damping factor
                   c, constant offset, default 0

           OUTPUTS: Lorentzian amplitude'''
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    gamma = 2. * np.pi * g
    denom = np.sqrt( (w0**2 - w**2)**2 + w**2 * gamma**2 )
    return A / denom




def damped_osc_phase(f, A, f0, g, phase0 = 0.):
    '''Fitting function for PHASE of a damped harmonic oscillator.
       Includes an arbitrary DC phase to fit over out of phase responses
           INPUTS: f [Hz], frequency 
                   A, amplitude
                   f0 [Hz], resonant frequency
                   g [Hz], damping factor

           OUTPUTS: Lorentzian amplitude'''
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    gamma = 2. * np.pi * g
    # return A * np.arctan2(-w * g, w0**2 - w**2) + phase0
    return 1.0 * np.arctan2(-w * gamma, w0**2 - w**2) + phase0



def damped_osc_amp_squash(f, A, f0, g0, noise, deriv_gain, tfb):
    '''Fitting function for AMPLITUDE of a damped harmonic oscillator
           INPUTS: f [Hz], frequency 
                   A, amplitude
                   f0 [Hz], resonant frequency
                   g [Hz], damping factor
                   c, constant offset, default 0

           OUTPUTS: Lorentzian amplitude'''

    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    gamma0 = 2. * np.pi * g0

    ### Libration paper referee correctly caught this error where I used
    ### the resonant frequency, not the frequency argument, in the second
    ### term, which was derived from a mistake in handling the equations
    # num = np.sqrt( A**2 + noise*((w0**2 - w**2)**2 + w0**2 * gamma0**2) )
    num = np.sqrt( A**2 + noise*((w0**2 - w**2)**2 + w**2 * gamma0**2) )
    
    denom = np.sqrt( (w0**2 - w**2 + w * deriv_gain * np.sin(w * tfb))**2 \
                        + w**2 * (gamma0 + deriv_gain * np.cos(w * tfb))**2 )

    return num / denom



def fit_damped_osc_amp(sig, fsamp, fit_band=[], optimize_fit_band=False, \
                       ngamma=5.0, plot=False, sig_asd=False, linearize=False, \
                       asd_errs=[], gamma_guess=0, freq_guess=0, \
                       constant_guess = 0, \
                       weight_lowf=False, weight_lowf_val=1.0, \
                       weight_lowf_thresh=100.0, maxfev=100000, \
                       return_func=False, verbose=False):
    '''
    Routine to fit the above defined damped HO amplitude spectrum
    to the ASD of some input signal, assumed to have a single
    resonance etc etc

    INPUTS: 

        sig - signal to analyze 
                   
        fsamp [Hz] - sampling frequency of input signal
                   
        fit_band [Hz] - array-like with upper/lower limits
                   
        plot - boolean flag for plotting of fit results
                   
        sig_asd - boolean indicating if 'sig' is already an
            an amplitude spectral density
                   
        linearize, try linearizing the fit
                   
        asd_errs, uncertainties to use in fitting

    OUTPUTS: 

        popt, optimal parameters from curve_fit
                    
        pcov, covariance matrix from curve_fit'''

    ### Generate the frequencies and ASD values from the given time
    ### domain signal, or given ASD
    if not sig_asd:
        nsamp = len(sig)
        freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)
        asd = np.abs( np.fft.rfft(sig) ) * fft_norm(nsamp, fsamp)
    else:
        asd = np.abs(sig)
        freqs = np.linspace(0, 0.5*fsamp, len(asd))

    df = freqs[1] - freqs[0]
    derp_inds = np.arange(len(freqs))

    ### Define some indicies for fitting and plotting
    if len(fit_band):
        inds = (freqs > fit_band[0]) * (freqs < fit_band[1])
    else:
        inds = (freqs > 0.1 * freq_guess) * (freqs < 10.0 * freq_guess)

    first_ind = np.min(derp_inds[inds])
    last_ind = np.max(derp_inds[inds])

    ### Generate some initial guesses
    maxind = np.argmax(asd[inds])   # Look for max within fit_band
    if not freq_guess:
        freq_guess = (freqs[inds])[maxind]
    if not gamma_guess:
        for i in range(int(0.5*np.sum(inds))):
            if i == maxind:
                continue
            if asd[inds][i] >= 0.5*asd[inds][maxind]:
                gamma_guess = freqs[inds][maxind] - freqs[inds][i]
                break
        if not gamma_guess:
            gamma_guess = df / np.log10(asd[inds][maxind]**2/asd[inds][maxind-1]**2)

    ### Adjust the fit band for a certain number of gamma factors
    if optimize_fit_band:
        delta_band = np.max([0.5*ngamma*gamma_guess, 21*df])
        fit_band = [freq_guess - delta_band, \
                    freq_guess + delta_band]
        inds = (freqs > fit_band[0]) * (freqs < fit_band[1])
        maxind = np.argmax(asd[inds])

    amp_guess = (asd[inds])[maxind] * gamma_guess * freq_guess * (2.0 * np.pi)**2

    ### Define the fitting function to use. Keeping this modular in case
    ### we ever want to make more changes/linearization attempts
    fit_x = freqs[inds]
    if linearize:
        fit_func = lambda f,A,f0,g,c: np.log(np.sqrt(damped_osc_amp(f,A,f0,g)**2 + c**2))
        fit_y = np.log(asd[inds])
        if not len(asd_errs):
            errs = np.ones(len(fit_x))
            absolute_sigma = False
        else:
            errs = asd_errs[inds] / asd[inds]
            absolute_sigma = True
    else:
        fit_func = lambda f,A,f0,g,c: np.sqrt(damped_osc_amp(f,A,f0,g)**2 + c**2)
        fit_y = asd[inds]
        if not len(asd_errs):
            errs = np.ones(len(fit_x)) * np.mean(fit_y)
            absolute_sigma = False
        else:
            errs = asd_errs[inds]
            absolute_sigma = True

    if not constant_guess:
        constant_guess = \
            np.sqrt(np.mean(np.concatenate( (asd[first_ind-20:first_ind], \
                                             asd[last_ind:last_ind+20]) )**2))

    if weight_lowf:
        weight_inds = fit_x < weight_lowf_thresh
        errs[weight_inds] *= weight_lowf_val

    ### Fit the data
    p0 = [amp_guess, freq_guess, gamma_guess, constant_guess]

    ndof = len(fit_x) - 4
    def cost(A, f, g, c):
        resid = (fit_y - fit_func(fit_x, A, f, g, c))**2
        variance = errs**2
        return (1.0 / ndof) * np.sum(resid / variance)

    m = Minuit(cost, \
               A = p0[0], \
               f = p0[1], \
               g = p0[2],
               c = p0[3])

    ### Apply some limits to help keep the fitting well behaved
    m.limits['A'] = (0, np.inf)
    m.limits['f'] = (0, np.inf)
    m.limits['g'] = (1e-6, np.inf)
    m.limits['c'] = (np.min(fit_y), np.inf)

    m.errordef = 1
    m.print_level = 0

    ### Do the actual minimization
    m.migrad(ncall=500000)
    popt = np.array(m.values)
    pcov = np.array(m.covariance)

    ### Plot!
    if plot:
        fit_mult_span = fit_band[1] / fit_band[0]
        lower_plot_freq = fit_band[0] / (0.2 * (fit_mult_span - 1) + 1)
        upper_plot_freq = fit_band[1] * (0.2 * (fit_mult_span - 1) + 1)
        plot_inds = (freqs > lower_plot_freq) * (freqs < upper_plot_freq)

        # fit_freqs = np.linspace((freqs[inds])[0], (freqs[inds])[-1], \
        #                         10*len(freqs[inds])) 
        fit_freqs = np.linspace(fit_band[0], fit_band[1], 10*len(freqs[inds]))
        init_mag = fit_func(fit_freqs, *p0)
        fit_mag = fit_func(fit_freqs, *popt)

        if verbose:
            print()
            print('Initial parameter guess:')
            print(f'    {p0[0]:0.3g}, {p0[1]:0.3g}, {p0[2]:0.3g}, {p0[3]:0.3g}')
            print()
            print('Fit result:')
            print(f'    {popt[0]:0.3g}, {popt[1]:0.3g}, {popt[2]:0.3g}, {popt[3]:0.3g}')
            print()

        fig, ax = plt.subplots(1,1)
        ax.loglog(freqs[plot_inds], asd[plot_inds])
        ax.loglog(fit_freqs, init_mag, ls='--', color='k', label='init')
        ax.loglog(fit_freqs, fit_mag, ls='--', color='r', label='fit')
        ax.set_xlim((freqs[plot_inds])[0], (freqs[plot_inds])[-1])
        ax.set_ylim(0.5 * np.min(asd[plot_inds]), 2.0 * np.max(asd[plot_inds]))
        ax.legend()
        ax.set_title('Damped HO Fitting')
        ax.xaxis.set_major_formatter(lambda x, pos: f'{x:0.1f}')
        ax.xaxis.set_minor_formatter(lambda x, pos: f'{x:0.1f}')
        fig.tight_layout()
        plt.show()

        # input()

    if return_func:
        return popt, pcov, fit_func
    else:
        return popt, pcov




def fit_damped_osc_amp_squash(\
        sig, fsamp, fit_band=[], optimize_fit_band=False, \
        ngamma=5.0, plot=False, sig_asd=False, linearize=False, \
        asd_errs=[], amp_guess=0, gamma_guess=0, freq_guess=0, \
        constant_guess=0, noise_guess=0, \
        deriv_gain_guess=0, deriv_delay_guess=1.0/50000.0, \
        fix_delay=False, \
        weight_lowf=False, weight_lowf_val=1.0, \
        weight_lowf_thresh=100.0, maxfev=100000, \
        return_func=False, verbose=False):
    '''
    Routine to fit the above defined damped HO amplitude spectrum
    to the ASD of some input signal, assumed to have a single
    resonance etc etc

    INPUTS: 

        sig - signal to analyze 
                   
        fsamp [Hz] - sampling frequency of input signal
                   
        fit_band [Hz] - array-like with upper/lower limits
                   
        plot - boolean flag for plotting of fit results
                   
        sig_asd - boolean indicating if 'sig' is already an
            an amplitude spectral density
                   
        linearize, try linearizing the fit
                   
        asd_errs, uncertainties to use in fitting

    OUTPUTS: 

        popt, optimal parameters from curve_fit
                    
        pcov, covariance matrix from curve_fit'''

    ### Generate the frequencies and ASD values from the given time
    ### domain signal, or given ASD
    if not sig_asd:
        nsamp = len(sig)
        freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)
        asd = np.abs( np.fft.rfft(sig) ) * fft_norm(nsamp, fsamp)
    else:
        asd = np.abs(sig)
        freqs = np.linspace(0, 0.5*fsamp, len(asd))

    df = freqs[1] - freqs[0]
    derp_inds = np.arange(len(freqs))

    ### Define some indicies for fitting and plotting
    if len(fit_band):
        inds = (freqs > fit_band[0]) * (freqs < fit_band[1])
    else:
        inds = (freqs > 0.1 * freq_guess) * (freqs < 10.0 * freq_guess)

    first_ind = np.min(derp_inds[inds])
    last_ind = np.max(derp_inds[inds])
    gap = int(last_ind-first_ind)

    ### Generate some initial guesses
    maxind = np.argmax(asd[inds])   # Look for max within fit_band
    if not freq_guess:
        freq_guess = (freqs[inds])[maxind]
    if not gamma_guess:
        for i in range(int(0.5*np.sum(inds))):
            if i == maxind:
                continue
            if asd[inds][i] >= 0.5*asd[inds][maxind]:
                gamma_guess = freqs[inds][maxind] - freqs[inds][i]
                break
        if not gamma_guess:
            gamma_guess = df / np.log10(asd[inds][maxind]**2/asd[inds][maxind-1]**2)

    ### Adjust the fit band for a certain number of gamma factors
    if optimize_fit_band:
        delta_band = np.max([0.5*ngamma*gamma_guess, 21*df])
        fit_band = [freq_guess - delta_band, \
                    freq_guess + delta_band]
        inds = (freqs > fit_band[0]) * (freqs < fit_band[1])
        maxind = np.argmax(asd[inds])

    if not amp_guess:
        amp_guess = (asd[inds])[maxind]*gamma_guess*freq_guess * (2.0*np.pi)**2

    fit_x = freqs[inds]
    fit_y = asd[inds]
    if not len(asd_errs):
        errs = np.ones(len(fit_x)) * np.mean(fit_y)
        absolute_sigma = False
    else:
        errs = asd_errs[inds]
        absolute_sigma = True

    if not constant_guess:
        # constant_guess = np.mean( \
        #         np.concatenate( (asd[maxind-gap-20:maxind-gap], \
        #                          asd[maxind+gap:maxind+gap+20]) ) )
        # constant_guess = constant_guess**2
        # constant_guess *= 0.1
        constant_guess = np.min(fit_y)**2

    if not noise_guess:
        noise_guess = constant_guess / 10.0

    if not deriv_gain_guess:
        deriv_gain_guess = 1.0

    # print(deriv_gain_guess, fit_x[0], fit_x[-1])

    if weight_lowf:
        weight_inds = fit_x < weight_lowf_thresh
        errs[weight_inds] *= weight_lowf_val

    # ### Fit the data
    # p0 = [np.log10(amp_guess), freq_guess, gamma_guess, np.log10(noise_guess), \
    #         deriv_gain_guess, deriv_delay_guess, np.log10(constant_guess)]

    # ### Define the fitting function to use. Keeping this modular in case
    # ### we ever want to make more changes/linearization attempts
    # fit_func = lambda f,A,f0,g,noise,kd,tfb,c: \
    #                 np.sqrt( damped_osc_amp_squash(f,10**A,f0,g,\
    #                                                10**noise,kd,tfb)**2 \
    #                             + 10**c )

    ### Fit the data
    p0 = [amp_guess, freq_guess, gamma_guess, noise_guess, \
            deriv_gain_guess, deriv_delay_guess, constant_guess]

    ### Define the fitting function to use. Keeping this modular in case
    ### we ever want to make more changes/linearization attempts
    fit_func = lambda f,A,f0,g,noise,kd,tfb,c: \
                    np.sqrt( damped_osc_amp_squash(f,A,f0,g,\
                                                   noise,kd,tfb)**2 + c )

    ndof = len(fit_x) - 7
    def cost(A, f, g, noise, kd, tfb, c):
        resid = (fit_y - fit_func(fit_x, A, f, g, noise, kd, tfb, c))**2
        variance = errs**2
        return (1.0 / ndof) * np.sum(resid / variance)

    m = Minuit(cost, \
               A = p0[0], f = p0[1], g = p0[2], noise = p0[3], \
               kd = p0[4], tfb = p0[5], c = p0[6])

    ### Apply some limits to help keep the fitting well behaved
    m.limits['A'] = (0.1*p0[0], 10.0*p0[0])
    m.limits['f'] = (p0[1]-p0[4]/10.0, p0[1]+p0[4]/10.0)
    m.limits['g'] = (1e-3, 0.1)
    m.limits['noise'] = (p0[3]*0.1, p0[3]*10.0)
    m.limits['kd'] = (0.5*p0[4], 2.0*p0[4])
    m.limits['tfb'] = (1e-6, 1e-3)
    m.limits['c'] = (p0[6]*0.01, p0[6]*100.0)

    if fix_delay:
        m.fixed['tfb'] = True

    # print(m.values['kd'])

    m.errordef = 1
    m.print_level = 0

    ### Do the actual minimization
    m.migrad(ncall=5000000)
    # print(m.values['kd'])

    m.minos()

    popt = np.array(m.values)

    popt_unc = np.zeros_like(popt)
    i = 0
    for param in m.parameters:
        unc = np.mean( np.abs([m.merrors[param].lower, m.merrors[param].upper]) )
        popt_unc[i] = unc
        i += 1

    pcov = np.array(m.covariance)

    # m.draw_profile('g')
    # plt.tight_layout()
    # plt.show()

    # for param in m.parameters:
    #     plt.figure()
    #     m.draw_profile(param)
    #     plt.axvline(m.limits[param][0])
    #     plt.axvline(m.limits[param][1])
    #     plt.tight_layout()
    # plt.show()


    ### Plot!
    if plot:
        fit_mult_span = fit_band[1] / fit_band[0]
        lower_plot_freq = fit_band[0] / (0.2 * (fit_mult_span - 1) + 1)
        upper_plot_freq = fit_band[1] * (0.2 * (fit_mult_span - 1) + 1)
        plot_inds = (freqs > lower_plot_freq) * (freqs < upper_plot_freq)

        # fit_freqs = np.linspace((freqs[inds])[0], (freqs[inds])[-1], \
        #                         10*len(freqs[inds])) 
        fit_freqs = np.linspace(fit_band[0], fit_band[1], 10*len(freqs[inds]))
        init_mag = fit_func(fit_freqs, *p0)
        fit_mag = fit_func(fit_freqs, *popt)

        if verbose:
            print()
            print('Initial parameter guess:')
            my_str = '    '
            for i in range(7):
                my_str += f'{p0[i]:0.4g}, '
            print(my_str)
            print()
            print('Fit result:')
            my_str = '    '
            for i in range(7):
                my_str += f'{popt[i]:0.4g}, '
            print(my_str)
            print()

            # print(m.params)
            # print()

            print(f'VALID? {str(m.valid)}')
            print()

            # print('Gamma: ')
            # print('    ', p0[2], ' : ', \
            #       f'{popt[2]:0.3g} +- {np.sqrt(pcov[2,2]):0.3g}')

            # print('kd: ')
            # print('    ', p0[4], ' : ', \
            #       f'{popt[4]:0.3g} +- {np.sqrt(pcov[4,4]):0.3g}')

        # p0 = [np.log10(amp_guess), freq_guess, gamma_guess, np.log10(noise_guess), \
        #         np.log10(deriv_gain_guess), deriv_phase_guess, np.log10(constant_guess)]

        fig, ax = plt.subplots(1,1)
        ax.loglog(freqs[plot_inds], asd[plot_inds])
        # ax.axhline(10**p0[-1], color='magenta', lw=3)
        ax.loglog(fit_freqs, init_mag, ls=':', color='k', label='init')
        # facs = np.array([1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0, 20.0, 100.0, 1000.0])
        # facs = np.array([1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9])
        # colors = get_colormap(len(facs), cmap='plasma')
        # for fac_ind, fac in enumerate(facs):
        #     new_p0 = np.copy(p0)
        #     # new_p0[3] *= fac
        #     new_p0[0] += np.log10(fac)
        #     new_p0[3] += np.log10(fac)
        #     ax.loglog(fit_freqs, fit_func(fit_freqs, *new_p0), color=colors[fac_ind], \
        #               label=f'{new_p0[0]:0.3g}')

        ax.loglog(fit_freqs, fit_mag, ls='--', color='r', label='fit')
        ax.axhline(np.sqrt(constant_guess))
        ax.set_xlim((freqs[plot_inds])[0], (freqs[plot_inds])[-1])
        # ax.set_ylim(0.5 * np.min(asd[plot_inds]), 2.0 * np.max(asd[plot_inds]))
        ax.legend()
        ax.set_title('Damped HO Fitting')
        ax.xaxis.set_major_formatter(lambda x, pos: f'{x:0.1f}')
        ax.xaxis.set_minor_formatter(lambda x, pos: f'{x:0.1f}')
        fig.tight_layout()
        plt.show()

        # input()

    if return_func:
        return popt, popt_unc, fit_func
    else:
        return popt, popt_unc









def fit_voigt_profile(sig, fsamp, fit_band=[], optimize_fit_band=False, \
                      ngamma=5.0, plot=False, sig_asd=False, linearize=False, \
                      asd_errs=[], gamma_guess=0, freq_guess=0, sigma_guess=0, \
                      weight_lowf=False, weight_lowf_val=1.0, \
                      weight_lowf_thresh=100.0, maxfev=100000, \
                      return_func=False, verbose=False):
    '''
    Routine to fit the above defined damped HO amplitude spectrum
    to the ASD of some input signal, assumed to have a single
    resonance etc etc

    INPUTS: 

        sig - signal to analyze 
                   
        fsamp [Hz] - sampling frequency of input signal
                   
        fit_band [Hz] - array-like with upper/lower limits
                   
        plot - boolean flag for plotting of fit results
                   
        sig_asd - boolean indicating if 'sig' is already an
            an amplitude spectral density
                   
        linearize, try linearizing the fit
                   
        asd_errs, uncertainties to use in fitting

    OUTPUTS: 

        popt, optimal parameters from curve_fit
                    
        pcov, covariance matrix from curve_fit'''

    ### Generate the frequencies and ASD values from the given time
    ### domain signal, or given ASD
    if not sig_asd:
        nsamp = len(sig)
        freqs = np.fft.rfftfreq(nsamp, d=1.0/fsamp)
        asd = np.abs( np.fft.rfft(sig) ) * fft_norm(nsamp, fsamp)
    else:
        asd = np.abs(sig)
        freqs = np.linspace(0, 0.5*fsamp, len(asd))

    df = freqs[1] - freqs[0]
    derp_inds = np.arange(len(freqs))

    ### Define some indicies for fitting and plotting
    if len(fit_band):
        inds = (freqs > fit_band[0]) * (freqs < fit_band[1])
    else:
        inds = (freqs > 0.1 * freq_guess) * (freqs < 10.0 * freq_guess)

    first_ind = np.min(derp_inds[inds])

    ### Generate some initial guesses
    maxind = np.argmax(asd[inds])   # Look for max within fit_band
    if not freq_guess:
        freq_guess = (freqs[inds])[maxind]
    if not gamma_guess:
        for i in range(int(0.5*np.sum(inds))):
            if asd[inds][i] >= 0.5*asd[inds][maxind]:
                gamma_guess = freqs[inds][maxind] - freqs[inds][i]
                break
        if gamma_guess == 0.0:
            gamma_guess = df / np.log10(asd[inds][maxind]**2/asd[inds][maxind-1]**2)

    ### Adjust the fit band for a certain number of gamma factors
    if optimize_fit_band:
        delta_band = np.max([0.5*ngamma*gamma_guess, 21*df])
        fit_band = [freq_guess - delta_band, \
                    freq_guess + delta_band]
        inds = (freqs > fit_band[0]) * (freqs < fit_band[1])
        maxind = np.argmax(asd[inds])

    amp_guess = 1e-4*(asd[inds])[maxind] * gamma_guess * freq_guess * (2.0 * np.pi)**2

    ### Define the fitting function to use. Keeping this modular in case
    ### we ever want to make more changes/linearization attempts
    fit_x = freqs[inds]

    fit_func = lambda f,A,f0,s,g,c: \
        np.sqrt( (A*special.voigt_profile((f-f0),s,g))**2 + c**2 )
    fit_y = asd[inds]
    if not len(asd_errs):
        errs = np.ones(len(fit_x)) * np.mean(fit_y)
        absolute_sigma = False
    else:
        errs = asd_errs[inds]
        absolute_sigma = True

    if weight_lowf:
        weight_inds = fit_x < weight_lowf_thresh
        errs[weight_inds] *= weight_lowf_val

    ### Fit the data
    p0 = [amp_guess, freq_guess, 0.1*gamma_guess, gamma_guess, np.min(fit_y)]

    ndof = len(fit_x) - 4
    def cost(A, f, s, g, c):
        resid = (fit_y - fit_func(fit_x, A, f, s, g, c))**2
        variance = errs**2
        return (1.0 / ndof) * np.sum(resid / variance)

    m = Minuit(cost, \
               A = p0[0], \
               f = p0[1], \
               s = p0[2], \
               g = p0[3], \
               c = p0[4])

    ### Apply some limits to help keep the fitting well behaved
    m.limits['A'] = (0, np.inf)
    m.limits['f'] = (0, np.inf)
    m.limits['s'] = (1e-6, np.inf)
    m.limits['g'] = (1e-6, np.inf)
    m.limits['c'] = (np.min(fit_y), np.inf)

    m.errordef = 1
    m.print_level = 0

    ### Do the actual minimization
    m.migrad(ncall=500000)
    popt = np.array(m.values)
    pcov = np.array(m.covariance)

    ### Plot!
    if plot:
        fit_mult_span = fit_band[1] / fit_band[0]
        lower_plot_freq = fit_band[0] / (0.2 * (fit_mult_span - 1) + 1)
        upper_plot_freq = fit_band[1] * (0.2 * (fit_mult_span - 1) + 1)
        plot_inds = (freqs > lower_plot_freq) * (freqs < upper_plot_freq)

        # fit_freqs = np.linspace((freqs[inds])[0], (freqs[inds])[-1], \
        #                         10*len(freqs[inds])) 
        fit_freqs = np.linspace(fit_band[0], fit_band[1], 10*len(freqs[inds]))
        init_mag = fit_func(fit_freqs, *p0)
        fit_mag = fit_func(fit_freqs, *popt)

        if verbose:
            print()
            print('Initial Voigt parameter guess [A, f0, sigma, gamma, c]:')
            print(f'    {p0[0]:0.3g}, {p0[1]:0.3g}, {p0[2]:0.3g}, '
                    + f'{p0[3]:0.3g}, {p0[4]:0.3g}')
            print()
            print('Fit result:')
            print(f'    {popt[0]:0.3g}, {popt[1]:0.3g}, {popt[2]:0.3g}, '
                    + f'{popt[3]:0.3g}, {popt[4]:0.3g}')
            print()

        fig, ax = plt.subplots(1,1)
        ax.loglog(freqs[plot_inds], asd[plot_inds])
        ax.loglog(fit_freqs, init_mag, ls='--', color='k', label='init')
        ax.loglog(fit_freqs, fit_mag, ls='--', color='r', label='fit')
        ax.set_xlim((freqs[plot_inds])[0], (freqs[plot_inds])[-1])
        ax.set_ylim(0.5 * np.min(asd[plot_inds]), 2.0 * np.max(asd[plot_inds]))
        ax.legend()
        ax.set_title('Voigt Profile Fitting')
        ax.xaxis.set_major_formatter(lambda x, pos: f'{x:0.1f}')
        ax.xaxis.set_minor_formatter(lambda x, pos: f'{x:0.1f}')
        fig.tight_layout()
        plt.show()

        # input()

    if return_func:
        return popt, pcov, fit_func
    else:
        return popt, pcov






def find_fft_peaks(freqs, fft, window=100, delta_fac=5.0, \
                   lower_delta_fac=0.0, exclude_df=10, band=[], \
                   plot=False):
    '''Function to scan the ASD associated to an input FFT, looking for 
       values above the baseline of the ASD, and then attempting to fit
       a gaussian to the region around the above-baseline feature. It 
       should avoid fitting the same feature multiple times.

            freqs : the array of frequencies associated to the input
                FFT. Assumed to be in Hz

            fft : the FFT in which we are trying to find peaks

            window : the width (in frequency bins) of the region where
                the fit is performed

            delta_fac : the factor above the baseline required to 
                trigger a peak fitting

            lower_delta_fac : a secondary factor to help avoid fitting
                spuriously large frequency components that aren't peaks.
                If set to the same value as delta_fac, it basically 
                does nothing

            exclude_df : the number of frequency bins to exclude 
                redundant peak finding around a feature that was already
                found and fit with a gaussian

            band : the frequenncy band (if any) in which to limit the search
       '''

    if not lower_delta_fac:
        lower_delta_fac = delta_fac

    ### Determin the frequency spacing
    df = freqs[1] - freqs[0]

    ### Compute the ASD
    asd = np.abs(fft)

    ### Define some arrays for later use
    baseline = []
    all_inds = np.arange(len(freqs))

    ### Define the fitting function, which in this case is a gaussian
    ### without a constant
    fit_fun = lambda x,a,b,c: gauss(x, a, b, c, 0)

    ### Loop over all (frequency, ASD) pairs, looking for peaks
    peaks = []
    for ind, (freq, val) in enumerate(zip(freqs, asd)):

        ### See if the current value is above the baseline
        if ind != 0:
            cond1 = val > lower_delta_fac * np.mean(baseline)
            cond2 = val > delta_fac * np.mean(baseline)
        else:
            baseline.append(val)
            continue

        ### Update the baseline, appending a value if it hasn't been
        ### fully built yet, skipping spuriously large values (based on
        ### the lower_delta_fac threshold), or updating an old value of
        ### baseline to properly reflect the local baseline
        if ind < window:
            baseline.append(val)
        elif cond1 and not cond2:
            continue
        else:
            baseline[int(ind%window)] = val

        ### If the current frequency is too close to the most recently found
        ### peak, then skip it and move on
        if len(peaks):
            if np.abs(freq - peaks[-1][0]) < exclude_df*df:
                continue

        ### If the current ASD value is sufficiently above the baseline,
        ### fit the region around the current value to a gaussian
        if cond2:
            ### Define a mask for local fitting
            mask = (all_inds > ind - window/2) * (all_inds < ind + window/2)

            ### Try the fit
            p0 = [val, freq, df]
            try:
                popt, pcov = optimize.curve_fit(fit_fun, freqs[mask], \
                                                asd[mask], p0=p0, maxfev=5000)
            except:
                popt = p0

            ### Append either the fit result, or our best guess if the fit failed
            peaks.append( [popt[1], popt[0]] )

    if plot:
        plot_pdet([peaks, []], freqs, asd, loglog=True)

    return np.array(peaks)






def select_largest_n_fft_peaks(freqs, fft, npeak=1, exclude_df=10, \
                               band=[], adjust_baseline=False, plot=False):
    '''Function to scan the ASD associated to an input FFT, looking for 
       values above the baseline of the ASD, and then attempting to fit
       a gaussian to the region around the above-baseline feature. It 
       should avoid fitting the same feature multiple times.

            freqs : the array of frequencies associated to the input
                FFT. Assumed to be in Hz

            fft : the FFT in which we are trying to find peaks

            npeak : the number of peaks to search for

            exclude_df : the number of frequency bins to exclude 
                redundant peak finding around a feature that was already
                found and fit with a gaussian

            band : the frequenncy band (if any) in which to limit the search
       '''

    ### Make sure the provided frequency band is right
    band = list(band)
    band.sort()

    ### Determine the frequency spacing
    df = freqs[1] - freqs[0]
    nfreq = len(freqs)

    ### Compute the ASD
    asd = np.abs(fft)

    start = np.mean(asd[:20])
    end = np.mean(asd[-20:])
    if np.abs(start - end) / np.mean([start, end]) < 10.0:
        adjust_baseline = False

    if adjust_baseline:
        power_law = lambda f,a,n,c: a * f**n + c
        p0 = [0.0, 0.1, np.mean(asd)]
        popt, pcov = optimize.curve_fit(power_law, freqs, asd, p0=p0, maxfev=10000, \
                                        bounds=([0, -3, 0], [np.inf, 3, np.inf]))
        asd /= popt[0] * freqs**popt[1] + popt[2]

    mag_sorter = np.argsort(asd)[::-1]

    peaks = []
    found_peaks = 0
    search_index = 0
    while found_peaks < npeak and search_index < nfreq:
        too_close = False

        freq_val = freqs[mag_sorter[search_index]]
        if len(band):
            if freq_val < band[0] or freq_val > band[1]:
                search_index += 1
                continue

        for peak in peaks:
            peak_loc = peak[0]
            if np.abs(freq_val - peak_loc) < exclude_df*df:
                too_close = True
                break

        if not too_close:
            asd_val = asd[mag_sorter[search_index]]
            if adjust_baseline:
                asd_val *= popt[0] * freq_val**popt[1] + popt[2]
            peaks.append([freq_val, asd_val])
            found_peaks += 1

        search_index += 1

    if plot:
        plot_pdet([peaks, []], freqs, np.abs(fft), loglog=True)

    return np.array(peaks)






def track_spectral_feature(peaks_list, init_features=[], first_fft=(), \
                           allowed_jumps=0.05):
    '''Function to track a spectral feature over successive integrations,
       making use of paralleization.

            peaks_list : the list of successive FFT peaks (found by the
                function find_fft_peaks()) in which we're trying to 
                track a feature

            init_features : list of initial feature locations used to seed
                the tracking. if empty, it will plot the first ASD and 
                ask for user input

            allowed_jump : fraction of feature fequency which it's allowed
                to jump between successive integrations
       '''

    if not len(init_features) and not len(first_fft):
        raise ValueError('No input features, and no first_fft to identify features')

    ### If initial features are not given, plot the first ASD and request user
    ### input to seed the feature tracking
    if not len(init_features):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Identify Frequencies of Interest')
        ax.loglog(first_fft[0], np.abs(first_fft[1]))
        fig.tight_layout()
        plt.show()

        init_str = input('Enter frequencies separated by comma: ')
        init_features = list(map(float, init_str.split(',')))

    if not iterable(allowed_jumps):
        allowed_jumps = [allowed_jumps for i in range(len(init_features))]


    ### Loop over each of the features we're interested in and track it across
    ### the successive integrations
    feature_lists = []
    for i, init_feature in enumerate(init_features):
        clist = []
        feature = [init_feature, 0.0]

        ### Loop over all the integrations
        for j, peaks in enumerate(peaks_list):

            try:
                ### Compute the distances between the current feature location
                ### and the frequency of all the peaks found for a particular
                ### integration
                distances = np.abs(peaks[:,0] - feature[0])

                ### Sort the distances and find which are within the allowed jump
                sorter = np.argsort(distances)
                close_enough = distances[sorter] < allowed_jumps[i] * feature[0]

                ### Sort the peaks and sub-select the ones that are close enough
                peaks_sorted = peaks[sorter,:]
                peaks_valid = peaks_sorted[close_enough,:]

                ### Try to find the biggest peak within those that are close enough.
                ### Sometimes this fails if the peak gets anomalously small or lost
                ### in the noise, since peaks_valid might be an empty array                
                feature_ind = np.argmax(peaks_valid[:3,1])
                feature = peaks_valid[feature_ind]

                clist.append(feature)

            except:
                clist.append([0.0, 0.0])

        ### Add the result from tracking this specific feature
        feature_lists.append(clist)

    return np.array(feature_lists)





def refine_pdet(peaks, xdat, ydat, half_window=5, sort=False):
    '''Function to refine the output from any of the many peakdetection
       algorithms in the 'peakdetect' library often used. Sometimes,
       result is off by a bin or two for some reason.'''

    pospeaks = peaks[0]
    negpeaks = peaks[1]

    new_pospeaks = []
    new_negpeaks = []

    for pospeak in pospeaks:
        xind = np.argmin(np.abs(xdat - pospeak[0]))
        inds = np.arange(len(xdat))

        mask = (inds > xind - half_window) * (inds < xind + half_window)
        maxind = np.argmax(ydat*mask)

        new_pospeaks.append([xdat[maxind], ydat[maxind]])

    for negpeak in negpeaks:
        xind = np.argmin(np.abs(xdat - negpeak[0]))
        inds = np.arange(len(xdat))

        mask = (inds > xind - half_window) * (inds < xind + half_window)
        minind = np.argmax((1.0/ydat)*mask)

        new_negpeaks.append([xdat[minind], ydat[minind]])

    if sort:
        try:
            new_pospeaks = sorted(new_pospeaks, key = lambda x: x[1])
        except:
            new_pospeaks = []

        try:
            new_negpeaks = sorted(new_negpeaks, key = lambda x: x[1])
        except:
            new_negpeaks = []

    return [new_pospeaks, new_negpeaks]






def plot_pdet(peaks, xdat, ydat, loglog=False, show=True):
    '''Function to plot the output from any of the many peakdetection
       algorithms in the 'peakdetect' library often used. Useful for 
       figuring out exactly what algorithm parameters are most robust
       for a particular dataset.'''

    fig, ax = plt.subplots(1,1)
    if loglog:
        ax.loglog(xdat, ydat, label='data', zorder=1)
    else:
        ax.plot(xdat, ydat, label='data', zorder=1)

    pospeaks = peaks[0]
    negpeaks = peaks[1]

    for peak in pospeaks:
        ax.scatter([peak[0]], [peak[1]], zorder=2, \
                   color='r', marker='X', s=30)

    for peak in negpeaks:
        ax.scatter([peak[0]], [peak[1]], zorder=2, \
                   color='k', marker='X', s=30)

    if show:
        plt.show()











def print_quadrant_indices():
    outstr = '\n'
    outstr += '     Quadrant diode indices:      \n'
    outstr += '   (looking at sensing elements)  \n'
    outstr += '                                  \n'
    outstr += '                                  \n'
    outstr += '              top                 \n'
    outstr += '          ___________             \n'
    outstr += '         |     |     |            \n'
    outstr += '         |  2  |  0  |            \n'
    outstr += '  left   |_____|_____|   right    \n'
    outstr += '         |     |     |            \n'
    outstr += '         |  3  |  1  |            \n'
    outstr += '         |_____|_____|            \n'
    outstr += '                                  \n'
    outstr += '             bottom               \n'
    outstr += '\n'
    print(outstr)





def print_electrode_indices():
    outstr = '\n'
    outstr += '        Electrode face indices:            \n'
    outstr += '                                           \n'
    outstr += '                                           \n'
    outstr += '                  top (1)                  \n'
    outstr += '                               back (4)    \n'
    outstr += '                  +---------+  cantilever  \n'
    outstr += '                 /         /|              \n'
    outstr += '                /    1    / |              \n'
    outstr += '               /         /  |              \n'
    outstr += '   left (6)   +---------+   |   right (5)  \n'
    outstr += '   input      |         | 5 |   output     \n'
    outstr += '              |         |   +              \n'
    outstr += '              |    3    |  /               \n'
    outstr += '              |         | /                \n'
    outstr += '              |         |/                 \n'
    outstr += ' front (3)    +---------+                  \n'
    outstr += ' bead dropper                              \n'
    outstr += '                 bottom (2)                \n'
    outstr += '                                           \n'
    outstr += '                                           \n'
    outstr += '      cantilever (0),   shield (7)         \n'
    outstr += '\n'
    print(outstr)





