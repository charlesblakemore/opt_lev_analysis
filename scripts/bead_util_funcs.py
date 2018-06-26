import h5py, os, re, glob, time, sys, fnmatch
import numpy as np
import datetime as dt
import dill as pickle 

from obspy.signal.detrend import polynomial

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.mlab as mlab

import scipy.interpolate as interp
import scipy.optimize as optimize
import scipy.signal as signal
import scipy

import configuration
import transfer_func_util as tf

import warnings

#######################################################
# This module has basic utility functions for analyzing bead
# data. In particular, this module has the basic data
# loading function, file finding/sorting, colormapping,
# FFT normalization, spatial binning etc.
# 
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
# ----------------------------------------------------
### The DataFile class is stored in a companion module
### bead_util, which imports these helper functions
# ----------------------------------------------------
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
# READ THIS PART  :  READ THIS PART  :  READ THIS PART
#
# This version has been significantly trimmed from previous
# bead_util in an attempt to force modularization.
# Previous code for millicharge and chameleon data
# can be found by reverting opt_lev_analysis
#######################################################


#### Generic Helper functions

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
        max_bar_len = 80 - len(suffix) - 15
        if bar_len > max_bar_len:
            bar_len = max_bar_len

    if count == total - 1:
        percents = 100.0
        bar = '#' * bar_len
    else:
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ... %s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()
    
    if (count == total - 1) and newline:
        print



def get_color_map( n, cmap='jet' ):
    '''Gets a map of n colors from cold to hot for use in
       plotting many curves.

           INPUTS: n, length of color array to make
                   cmap, color map for final output

           OUTPUTS: outmap, color map in rgba format'''

    cNorm  = colors.Normalize(vmin=0, vmax=n)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap) #cmap='viridis')
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

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



def fft_norm(N, fsamp):
    return np.sqrt(2 / (N * fsamp))


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
    

def make_all_pardirs(path):
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
            os.mkdir(parent_dir)



def find_all_fnames(dirlist, ext='.h5', sort=True, exclude_fpga=True):
    '''Finds all the filenames matching a particular extension
       type in the directory and its subdirectories .

       INPUTS: dirlist, list of directory names to loop over
               ext, file extension you're looking for
               sort, boolean specifying whether to do a simple sort

       OUTPUTS: files, list of files names as strings'''

    print "Finding files in: "
    print dirlist
    sys.stdout.flush()

    was_list = True

    lengths = []
    files = []

    if type(dirlist) == str:
        dirlist = [dirlist]
        was_list = False

    for dirname in dirlist:
        for root, dirnames, filenames in os.walk(dirname):
            for filename in fnmatch.filter(filenames, '*' + ext):
                if ('_fpga.h5' in filename) and exclude_fpga:
                    continue
                files.append(os.path.join(root, filename))
        if was_list:
            if len(lengths) == 0:
                lengths.append(len(files))
            else:
                lengths.append(len(files) - np.sum(lengths)) 
            
    if sort:
        # Sort files based on final index
        files.sort(key = find_str)

    if len(files) == 0:
        print "DIDN'T FIND ANY FILES :("

    print "Found %i files..." % len(files)
    if was_list:
        return files, lengths
    else:
        return files



def sort_files_by_timestamp(files):
    '''Pretty self-explanatory function.'''
    files = [(get_hdf5_time(path), path) for path in files]
    files.sort(key = lambda x: (x[0]))
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

def copy_attribs(attribs):
    '''copies an hdf5 attributes into a new dictionary 
       so the original file can be closed.'''
    new_dict = {}
    for k in attribs.keys():
        new_dict[k] = attribs[k]
    return new_dict



def getdata(fname, gain_error=1.0):
    '''loads a .h5 file from a path into data array and 
       attribs dictionary, converting ADC bits into 
       volatage. The h5 file is closed.'''

    #factor to convert between adc bits and voltage 
    adc_fac = (configuration.adc_params["adc_res"] - 1) / \
               (2. * configuration.adc_params["adc_max_voltage"])

    try:
        f = h5py.File(fname,'r')
        dset = f['beads/data/pos_data']
        dat = np.transpose(dset)
        dat = dat / adc_fac
        attribs = copy_attribs(dset.attrs)
        f.close()

    except (KeyError, IOError):
        print "Warning, got no keys for: ", fname
        dat = []
        attribs = {}
        f = []

    return dat, attribs

def get_hdf5_time(fname):
    try:
        f = h5py.File(fname,'r')
        dset = f['beads/data/pos_data']
        attribs = copy_attribs(dset.attrs)
        f.close()

    except (KeyError, IOError):
        print "Warning, got no keys for: ", fname
        attribs = {}

    return attribs["Time"]

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
    for k in dic.keys():
        out_dict[k] = vec[dic[k]]
    return out_dict 




def spatial_bin(drive, resp, dt, nbins=100, nharmonics=10, width=0, \
                sg_filter=False, sg_params=[3,1], verbose=True):
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
       	        width, filter width in Hertz [Hz]
                sg_filter, boolean value indicating use of a Savitsky-Golay 
                            filter for final smoothing of resp(drive)
                sg_params, parameters of the savgol filter 
                            (see scipy.signal.savgol_filter for explanation)

       OUTPUT:  drivevec, vector of drive values, monotonically increasing
                respvec, resp as a function of drivevec'''

    def fit_fun(t, A, f, phi, C):
        return A * np.sin(2 * np.pi * f * t + phi) + C

    Nsamp = len(drive)
    if len(resp) != Nsamp:
        if verbose:
            print "Data Error: x(t) and f(t) don't have the same length"
            sys.stdout.flush()
        return

    # Generate t array
    t = np.linspace(0, len(drive) - 1, len(drive)) * dt

    # Generate FFTs for filtering
    drivefft = np.fft.rfft(drive)
    respfft = np.fft.rfft(resp)
    freqs = np.fft.rfftfreq(len(drive), d=dt)

    # Find the drive frequency, ignoring the DC bin
    fund_ind = np.argmax( np.abs(drivefft[1:]) ) + 1
    drive_freq = freqs[fund_ind]

    meandrive = np.mean(drive)
    mindrive = np.min(drive)
    maxdrive = np.max(drive)

    meanresp = np.mean(resp)

    # Build the notch filter
    drivefilt = np.zeros(len(drivefft)) #+ np.random.randn(len(drivefft))*1.0e-3
    drivefilt[fund_ind] = 1.0

    # Error message triggered by verbose option
    if verbose:
        if ( np.abs(drivefft[fund_ind-1]) > 0.01 * np.abs(drivefft[fund_ind]) or \
             np.abs(drivefft[fund_ind+1]) > 0.01 * np.abs(drivefft[fund_ind]) ):
            print "More than 1% power in neighboring bins: spatial binning may be suboptimal"
            sys.stdout.flush()

    # Expand the filter to more than a single bin. This can introduce artifacts
    # that appear like lissajous figures in the resp vs. drive final result
    if width:
        lower_ind = np.argmin(np.abs(drive_freq - 0.5 * width - freqs))
        upper_ind = np.argmin(np.abs(drive_freq + 0.5 * width - freqs))
        drivefilt[lower_ind:upper_ind+1] = drivefilt[fund_ind]

    # Generate an array of harmonics
    harms = np.array([x+2 for x in range(nharmonics)])

    # Loop over harmonics and add them to the filter
    for n in harms:
        harm_ind = np.argmin( np.abs(n * drive_freq - freqs) )
        drivefilt[harm_ind] = 1.0 
        if width:
            h_lower_ind = harm_ind - (fund_ind - lower_ind)
            h_upper_ind = harm_ind + (upper_ind - fund_ind)
            drivefilt[h_lower_ind:h_upper_ind+1] = drivefilt[harm_ind]

    # Apply the filter to both drive and response
    #drivefilt = np.ones_like(drivefilt)
    #drivefilt[0] = 0
    drivefft_filt = drivefilt * drivefft
    respfft_filt = drivefilt * respfft

    #plt.loglog(freqs, np.abs(respfft))
    #plt.loglog(freqs[drivefilt>0], np.abs(respfft[drivefilt>0]), 'x', ms=10)
    #plt.show()

    # Reconstruct the filtered data
    
    #plt.loglog(freqs, np.abs(drivefft_filt))
    #plt.show()

    fac = np.sqrt(2) * fft_norm(len(t),1.0/(t[1]-t[0])) * np.sqrt(freqs[1] - freqs[0])

    #drive_r = np.zeros(len(t)) + meandrive
    #for ind, freq in enumerate(freqs[drivefilt>0]):
    #    drive_r += fac * np.abs(drivefft_filt[drivefilt>0][ind]) * \
    #               np.cos( 2 * np.pi * freq * t + \
    #                       np.angle(drivefft_filt[drivefilt>0][ind]) )
    drive_r = np.fft.irfft(drivefft_filt) + meandrive

    #resp_r = np.zeros(len(t))
    #for ind, freq in enumerate(freqs[drivefilt>0]):
    #    resp_r += fac * np.abs(respfft_filt[drivefilt>0][ind]) * \
    #              np.cos( 2 * np.pi * freq * t + \
    #                      np.angle(respfft_filt[drivefilt>0][ind]) )
    resp_r = np.fft.irfft(respfft_filt) #+ meanresp

    # Sort reconstructed data, interpolate and resample
    mindrive = np.min(drive_r)
    maxdrive = np.max(drive_r)

    grad = np.gradient(drive_r)

    sortinds = drive_r.argsort()
    drive_r = drive_r[sortinds]
    resp_r = resp_r[sortinds]

    #plt.plot(drive_r, resp_r, '.')
    #plt.show()

    ginds = grad[sortinds] < 0

    bin_spacing = (maxdrive - mindrive) * (1.0 / nbins)
    drivevec = np.linspace(mindrive+0.5*bin_spacing, maxdrive-0.5*bin_spacing, nbins)

    respvec = []
    for bin_loc in drivevec:
        inds = (drive_r > bin_loc - 0.5*bin_spacing) * (drive_r < bin_loc + 0.5*bin_spacing)
        val = np.mean( resp_r[inds] )
        respvec.append(val)

    respvec = np.array(respvec)
    
    #plt.plot(drive_r, resp_r)
    #plt.plot(drive_r[ginds], resp_r[ginds], linewidth=2)
    #plt.plot(drive_r[np.invert(ginds)], resp_r[np.invert(ginds)], linewidth=2)
    #plt.plot(drivevec, respvec, linewidth=5)
    #plt.show()

    #sortinds = drive_r.argsort()
    #interpfunc = interp.interp1d(drive_r[sortinds], resp_r[sortinds], \
    #                             bounds_error=False, fill_value='extrapolate')

    #respvec = interpfunc(drivevec)
    if sg_filter:
        respvec = signal.savgol_filter(respvec, sg_params[0], sg_params[1])

    return drivevec, respvec













########################################################
########################################################
########### Functions to Handle FPGA data ##############
########################################################
########################################################





def extract_quad(quad_dat, timestamp, verbose=False):
    '''Reads a stream of I32s, finds the first timestamp,
       then starts de-interleaving the demodulated data
       from the FPGA'''
    
    if timestamp == 0.0:
        # if no timestamp given, use current time
        # and set the timing threshold for 1 month.
        # This threshold is used to identify the timestamp 
        # in the stream of I32s
        timestamp = time.time()
        diff_thresh = 31.0 * 24.0 * 3600.0
    else:
        timestamp = timestamp * (10.0**(-9))
        diff_thresh = 60.0

    for ind, dat in enumerate(quad_dat): ## % 12
        # Assemble time stamp from successive I32s, since
        # it's a 64 bit object
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            high = np.int32(quad_dat[ind])
            low = np.int32(quad_dat[ind+1])
            dattime = (high.astype(np.uint64) << np.uint64(32)) \
                      + low.astype(np.uint64)

        # Time stamp from FPGA is a U64 with the UNIX epoch 
        # time in nanoseconds, synced to the host's clock
        if (np.abs(timestamp - float(dattime) * 10**(-9)) < diff_thresh):
            if verbose:
                print "found timestamp  : ", float(dattime) * 10**(-9)
                print "comparison time  : ", timestamp 
            break

    # Once the timestamp has been found, select each dataset
    # wit thhe appropriate decimation of the primary array
    quad_time_high = np.int32(quad_dat[ind::12])
    quad_time_low = np.int32(quad_dat[ind+1::12])
    if len(quad_time_low) != len(quad_time_high):
        quad_time_high = quad_time_high[:-1]
    quad_time = quad_time_high.astype(np.uint64) << np.uint64(32) \
                  + quad_time_low.astype(np.uint64)

    amp = [quad_dat[ind+2::12], quad_dat[ind+3::12], quad_dat[ind+4::12], \
           quad_dat[ind+5::12], quad_dat[ind+6::12]]
    phase = [quad_dat[ind+7::12], quad_dat[ind+8::12], quad_dat[ind+9::12], \
             quad_dat[ind+10::12], quad_dat[ind+11::12]]
            

    # Since the FIFO read request is asynchronous, sometimes
    # the timestamp isn't first to come out, but the total amount of data
    # read out is a multiple of 12 (2 time + 5 amp + 5 phase) so an
    # amplitude or phase channel ends up with less samples.
    # The following is coded very generally

    min_len = 10.0**9  # Assumes we never more than 1 billion samples
    for ind in [0,1,2,3,4]:
        if len(amp[ind]) < min_len:
            min_len = len(amp[ind])
        if len(phase[ind]) < min_len:
            min_len = len(phase[ind])

    # Re-size everything by the minimum length and convert to numpy array
    quad_time = np.array(quad_time[:min_len])
    for ind in [0,1,2,3,4]:
        amp[ind]   = amp[ind][:min_len]
        phase[ind] = phase[ind][:min_len]
    amp = np.array(amp)
    phase = np.array(phase)
      

    return quad_time, amp, phase






def extract_xyz(xyz_dat, timestamp, verbose=False):
    '''Reads a stream of I32s, finds the first timestamp,
       then starts de-interleaving the demodulated data
       from the FPGA'''
    
    if timestamp == 0.0:
        # if no timestamp given, use current time
        # and set the timing threshold for 1 month.
        # This threshold is used to identify the timestamp 
        # in the stream of I32s
        timestamp = time.time()
        diff_thresh = 31.0 * 24.0 * 3600.0
    else:
        timestamp = timestamp * (10.0**(-9))
        diff_thresh = 60.0


    for ind, dat in enumerate(xyz_dat):
        # Assemble time stamp from successive I32s, since
        # it's a 64 bit object
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            high = np.int32(xyz_dat[ind])
            low = np.int32(xyz_dat[ind+1])
            dattime = (high.astype(np.uint64) << np.uint64(32)) \
                      + low.astype(np.uint64)

        # Time stamp from FPGA is a U64 with the UNIX epoch 
        # time in nanoseconds, synced to the host's clock
        if (np.abs(timestamp - float(dattime) * 10**(-9)) < diff_thresh):
            tind = ind
            if verbose:
                print "found timestamp  : ", float(dattime) * 10**(-9)
                print "comparison time  : ", timestamp 
            break

    # Once the timestamp has been found, select each dataset
    # wit thhe appropriate decimation of the primary array
    xyz_time_high = np.int32(xyz_dat[tind::11])
    xyz_time_low = np.int32(xyz_dat[tind+1::11])
    if len(xyz_time_low) != len(xyz_time_high):
        xyz_time_high = xyz_time_high[:-1]

    xyz_time = xyz_time_high.astype(np.uint64) << np.uint64(32) \
                  + xyz_time_low.astype(np.uint64)

    xyz = [xyz_dat[tind+4::11], xyz_dat[tind+5::11], xyz_dat[tind+6::11]]
    xy_2 = [xyz_dat[tind+2::11], xyz_dat[tind+3::11]]
    xyz_fb = [xyz_dat[tind+8::11], xyz_dat[tind+9::11], xyz_dat[tind+10::11]]
    
    sync = np.int32(xyz_dat[tind+7::11])

    #plt.plot(np.int32(xyz_dat[tind+1::9]).astype(np.uint64) << np.uint64(32) \
    #         + np.int32(xyz_dat[tind::9]).astype(np.uint64) )
    #plt.show()

    # Since the FIFO read request is asynchronous, sometimes
    # the timestamp isn't first to come out, but the total amount of data
    # read out is a multiple of 5 (2 time + X + Y + Z) so the Z
    # channel usually  ends up with less samples.
    # The following is coded very generally

    min_len = 10.0**9  # Assumes we never more than 1 billion samples
    for ind in [0,1,2]:
        if len(xyz[ind]) < min_len:
            min_len = len(xyz[ind])
        if len(xyz_fb[ind]) < min_len:
            min_len = len(xyz_fb[ind])
        if ind != 2:
            if len(xy_2[ind]) < min_len:
                min_len = len(xy_2[ind])

    # Re-size everything by the minimum length and convert to numpy array
    xyz_time = np.array(xyz_time[:min_len])
    sync = np.array(sync[:min_len])
    for ind in [0,1,2]:
        xyz[ind]    = xyz[ind][:min_len]
        xyz_fb[ind] = xyz_fb[ind][:min_len]
        if ind != 2:
            xy_2[ind] = xy_2[ind][:min_len]
    xyz = np.array(xyz)
    xyz_fb = np.array(xyz_fb)
    xy_2 = np.array(xy_2)

    return xyz_time, xyz, xy_2, xyz_fb, sync







def get_fpga_data(fname, timestamp=0.0, verbose=False):
    '''Raw data from the FPGA is saved in an hdf5 (.h5) 
       file in the form of 3 continuous streams of I32s
       (32-bit integers). This script reads it out and 
       makes sense of it for post-processing'''

    # Open the file and bring datasets into memory
    try:
        f = h5py.File(fname,'r')
        dset0 = f['beads/data/raw_data']
        dset1 = f['beads/data/quad_data']
        dset2 = f['beads/data/pos_data']
        dat0 = np.transpose(dset0)
        dat1 = np.transpose(dset1)
        dat2 = np.transpose(dset2)
        f.close()

    # Shit failure mode. What kind of sloppy coding is this
    except (KeyError, IOError):
        if verbose:
            print "Warning, got no keys for: ", fname
        dat0 = []
        dat1 = []
        dat2 = []
        attribs = {}
        try:
            f.close()
        except:
            if verbose:
                print "couldn't close file, not sure if it's open"

    if len(dat1):
        # Use subroutines to handle each type of data
        # raw_time, raw_dat = extract_raw(dat0, timestamp)
        raw_time, raw_dat = (None, None)
        quad_time, amp, phase = extract_quad(dat1, timestamp, verbose=verbose)
        xyz_time, xyz, xy_2, xyz_fb, sync = extract_xyz(dat2, timestamp, verbose=verbose)
    else:
        raw_time, raw_dat = (None, None)
        quad_time, amp, phase = (None, None, None)
        xyz_time, xyz, xy_2, xyz_fb, sync = (None, None, None, None, None)

    # Assemble the output as a human readable dictionary
    out = {'raw_time': raw_time, 'raw_dat': raw_dat, \
           'xyz_time': xyz_time, 'xyz': xyz, 'xy_2': xy_2, \
           'fb': xyz_fb, 'quad_time': quad_time, 'amp': amp, \
           'phase': phase, 'sync': sync}

    return out



def sync_and_crop_fpga_data(fpga_dat, timestamp, nsamp, encode_bin, \
                            encode_len=500, plot_sync=False):
    '''Align the psuedo-random bits the DAQ card spits out to the FPGA
       to synchronize the acquisition of the FPGA.'''

    out = {}
    notNone = False
    for key in fpga_dat:
        if type(fpga_dat[key]) != type(None):
            notNone = True
    if not notNone:
        return fpga_dat

    # The FIFOs to read the raw data aren't even setup yet
    # so this is just some filler code
    out['raw_time'] = fpga_dat['raw_time']
    out['raw_dat'] = fpga_dat['raw_dat']

    # Cutoff irrelevant zeros
    if len(encode_bin) < encode_len:
        encode_len = len(encode_bin)
    encode_bin = np.array(encode_bin[:encode_len])

    # Load the I32 representation of the synchronization data
    # At each 500 kHz sample of the FPGA, the state of the sync
    # digital pin is sampled: True->(I32+1), False->(I32-1)
    sync_dat = fpga_dat['sync']

    #plt.plot(sync_dat)
    #plt.show()

    sync_dat = sync_dat[:len(encode_bin) * 10]
    sync_dat_bin = np.zeros(len(sync_dat)) + 1.0 * (np.array(sync_dat) > 0)

    dat_inds = np.linspace(0,len(sync_dat)-1,len(sync_dat))

    # Find correct starting sample to sync with the DAQ by
    # maximizing the correlation between the FPGA's digitized
    # sync line and the encoded bits from the DAQ file.
    # Because of how the DAQ tasks are setup, the sync bits come
    # out for the first Nsync samples, and then again after 
    # Nsamp_DAQ samples. Thus we take the maximum of the correlation
    # found in the first half of the array corr
    corr = np.correlate(sync_dat_bin, encode_bin)
    off_ind = np.argmax(corr[:int(0.5*len(corr))])

    if plot_sync:
        # Make an array of indices for plotting
        inds = np.linspace(0,encode_len-1,encode_len)
        dat_inds = np.linspace(0,len(sync_dat)-1,len(sync_dat))

        plt.step(inds, encode_bin, lw=1.5, where='pre', label='encode_bits', \
                 linestyle='dotted')
        plt.step(dat_inds-off_ind, sync_dat_bin, where='pre', label='aligned_data', \
                 alpha=0.5)
        plt.xlim(-5, encode_len+10)

        plt.legend()
        plt.show()

    # Find the xyz and quad timestamps that match the daqmx first 
    # sample timestamp

    # Crop the xyz arrays
    out['xyz_time'] = fpga_dat['xyz_time'][off_ind:off_ind+nsamp]
    out['xyz'] = fpga_dat['xyz'][:,off_ind:off_ind+nsamp]
    out['xy_2'] = fpga_dat['xy_2'][:,off_ind:off_ind+nsamp]
    out['fb'] = fpga_dat['fb'][:,off_ind:off_ind+nsamp]
    out['sync'] = sync_dat_bin[off_ind:off_ind+nsamp]

    # Crop the quad arrays
    out['quad_time'] = fpga_dat['quad_time'][off_ind:off_ind+nsamp]
    out['amp'] = fpga_dat['amp'][:,off_ind:off_ind+nsamp]
    out['phase'] = fpga_dat['phase'][:,off_ind:off_ind+nsamp]

    # return data in the same format as it was given
    return out
