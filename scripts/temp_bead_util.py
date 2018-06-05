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
        diff_thresh = 60.0

    writing_data = False
    quad_ind = 0

    quad_time = []
    amp = [[], [], [], [], []]
    phase = [[], [], [], [], []]
    for ind, dat in enumerate(quad_dat):

        # Data in the 'quad' FIFO comes through as:
        # time_MSB -> time_LSB ->
        # amp0     -> amp1     -> amp2   -> amp3   -> amp4   ->
        # phase0   -> phase1   -> phase2 -> phase3 -> phase4 ->
        # and then repeats. Amplitude and phase variables are 
        # arbitrarily scaled so thinking of them as 32-bit integers
        # is okay. We just care about the bits anyway. The amplitude
        # is unsigned, so we get an extra bit of precision there
        if writing_data:
            if quad_ind == 0:
                high = np.uint32(quad_dat[ind])
                low = np.uint32(quad_dat[ind+1])
                dattime = (high.astype(np.uint64) << np.uint64(32)) \
                           + low.astype(np.uint64)
                quad_time.append(dattime)
            elif quad_ind == 2:
                amp[0].append(dat.astype(np.uint32))
            elif quad_ind == 3:
                amp[1].append(dat.astype(np.uint32))
            elif quad_ind == 4:
                amp[2].append(dat.astype(np.uint32))
            elif quad_ind == 5:
                amp[3].append(dat.astype(np.uint32))
            elif quad_ind == 6:
                amp[4].append(dat.astype(np.uint32))
            elif quad_ind == 7:
                phase[0].append(dat)
            elif quad_ind == 8:
                phase[1].append(dat)
            elif quad_ind == 9:
                phase[2].append(dat)
            elif quad_ind == 10:
                phase[3].append(dat)
            elif quad_ind == 11:
                phase[4].append(dat)
            
            quad_ind += 1
            quad_ind = quad_ind % 12

                # Check for the timestamp
        if not writing_data and quad_ind == 0:
            # Assemble time stamp from successive I32s, since
            # it's a 64 bit object
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
                quad_time.append(dattime)
                quad_ind += 1
                writing_data = True

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
        diff_thresh = 60.0

    writing_data = False
    xyz_ind = 0

    xyz_time = []
    xyz = [[], [], []]

    for ind, dat in enumerate(xyz_dat):

        # Data in the 'xyz' FIFO comes through as:
        # time_MSB -> time_LSB ->
        # X        -> Y        -> Z   -> 
        # and then repeats. Position  variables are 
        # arbitrarily scaled so thinking of them as 32-bit integers
        # is okay. We just care about the bits anyway
        if writing_data:
            if xyz_ind == 0:
                high = np.uint32(xyz_dat[ind])
                low = np.uint32(xyz_dat[ind+1])
                dattime = (high.astype(np.uint64) << np.uint64(32)) \
                           + low.astype(np.uint64)
                xyz_time.append(dattime)
            elif xyz_ind == 2:
                xyz[0].append(dat)
            elif xyz_ind == 3:
                xyz[1].append(dat)
            elif xyz_ind == 4:
                xyz[2].append(dat)
            
            xyz_ind += 1
            xyz_ind = xyz_ind % 5

        # Check for the timestamp
        if not writing_data and xyz_ind == 0:
            # Assemble time stamp from successive I32s, since
            # it's a 64 bit object
            high = np.int32(xyz_dat[ind])
            low = np.int32(xyz_dat[ind+1])
            dattime = (high.astype(np.uint64) << np.uint64(32)) \
                        + low.astype(np.uint64)

            # Time stamp from FPGA is a U64 with the UNIX epoch 
            # time in nanoseconds, synced to the host's clock
            if (np.abs(timestamp - float(dattime) * 10**(-9)) < diff_thresh):
                if verbose:
                    print "found timestamp  : ", float(dattime) * 10**(-9)
                    print "comparison time  : ", timestamp 
                xyz_time.append(dattime)
                xyz_ind += 1
                writing_data = True

    # Since the FIFO read request is asynchronous, sometimes
    # the timestamp isn't first to come out, but the total amount of data
    # read out is a multiple of 5 (2 time + X + Y + Z) so the Z
    # channel usually  ends up with less samples.
    # The following is coded very generally

    min_len = 10.0**9  # Assumes we never more than 1 billion samples
    for ind in [0,1,2]:
        if len(xyz[ind]) < min_len:
            min_len = len(xyz[ind])

    # Re-size everything by the minimum length and convert to numpy array
    xyz_time = np.array(xyz_time[:min_len])
    for ind in [0,1,2]:
        xyz[ind]   = xyz[ind][:min_len]
    xyz = np.array(xyz)        

    return xyz_time, xyz







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
        if verbsose:
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

    # Use subroutines to handle each type of data
    # raw_time, raw_dat = extract_raw(dat0, timestamp)
    raw_time, raw_dat = (None, None)
    quad_time, amp, phase = extract_quad(dat1, timestamp, verbose=verbose)
    xyz_time, xyz = extract_xyz(dat2, timestamp, verbose=verbose)

    # Assemble the output as a human readable dictionary
    out = {'raw_time': raw_time, 'raw_dat': raw_dat, \
           'xyz_time': xyz_time, 'xyz': xyz, \
           'quad_time': quad_time, 'amp': amp, \
           'phase': phase}

    return out


