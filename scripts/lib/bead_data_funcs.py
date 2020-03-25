import h5py, os, re, glob, time, sys, fnmatch, inspect
import subprocess, math, xmltodict, traceback
import numpy as np
import datetime as dt
import dill as pickle 

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.mlab as mlab

import scipy.interpolate as interp
import scipy.optimize as optimize
import scipy.signal as signal
import scipy.stats as stats
import scipy.constants as constants
import scipy

import configuration
import transfer_func_util as tf

import warnings


def copy_attribs(attribs):
    '''copies an hdf5 attributes into a new dictionary 
       so the original file can be closed.'''
    new_dict = {}
    for k in list(attribs.keys()):
        new_dict[k] = attribs[k]
    return new_dict


def getdata(fname, gain_error=1.0, verbose=False):
    '''loads a .h5 file from a path into data array and 
       attribs dictionary, converting ADC bits into 
       volatage. The h5 file is closed.'''

    #factor to convert between adc bits and voltage 
    adc_fac = (configuration.adc_params["adc_res"] - 1) / \
               (2. * configuration.adc_params["adc_max_voltage"])

    message = ''
    try:
        try:
            f = h5py.File(fname,'r')
        except Exception:
            message = "Can't find/open HDF5 file : " + fname
            traceback.print_exc()
            raise

        try:
            dset = f['beads/data/pos_data']
        except Exception:
            message = "Can't find any dataset in : " + fname
            f.close()
            traceback.print_exc()
            raise

        dat = np.transpose(dset)
        dat = dat / adc_fac
        attribs = copy_attribs(dset.attrs)
        if attribs == {}:
            attribs = load_xml_attribs(fname)
        f.close()

    except Exception:
        print(message)
        dat = []
        attribs = {}
        f = []
        traceback.print_exc()

    return dat, attribs





def getdata_new(fname, gain_error=1.0, verbose=False):
    '''loads a .h5 file from a path into data array and 
       attribs dictionary, converting ADC bits into 
       volatage. The h5 file is closed.'''

    message = ''
    try:
        try:
            f = h5py.File(fname,'r')
        except Exception:
            message = "Can't find/open HDF5 file : " + fname
            traceback.print_exc()
            raise

        try:
            dset1 = np.copy(f['pos_data'])
            dset2 = np.copy(f['quad_data'])

            try:
                dset3 = np.copy(f['spin_data'])
            except Exception:
                dset3 = []
                print('No spin data')

            try:
                dset4 = np.copy(f['cant_data'])
                cant_settings = np.copy(f['cantilever_settings'])
            except Exception:
                dset4 = []
                print('No attractor data')

            try:
                dset5 = np.copy(f['electrode_data'])
            except Exception:
                dset5 = []
                print('No electrode data')

            attribs = copy_attribs(f.attrs)

        except Exception:
            message = "Can't find any dataset in : " + fname
            f.close()
            traceback.print_exc()
            raise

        if attribs == {}:
            try:
                attribs = load_xml_attribs(fname)
            except Exception:
                message = "Can't find any attributes for : " + fname
                f.close()
                traceback.print_exc()
                raise

        f.close()

    except Exception:
        print(message)
        dat = []
        attribs = {}
        f = []
        traceback.print_exc()

    return dset1, dset2, dset3, dset4, dset5, attribs




def get_hdf5_time(fname):
    try:
        try:
            f = h5py.File(fname,'r')
            attribs = copy_attribs(f.attrs)
            f.close()
        except:
            print('HDF5 file has no attributes object...')

        if attribs == {}:
            attribs = load_xml_attribs(fname)
        

    except Exception:
        traceback.print_exc()
        # print "Warning, got no keys for: ", fname
        attribs = {}

    try:
        file_time = attribs["time"]
    except Exception:
        print("Couldn't find a value for the time...")
        print()
        traceback.print_exc()

    return file_time



def load_xml_attribs(fname, types=['DBL', 'Array', 'Boolean', 'String']):
    """LabVIEW Live HDF5 stopped saving datasets with attributes at some point.
    To get around this, the attribute cluster is saved to an XML string and 
    parsed into a dictionary here."""

    attr_fname = fname[:-3] + '.attr'

    xml = open(attr_fname, 'r').read()

    attr_dict = xmltodict.parse(xml)['Cluster']
    n_attr = int(attr_dict['NumElts'])

    new_attr_dict = {}
    for attr_type in types:
        try:
            c_list = attr_dict[attr_type]
        except Exception:
            traceback.print_exc()
            continue

        if type(c_list) != list:
            c_list = [c_list]

        for item in c_list:
            new_key = item['Name']

            # Keep the time as 64 bit unsigned integer
            if new_key == 'Time' or new_key == 'time':
                new_attr_dict['time'] = np.uint64(float(item['Val']))

            # Conver 32-bit integers to their correct datatype
            elif (attr_type == 'I32'):
                new_attr_dict[new_key] = np.int32(item['Val'])

            # Convert single numbers/bool from their xml string representation
            elif (attr_type == 'DBL') or (attr_type == 'Boolean'):
                new_attr_dict[new_key] = float(item['Val'])

            # Convert arrays of numbers from their parsed xml
            elif (attr_type == 'Array'):
                new_arr = []
                vals = item['DBL']
                for val in vals:
                    new_arr.append(float(val['Val']))
                new_attr_dict[new_key] = new_arr

            # Move string attributes to new attribute dictionary
            elif (attr_type == 'String'):
                new_attr_dict[new_key] = item['Val']

            # Catch-all for unknown attributes, keep as string
            else:
                print('Found an attribute whose type is unknown. Left as string...')
                new_attr_dict[new_key] = item['Val']

    # assert n_attr == len(new_attr_dict.keys())

    return new_attr_dict




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
        diff_thresh = 365.0 * 24.0 * 3600.0
    else:
        timestamp = timestamp * (10.0**(-9))
        diff_thresh = 60.0

    for ind, dat in enumerate(quad_dat): ## % 12
        # Assemble time stamp from successive I32s, since
        # it's a 64 bit object
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            high = np.uint32(quad_dat[ind])
            low = np.uint32(quad_dat[ind+1])
            dattime = (high.astype(np.uint64) << np.uint64(32)) \
                      + low.astype(np.uint64)

        # Time stamp from FPGA is a U64 with the UNIX epoch 
        # time in nanoseconds, synced to the host's clock
        if (np.abs(timestamp - float(dattime) * 10**(-9)) < diff_thresh):
            if verbose:
                print("found timestamp  : ", float(dattime) * 10**(-9))
                print("comparison time  : ", timestamp) 
            break

    # Once the timestamp has been found, select each dataset
    # wit thhe appropriate decimation of the primary array
    quad_time_high = np.uint32(quad_dat[ind::12])
    quad_time_low = np.uint32(quad_dat[ind+1::12])
    if len(quad_time_low) != len(quad_time_high):
        quad_time_high = quad_time_high[:-1]
    quad_time = np.left_shift(quad_time_high.astype(np.uint64), np.uint64(32)) \
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






def extract_quad_new(quad_dat, verbose=False):
    '''Reads a stream of I32s, finds the first timestamp,
       then starts de-interleaving the demodulated data
       from the FPGA'''

    quad_time_high = np.uint32(quad_dat[10::12])
    quad_time_low = np.uint32(quad_dat[11::12])
    if len(quad_time_low) != len(quad_time_high):
        quad_time_high = quad_time_high[:-1]
    quad_time = np.left_shift(quad_time_high.astype(np.uint64), np.uint64(32)) \
                  + quad_time_low.astype(np.uint64)

    amp = [quad_dat[0::12], quad_dat[1::12], quad_dat[2::12], \
           quad_dat[3::12], quad_dat[4::12]]
    phase = [quad_dat[5::12], quad_dat[6::12], quad_dat[7::12], \
             quad_dat[8::12], quad_dat[9::12]]
            

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
        # and set the timing threshold for 1 year.
        # This threshold is used to identify the timestamp 
        # in the stream of I32s
        timestamp = time.time()
        diff_thresh = 365.0 * 24.0 * 3600.0
    else:
        timestamp = timestamp * (10.0**(-9))
        # 2-minute difference allowed for longer integrations
        diff_thresh = 120.0


    for ind, dat in enumerate(xyz_dat):
        # Assemble time stamp from successive I32s, since
        # it's a 64 bit object
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            high = np.uint32(xyz_dat[ind])
            low = np.uint32(xyz_dat[ind+1])
            dattime = (high.astype(np.uint64) << np.uint64(32)) \
                      + low.astype(np.uint64)

        # Time stamp from FPGA is a U64 with the UNIX epoch 
        # time in nanoseconds, synced to the host's clock
        if (np.abs(timestamp - float(dattime) * 10**(-9)) < diff_thresh):
            tind = ind
            if verbose:
                print("found timestamp  : ", float(dattime) * 10**(-9))
                print("comparison time  : ", timestamp) 
            break

    # Once the timestamp has been found, select each dataset
    # wit thhe appropriate decimation of the primary array
    xyz_time_high = np.uint32(xyz_dat[tind::11])
    xyz_time_low = np.uint32(xyz_dat[tind+1::11])
    if len(xyz_time_low) != len(xyz_time_high):
        xyz_time_high = xyz_time_high[:-1]

    xyz_time = np.left_shift(xyz_time_high.astype(np.uint64), np.uint64(32)) \
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





def extract_xyz_new(xyz_dat, verbose=False):
    '''Reads a stream of I32s, finds the first timestamp,
       then starts de-interleaving the demodulated data
       from the FPGA'''
    
    xyz_time_high = np.uint32(xyz_dat[9::11])
    xyz_time_low = np.uint32(xyz_dat[10::11])
    if len(xyz_time_low) != len(xyz_time_high):
        xyz_time_high = xyz_time_high[:-1]

    xyz_time = np.left_shift(xyz_time_high.astype(np.uint64), np.uint64(32)) \
                  + xyz_time_low.astype(np.uint64)

    xyz = [xyz_dat[2::11], xyz_dat[3::11], xyz_dat[4::11]]
    xy_2 = [xyz_dat[0::11], xyz_dat[1::11]]
    sync = np.int32(xyz_dat[5::11])
    xyz_fb = [xyz_dat[6::11], xyz_dat[7::11], xyz_dat[8::11]]
    
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








def extract_power(pow_dat, timestamp, verbose=False):
    '''Reads a stream of I32s, finds the first timestamp,
       then starts de-interleaving the demodulated data
       from the FPGA'''

    interleave_num = 4
    
    if timestamp == 0.0:
        # if no timestamp given, use current time
        # and set the timing threshold for 1 year.
        # This threshold is used to identify the timestamp 
        # in the stream of I32s
        timestamp = time.time()
        diff_thresh = 365.0 * 24.0 * 3600.0
    else:
        timestamp = timestamp * (10.0**(-9))
        # 2-minute difference allowed for longer integrations
        diff_thresh = 120.0


    for ind, dat in enumerate(pow_dat):
        # Assemble time stamp from successive I32s, since
        # it's a 64 bit object
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            high = np.uint32(pow_dat[ind])
            low = np.uint32(pow_dat[ind+1])
            dattime = (high.astype(np.uint64) << np.uint64(32)) \
                      + low.astype(np.uint64)

        # Time stamp from FPGA is a U64 with the UNIX epoch 
        # time in nanoseconds, synced to the host's clock
        if (np.abs(timestamp - float(dattime) * 10**(-9)) < diff_thresh):
            tind = ind
            if verbose:
                print("found timestamp  : ", float(dattime) * 10**(-9))
                print("comparison time  : ", timestamp) 
            break

    # Once the timestamp has been found, select each dataset
    # with the appropriate decimation of the primary array
    pow_time_high = np.uint32(pow_dat[tind::interleave_num])
    pow_time_low = np.uint32(pow_dat[tind+1::interleave_num])
    if len(pow_time_low) != len(pow_time_high):
        pow_time_high = pow_time_high[:-1]

    pow_time = np.left_shift(pow_time_high.astype(np.uint64), np.uint64(32)) \
                  + pow_time_low.astype(np.uint64)

    power = pow_dat[tind+2::interleave_num]
    power_fb = pow_dat[tind+3::interleave_num]
    # power_fb = np.zeros_like(power)

    #plt.plot(np.int32(xyz_dat[tind+1::9]).astype(np.uint64) << np.uint64(32) \
    #         + np.int32(xyz_dat[tind::9]).astype(np.uint64) )
    #plt.show()

    # Since the FIFO read request is asynchronous, sometimes
    # the timestamp isn't first to come out, but the total amount of data
    # read out is a multiple of 3 (2 time + power) so the power
    # channel usually  ends up with less samples.
    # The following is coded very generally

    min_len = 10.0**9  # Assumes we never more than 1 billion samples
    if len(power) < min_len:
        min_len = len(power)
    if len(power_fb) < min_len:
        min_len = len(power_fb)

    # Re-size everything by the minimum length and convert to numpy array
    pow_time = np.array(pow_time[:min_len])
    power = np.array(power[:min_len])
    power_fb = np.array(power_fb[:min_len])

    return pow_time, power, power_fb










def get_fpga_data(fname, timestamp=0.0, verbose=False):
    '''Raw data from the FPGA is saved in an hdf5 (.h5) 
       file in the form of 3 continuous streams of I32s
       (32-bit integers). This script reads it out and 
       makes sense of it for post-processing'''

    # Open the file and bring datasets into memory
    try:
        f = h5py.File(fname,'r')
        dset1 = f['beads/data/quad_data']
        dset2 = f['beads/data/pos_data']
        dat1 = np.transpose(dset1)
        dat2 = np.transpose(dset2)
        if 'beads/data/pow_data' in f:
            dset3 = f['beads/data/pow_data']
            dat3 = np.transpose(dset3)
        else:
            dat3 = []
        f.close()

    # Shit failure mode. What kind of sloppy coding is this
    except (KeyError, IOError):
        if verbose:
            print("Warning, couldn't load HDF5 datasets: ", fname)
        dat1 = []
        dat2 = []
        dat3 = []
        attribs = {}
        try:
            f.close()
        except Exception:
            if verbose:
                print("couldn't close file, not sure if it's open")
            traceback.print_exc()

    if len(dat1):
        # Use subroutines to handle each type of data
        # raw_time, raw_dat = extract_raw(dat0, timestamp)
        quad_time, amp, phase = extract_quad(dat1, timestamp, verbose=verbose)
        xyz_time, xyz, xy_2, xyz_fb, sync = extract_xyz(dat2, timestamp, verbose=verbose)
        if len(dat3):
            pow_time, power, power_fb = extract_power(dat3, timestamp, verbose=verbose)
    else:
        quad_time, amp, phase = (None, None, None)
        xyz_time, xyz, xy_2, xyz_fb, sync = (None, None, None, None, None)

    # Assemble the output as a human readable dictionary
    out = {'xyz_time': xyz_time, 'xyz': xyz, 'xy_2': xy_2, \
           'fb': xyz_fb, 'quad_time': quad_time, 'amp': amp, \
           'phase': phase, 'sync': sync}
    if len(dat3):
        out['pow_time'] = pow_time
        out['power'] = power
        out['power_fb'] = power_fb
    else:
        out['pow_time'] = np.zeros_like(xyz_time)
        out['power'] = np.zeros_like(xyz[0])
        out['power_fb'] = np.zeros_like(xyz[0])

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

    out['pow_time'] = fpga_dat['pow_time'][off_ind:off_ind+nsamp]
    out['power'] = fpga_dat['power'][off_ind:off_ind+nsamp]
    out['power_fb'] = fpga_dat['power_fb'][off_ind:off_ind+nsamp]

    # return data in the same format as it was given
    return out