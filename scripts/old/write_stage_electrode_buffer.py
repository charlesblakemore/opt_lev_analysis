## write a buffer that can be played back by the DAQ
## for controlling the stage and electrode potential at the same time

import glob, re, os
import numpy as np
import bead_util as bu
import scipy.optimize as opt
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

############################################

stage_channel_to_sweep = 2 ## ch2 is Z
stage_amp = 1. ## V
stage_freq = 10. ## Hz
stage_dc = 5. ## V

stage_position_at_5V = 60. ## um
elec_channel_to_sweep = 0 ## ch0 is cantilever

## path to files where we measure the force as a
## function of cantilever position and bias voltage
path_to_force_vs_elec_pot = "/data/20150921/Bead1/elec_force_vs_cant_pos3"
data_column = 1 ## column to calculate the response force against

sample_rate = 5000. ## Hz
number_of_samples = 250000 

cal_volts_to_N = 3.2e-14

use_fft_amplitude = True ## take the amplitude of the response
                         ## from the FFT.  Set to False to use
                         ## the amplitude from a time domain fit
plot_cal_files = True
plot_fits = True

##############################################

stage_um_per_volt = 8. ## um

def get_ac_volts(s):
    elec_str = "elec%d" % elec_channel_to_sweep
    return float(re.findall(elec_str+"_-?\d+mV", s)[0][6:-2])

def get_z_pos(s):
    return float( re.findall("Z\d+nm", s)[0][1:-2] )

def sort_fun( s):
    # sort first by the cantilever position, then sort by 
    # the voltage on the cantilever
    return get_z_pos(s)*1e6 + get_ac_volts(s)

def get_stage_elec_lists( flist ):
    stage_pos_list = []
    elec_volt_list = []
    for f in flist:
        elec_volt_list.append( get_ac_volts(f) )
        stage_pos_list.append( get_z_pos(f) )
    return np.unique(stage_pos_list), np.unique(elec_volt_list)

def stage_V_to_m(pos_volt):
    ## convert a stage position in V to m
    pos_um = (pos_volt - 5.)*stage_um_per_volt  + stage_position_at_5V
    return pos_um * 1e-6

## first make the stage array
tvec = np.arange( number_of_samples ) / sample_rate ## sec 
stage_arr = np.sin(2.*np.pi*stage_freq*tvec)

## convert the stage_arr to meters
stage_arr_m = stage_V_to_m(stage_arr)

## get the chameleon force that we want to mimic at each position
cham_force = bu.get_chameleon_force( stage_arr_m )

## now determine the voltage needed to reproduce this chameleon force
cal_list = sorted(glob.glob( os.path.join(path_to_force_vs_elec_pot, "*.h5") ), key=sort_fun)

## get the number of stage positions
stage_pos_list, elec_volt_list = get_stage_elec_lists( cal_list )

## if desired, make a plot of the power spectrum vs pos at the max volt
if( plot_cal_files ):
    max_volt = np.max( elec_volt_list )
    col_list = bu.get_color_map( len( stage_pos_list ) )
    plt.figure()
    color_idx = 0
    for f in cal_list:
        cac = get_ac_volts( f )
        czp = get_z_pos( f )
        if( cac != max_volt ): continue
        cdat, attribs, _ = bu.getdata( f )    

        NFFT = 2**(len( cdat[:,1] ).bit_length() - 1)
        cpsd, freqs = mlab.psd(cdat[:, 1]-np.mean(cdat[:,1]), Fs = attribs['Fsamp'], NFFT = NFFT) 
        plt.loglog( freqs, np.sqrt(cpsd)*cal_volts_to_N, label = czp, color=col_list.to_rgba(color_idx) )
        color_idx += 1

    plt.legend()
    plt.title("Response vs. stage position, %d mV drive"%max_volt)
    plt.xlabel("Freq [Hz]")
    plt.ylabel("Y PSD [N/rtHz]")

    max_pos = np.max( stage_pos_list )
    col_list = bu.get_color_map( len( elec_volt_list ) )
    plt.figure()
    color_idx = 0
    for f in cal_list:
        cac = get_ac_volts( f )
        czp = get_z_pos( f )
        if( czp != max_pos ): continue
        cdat, attribs, _ = bu.getdata( f )    

        NFFT = 2**(len( cdat[:,1] ).bit_length() - 1)
        cpsd, freqs = mlab.psd(cdat[:, 1]-np.mean(cdat[:,1]), Fs = attribs['Fsamp'], NFFT = NFFT) 
        plt.loglog( freqs, np.sqrt(cpsd)*cal_volts_to_N, label = cac, color=col_list.to_rgba(color_idx) )
        color_idx += 1

    plt.legend()
    plt.title("Response vs. voltage, stage at %d mV drive"%max_pos)
    plt.xlabel("Freq [Hz]")
    plt.ylabel("Y PSD [N/rtHz]")    

    plt.show()

xx,yy = np.meshgrid( stage_pos_list, elec_volt_list )
zz = np.zeros_like(xx)

def min_fun( a, d, r ):
    return np.sum( (a*d - r)**2 )

for f in cal_list:
    
    cdat, attribs, _ = bu.getdata( f )    
    
    Fs = attribs['Fsamp']
    vac = attribs['electrode_settings'][electrode_channel_to_sweep]
    drive_freq = attribs['electrode_settings'][8+electrode_channel_to_sweep]
    stage_pos = attribs['stage_settings'][stage_channel_to_sweep]

    stage_idx = np.argwhere( stage_pos_list == stage_pos )[0]
    elec_idx = np.argwhere( elec_volt_list == vac )[0]

    ## find response amplitude of Y to the drive signal
    drive_pos = electrode_channel_to_sweep+8 if electrode_channel_to_sweep <= 2 else electrode_channel_to_sweep+9
    drive_sig = cdat[:,drive_pos]
    ## normalize to unit amplitude, assuming that it's a sin
    drive_sig /= (np.sqrt(2)*np.std(drive_sig))
    resp_sig = cdat[:,data_column]

    drive_sig -= np.mean(drive_sig)
    resp_sig -= np.mean(resp_sig)

    if( use_fft_amp ):

        psd = np.abs(np.fft.rfft( resp_sig ))**2
        freqs = np.fft.rfftfreq( len(resp_sig), 1./Fs )
        
        ## number of bins around max bin to integrate over
        nbins = 2
        
        ## find the bin for the second harmonic
        bidx = np.argwhere( np.abs( freqs - (drive_freq*2) ) )        
        bin_width = freqs[1]-freqs[0]
        amp_val = np.sqrt( np.sum( psd[bidx-nbins:bidx+nbins+1]*bin_width ) ) 

    else:
        ## find the amplitude of the response in volts to the drive
        ## time domain fit for now, but this could also be done
        ## with the fft
        cmf = lambda a: min_fun(a, drive_sig, resp_sig)
        result = opt.minimize( cmf, np.std( resp_sig ) )

        if( plot_fits ):
            plt.figure()
            plt.plot( resp_sig, 'x.' )
            plt.plot( drive_sig * result.x, 'r' )
            plt.show()
    
        amp_val = result.x

    zz[stage_idx, elec_idx] = amp_val * cal_volts_to_N

## now we have the amplitude of the induced force vs. position
## and votage.  we need to interpolate the inverse to find the voltage
## needed at each position to give a specified force

volts_vs_pos = interp.RectBivariateSpline(xx, zz, yy)
volt_arr = volts_vs_pos(stage_arr_m, cham_force)

## save the positions and voltages as a text file
out_arr = np.vstack( (volts_vs_pos, volt_arr) )
