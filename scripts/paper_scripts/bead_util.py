## set of utility functions useful for analyzing bead data

import h5py, os, matplotlib, re
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.signal as sp

bead_radius = 2.53e-6 ##m
bead_rho = 2.0e3 ## kg/m^3
kb = 1.3806488e-23 #J/K
bead_mass = 4./3*np.pi*bead_radius**3 * bead_rho
plate_sep = 1e-3 ## m
e_charge = 1.6e-19 ## C

nucleon_mass = 1.67e-27 ## kg
num_nucleons = bead_mass/nucleon_mass

## default columns for data files
data_columns = [0, 1, 2] ## column to calculate the correlation against
drive_column = -1
laser_column = 3

prime_comb = np.loadtxt("../../waveforms/rand_wf_primes.txt")
## normalize the prime_comb to have max = 1
prime_comb /= np.max( np.abs(prime_comb) )


def gain_fac( val ):
    ### Return the gain factor corresponding to a given voltage divider
    ### setting.  These numbers are from the calibration of the voltage
    ### divider on 2014/06/20 (in lab notebook)
    volt_div_vals = {0.:  1.,
                     1.:  1.,
                     20.0: 100./5.07,
                     40.0: 100./2.67,
                     80.0: 100./1.38,
                     200.0: 100./0.464}
    if val in volt_div_vals:
        return volt_div_vals[val]
    else:
        print "Warning, could not find volt_div value"
        return 1.
    

def getdata(fname, gain_error=1.0):
    ### Get bead data from a file.  Guesses whether it's a text file
    ### or a HDF5 file by the file extension

    _, fext = os.path.splitext( fname )
    if( fext == ".h5"):
        try:
            f = h5py.File(fname,'r')
            dset = f['beads/data/pos_data']
            dat = np.transpose(dset)
            max_volt = dset.attrs['max_volt']
            nbit = dset.attrs['nbit']
            dat = 1.0*dat*max_volt/nbit
            attribs = dset.attrs

            ## correct the drive amplitude for the voltage divider. 
            ## this assumes the drive is the last column in the dset
            vd = attribs['volt_div'] if 'volt_div' in attribs else 1.0
            if( vd > 0 ):
                curr_gain = gain_fac(vd*gain_error)
                dat[:,-1] *= curr_gain

        except (KeyError, IOError):
            print "Warning, got no keys for: ", fname
            dat = []
            attribs = {}
            f = []
    else:
        dat = np.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5])
        attribs = {}
        f = []

    return dat, attribs, f

def labview_time_to_datetime(lt):
    ### Convert a labview timestamp (i.e. time since 1904) to a 
    ### more useful format (python datetime object)
    
    ## first get number of seconds between Unix time and Labview's
    ## arbitrary starting time
    lab_time = dt.datetime(1904, 1, 1, 0, 0, 0)
    nix_time = dt.datetime(1970, 1, 1, 0, 0, 0)
    delta_seconds = (nix_time-lab_time).total_seconds()

    lab_dt = dt.datetime.fromtimestamp( lt - delta_seconds)
    
    return lab_dt
    

def inrange(x, xmin, xmax):
    return np.logical_and( x >= xmin, x<=xmax )

def bead_spec_rt_hz(f, A, f0, Damping):
    omega = 2*np.pi*f
    omega_0 = 2*np.pi*f0
    return np.sqrt( A*Damping/((omega_0**2 - omega**2)**2 + omega**2*Damping**2) )


def get_calibration(refname, fit_freqs, make_plot=False, 
                    data_columns = [0,1], drive_column=-1, NFFT=2**14, exclude_peaks=False, spars=[]):
    ## given a reference file, fit the spectrum to a Lorentzian and return
    ## the calibration from V to physical units
    dat, attribs, cf = getdata(refname)
    if( len(attribs) > 0 ):
        fsamp = attribs["Fsamp"]
    xdat = dat[:,data_columns[0]]
    xpsd, freqs = matplotlib.mlab.psd(xdat, Fs = fsamp, NFFT = NFFT) 
    xpsd = np.ndarray.flatten(xpsd)

    ##first, fit for the absolute calibration
    damp_guess = 400
    f0_guess = 150
    Aemp = np.median( xpsd[fit_freqs[0]:fit_freqs[0]+10] )
    if( len(spars) == 0):
        spars = [Aemp*(2*np.pi*f0_guess)**4/damp_guess, f0_guess, damp_guess]

    fit_bool = inrange( freqs, fit_freqs[0], fit_freqs[1] )

    ## if there's large peaks in the spectrum, it can cause the fit to fail
    ## this attempts to exclude them.  If a single boolean=True is passed,
    ## then any points 50% higher than the starting points are excluded (useful
    ## for th overdamped case). If a list defining frequency ranges is passed, e.g.:
    ## [[f1start, f1stop],[f2start, f2stop],...], then points within the given
    ## ranges are excluded
    if( isinstance(exclude_peaks, list) ):
        for cex in exclude_peaks:
            fit_bool = np.logical_and(fit_bool, np.logical_not( inrange(freqs, cex[0],cex[1])))
    elif(exclude_peaks):
        fit_bool = np.logical_and( fit_bool, xpsd < 1.5*Aemp )

    xdat_fit = freqs[fit_bool]
    ydat_fit = np.sqrt(xpsd[fit_bool])
    bp, bcov = opt.curve_fit( bead_spec_rt_hz, xdat_fit, ydat_fit, p0=spars, maxfev=10000, sigma=ydat_fit/100.)
    #bp = spars
    #bcov = 0.

    print bp

    print attribs["temps"][0]+273
    norm_rat = (2*kb*(attribs["temps"][0]+273)/(bead_mass)) * 1/bp[0]

    if(make_plot):
        fig = plt.figure()
        plt.loglog( freqs, np.sqrt(norm_rat * xpsd), '.' )
        plt.loglog( xdat_fit, np.sqrt(norm_rat * ydat_fit**2), 'k.' )
        xx = np.linspace( freqs[fit_bool][0], freqs[fit_bool][-1], 1e3)
        plt.loglog( xx, np.sqrt(norm_rat * bead_spec_rt_hz( xx, bp[0], bp[1], bp[2] )**2), 'r')
        plt.xlabel("Freq [Hz]")
        plt.ylabel("PSD [m Hz$^{-1/2}$]")
    
    return np.sqrt(norm_rat), bp, bcov


def find_str(str):
    """ Function to sort files.  Assumes that the filename ends
        in #mV_#Hz[_#].h5 and sorts by end index first, then
        by voltage """
    idx_offset = 1e10 ## large number to ensure sorting by index first

    fname, _ = os.path.splitext(str)

    endstr = re.findall("\d+mV_[\d+Hz_]*[a-zA-Z_]*[\d+]*", fname)
    if( len(endstr) != 1 ):
        ## couldn't find the expected pattern, just return the 
        ## second to last number in the string
        return int(re.findall('\d+', fname)[-2])
        
    ## now check to see if there's an index number
    sparts = endstr[0].split("_")
    if( len(sparts) >= 3 ):
        return idx_offset*int(sparts[2]) + int(sparts[0][:-2])
    else:
        return int(sparts[0][:-2])
    
def unwrap_phase(cycles):
    #Converts phase in cycles from ranging from 0 to 1 to ranging from -0.5 to 0.5 
    if cycles>0.5:
        cycles +=-1
    return cycles

def laser_reject(laser, low_freq, high_freq, thresh, N, Fs, plt_filt):
    #returns boolian vector of points where laser is quiet in band. Averages over N points.
    b, a = sp.butter(3, [2.*low_freq/Fs, 2.*high_freq/Fs], btype = 'bandpass')
    filt_laser_sq = np.convolve(np.ones(N)/N, sp.filtfilt(b, a, laser)**2, 'same')
    if plt_filt:
        plt.figure()
        plt.plot(filt_laser_sq)
        plt.plot(np.argwhere(filt_laser_sq>thresh),filt_laser_sq[filt_laser_sq>thresh],'r.')
        plt.show()
    return filt_laser_sq<=thresh


def good_corr(drive, response, fsamp, fdrive):
    corr = np.zeros(fsamp/fdrive)
    response = np.append(response, np.zeros( fsamp/fdrive-1 ))
    n_corr = len(drive)
    for i in range(len(corr)):
        #Correct for loss of points at end
        correct_fac = 1.0*n_corr/(n_corr-i)
        corr[i] = np.sum(drive*response[i:i+n_corr])*correct_fac
    return corr

def corr_func(drive, response, fsamp, fdrive, good_pts = [], filt = False, band_width = 1):
    #gives the correlation over a cycle of drive between drive and response.

    #First subtract of mean of signals to avoid correlating dc
    drive = drive-np.mean(drive)
    response  = response - np.mean(response)

    #bandpass filter around drive frequency if desired.
    if filt:
        b, a = sp.butter(3, [2.*(fdrive-band_width/2.)/fsamp, 2.*(fdrive+band_width/2.)/fsamp ], btype = 'bandpass')
        drive = sp.filtfilt(b, a, drive)
        response = sp.filtfilt(b, a, response)
    
    #Compute the number of points and drive amplitude to normalize correlation
    lentrace = len(drive)
    drive_amp = np.sqrt(2)*np.std(drive)

      
    #Throw out bad points if desired
    if len(good_pts):
        response[-good_pts] = 0.
        lentrace = np.sum(good_pts)    


    corr_full = good_corr(drive, response, fsamp, fdrive)/lentrace
    return corr_full

def corr_blocks(drive, response, fsamp, fdrive, good_pts = [], filt = False, band_width = 1, N_blocks = 20):
    #Computes correlation in blocks to determine error.

    #first determine average phase to use throughout.
    tot_phase =  np.argmax(corr_func(drive, response, fsamp, fdrive, good_pts, filt, band_width))
    
    #Now initialize arrays and loop over blocks
    corr_in_blocks = np.zeros(N_blocks)
    len_block = len(drive)/int(N_blocks)
    for i in range(N_blocks):
        corr_in_blocks[i] = corr_func(drive[i*len_block:(i+1)*len_block], response[i*len_block:(i+1)*len_block], fsamp, fdrive, good_pts, filt, band_width)[tot_phase]
    return [np.mean(corr_in_blocks), np.std(corr_in_blocks)/N_blocks]

def gauss_fun(x, A, mu, sig):
    return A*np.exp( -(x-mu)**2/(2*sig**2) )

def get_drive_amp(drive_data, fsamp, drive_freq="chirp", make_plot=False):

    ## first get rid of any offsets
    drive_data -= np.median(drive_data)

    ntile = len(drive_data)/fsamp
    samp_pts = np.linspace(0, ntile, ntile*fsamp)

    if( drive_freq == "chirp" ):
        ## get the amplitude from the correlation with the expected waveform

        tiled_chirp = np.tile( prime_comb, int(ntile) )
        tvec = np.linspace( 0, ntile, len(tiled_chirp) )
        chirp_sampled = np.interp( samp_pts, tvec, tiled_chirp )

    elif( isinstance(drive_freq, (int,long,float)) ):
        chirp_sampled = np.sin( 2.*np.pi*samp_pts*drive_freq )
        
    else:
        print "Warning: get_drive_amp requires numeric frequency or 'chirp'"
        return 0.
        
    ## chirp sampled lags the data by 1 samples, presumably due to the trigger,
    ## so account for that here
    chirp_sampled = np.roll( chirp_sampled, 1 )

    if(make_plot):
        plt.figure()
        plt.plot( drive_data )
        plt.plot( chirp_sampled*np.median( drive_data / chirp_sampled ) )
        plt.show()

    drive_amp = np.median( drive_data / chirp_sampled )
            
    return drive_amp, chirp_sampled*drive_amp

def rotate_data(x, y, ang):
    c, s = np.cos(ang), np.sin(ang)
    return c*x - s*y, s*x + c*y

def bead_spec_comp(f, A, f0, Damping, phi0):
    omega = 2*np.pi*f
    omega_0 = 2*np.pi*f0
    amp = np.sqrt(A*Damping/((omega_0**2 - omega**2)**2 + omega**2*Damping**2))
    phi = phi0 - np.arctan2( Damping*omega, (omega_0**2 - omega**2) )
    return amp*np.exp( 1j * phi )

def low_pass_tf( f, fc ):
    ### transfer function for low pass filter
    return 1./(1 + 1j*2*np.pi*f/fc)

def curve_fit_complex( fcomp, xvals, yvals, spars):

    ## make complex number into 1d array of reals
    ffn = lambda x: fcomp( xvals, x[0], x[1], x[2], x[3], x[4])
    err = lambda x: np.abs( ffn( x ) - yvals )
    
    bp = opt.leastsq( err, spars)
    return bp

def calc_orthogonalize_pars(d):
    ## take a dictionary containing x, y, z traces and return gram-schmidt
    ## coefficients that orthoganalize the set, i.e. the orthogonal components
    ## are: 
    ##       z_orth = z
    ##       y_orth = y - <y,z_orth>/<z_orth,z_orth>*z
    ##       x_orth = x - <x,z_orth>/<z_orth,z_orth>*z - <x,y_orth>/<y_orth,y_orth>*y_orth
    ## where the returned coefficients are:  
    ##       c_yz = <y,z_orth>/<z_orth,z_orth>
    ##       c_xz = <x,z_orth>/<z_orth,z_orth>
    ##       c_xy = <x,y_orth>/<y_orth,y_orth>

    z_orth = d['z'] - np.median( d['z'] )
    y0 = d['y']- np.median( d['y'])
    c_yz = np.sum( y0*z_orth )/np.sum( z_orth**2 )
    y_orth = y0 - c_yz*z_orth
    x0 = d['x']
    c_xz = np.sum( x0*z_orth )/np.sum( z_orth**2 )
    c_xy = np.sum( x0*y_orth )/np.sum( y_orth**2 )
    #x_orth = d['x'] - c_xz*z_orth - c_xy*y_orth

    return c_yz, c_xz, c_xy

def orthogonalize(xdat,ydat,zdat,c_yz,c_xz,c_xy):
    ## apply orthogonalization parameters generated by calc_orthogonalize pars
    zorth = zdat - np.median(zdat)
    yorth = ydat - np.median(ydat) - c_yz*zorth
    xorth = xdat - np.median(xdat) - c_xz*zorth - c_xy*yorth
    return xorth, yorth, zorth


def get_avg_trans_func( calib_list, fnums, make_plot=False, make_trans_func_plot=True ):
    ## Take a list of calibration files and return the averaged
    ## transfer function for the drive.
    ## For now this assumes the calibration was made with the drive
    ## plugged into the main computer.

    if( len(fnums) != 2 ):
        print "Single charge files not defined, doing simple correlation"
        tot_corr = []
        for f in calib_list:
            dat, attribs, cf = getdata(f)
            if( len(dat) == 0 ): continue

            fnum = int(re.findall("\d+.h5",f)[0][:-3])
            corr = np.sum( dat[:,data_columns[0]]*dat[:,drive_column])
            tot_corr.append( [fnum, corr] )

        tot_corr = np.array(tot_corr)
        fig = plt.figure()
        plt.plot( tot_corr[:,0], tot_corr[:,1], 'k.')
        plt.xlabel("File number")
        plt.ylabel("Correlation with drive")
        plt.show()

        out_nums = []
        while( len(out_nums) != 2 ):
            fvals = raw_input("Enter start file number, end file number: ")
            out_strs = fvals.split(",")
            if( len( out_strs) != 2 ): continue
            out_nums.append( int(out_strs[0]) )
            out_nums.append( int(out_strs[1]) )
        
        fnums = out_nums

            
    print "Making transfer function..."

    ## first make the average function
    tot_vec = {}

    dirs = ["x","y","z","drive"]
    cols = data_columns+[drive_column,]

    ntraces = 0.
    fsamp = None
    for f in calib_list:
        fnum = int(re.findall("\d+.h5",f)[0][:-3])
        if( fnum < fnums[0] or fnum > fnums[1] ): continue
        print "Calib trace: ", f
        dat, attribs, cf = getdata(f)
        if( len(dat) == 0 ): continue

        if( not fsamp ): fsamp = attribs["Fsamp"]

        for k,col in zip(dirs, cols):
        
            if k in tot_vec:
                tot_vec[k] += dat[:,col]
            else:
                tot_vec[k] = dat[:,col]
        ntraces += 1.

    for k in dirs:
        tot_vec[k] /= ntraces

    ## get exact drive function
    curr_drive_amp, tot_vec["drive"] = get_drive_amp(tot_vec["drive"], fsamp)

    ## now orthogonalize all coordinates
    orth_pars = calc_orthogonalize_pars( tot_vec )

    x_orth, y_orth, z_orth = orthogonalize( tot_vec['x'], tot_vec['y'], tot_vec['z'],
                                            orth_pars[0], orth_pars[1], orth_pars[2] )

    ## now calculate total transfer function
    dt = np.fft.rfft( tot_vec["drive"]/curr_drive_amp )
    Hobs = np.fft.rfft( x_orth ) / dt
    Hfreqs = np.fft.rfftfreq( len(x_orth), 1/fsamp )
    ## drive signal for 1 V max amplitude

    ## Hobs is only well-defined at the frequency points in the drive, so select
    ## only these points
    drive_freqs =  np.ndarray.flatten(np.argwhere(np.abs(dt) > 1e3 ))

    ## fit a smooth Lorentzian to the measured points
    ffn = lambda x,p0,p1,p2,p3,p4: bead_spec_comp(x,p0,p1,p2,p3)*low_pass_tf(x,p4)    
    spars = [5e5, 160, 500, 0., 300.]
    bp, bcov = curve_fit_complex( ffn, Hfreqs[drive_freqs], Hobs[drive_freqs], spars)

    ## get the best fit to the expected response
    st_fit = dt*ffn( Hfreqs, bp[0], bp[1], bp[2], bp[3], bp[4])
    st_dat = np.fft.rfft( x_orth/curr_drive_amp )

    ## time domain
    drive_pred_fit = np.fft.irfft( st_fit )
    drive_pred_dat = np.fft.irfft( st_dat )

    if( make_trans_func_plot ):
        fig_t0 = plt.figure()
        plt.loglog( Hfreqs[drive_freqs], np.abs(Hobs[drive_freqs]), 'ko', label="Data" )
        #plt.loglog( Hfreqs, np.abs(dt), label="Drive" )
        fth = np.linspace( Hfreqs[0], Hfreqs[-1], 1e3 )
        plt.loglog( fth, np.abs(ffn( fth, bp[0], bp[1], bp[2], bp[3], bp[4])), 'r', label="Fit")
        plt.xlabel("Freq. [Hz]")
        plt.ylabel("Transfer function amplitude")
        plt.legend()

        fig_t1 = plt.figure()
        plt.semilogx( Hfreqs[drive_freqs], np.angle(Hobs[drive_freqs]), 'ko' )
        plt.semilogx( fth, np.angle(ffn( fth, bp[0], bp[1], bp[2], bp[3], bp[4])), 'r')

        plt.show()

    if( make_plot ):
        fig = plt.figure()
        for k in dirs:
            plt.plot( tot_vec[k]/np.max(np.abs(tot_vec[k]) ), label=k )
        plt.legend()

        fig2 = plt.figure()
        plt.plot( tot_vec['x'] )
        plt.plot( x_orth )

        fig3 = plt.figure()
        plt.plot( drive_pred_fit )
        plt.plot( drive_pred_dat )

        plt.show()

    cf.close()

    return st_fit, st_dat, orth_pars


def get_avg_noise( calib_list, fnums, orth_pars, make_plot=False ):

    print "Making noise spectrum..."

    tot_vec = []
    dirs = ["x","y","z"]
    cols = data_columns+[drive_column,]

    ntraces = 0.
    fsamp = None
    for f in calib_list:
        fnum = int(re.findall("\d+.h5",f)[0][:-3])
        if( fnum < fnums ): continue
        print "Noise trace: ", f
        dat, attribs, cf = getdata(f)
        if( len(dat) == 0 ): continue

        if( not fsamp ): fsamp = attribs["Fsamp"]

        x_orth, y_orth, z_orth = orthogonalize(dat[:,cols[0]],dat[:,cols[1]],dat[:,cols[2]],
                                               orth_pars[0], orth_pars[1], orth_pars[2] )
        if( len(tot_vec) == 0 ):
            tot_vec = np.abs( np.fft.rfft( x_orth ) )**2
        else:
            tot_vec += np.abs( np.fft.rfft( x_orth ) )**2
        
        ntraces += 1.

    J = tot_vec/ntraces

    if( make_plot ):
        Jfreqs = np.fft.rfftfreq( len(x_orth), 1/fsamp )
        fig = plt.figure()
        plt.loglog( Jfreqs, J )
        plt.show()

    return J
        
