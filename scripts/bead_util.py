## set of utility functions useful for analyzing bead data

import h5py, os, matplotlib, re, glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.signal as sp
import scipy.interpolate as interp
import matplotlib.cm as cmx
import matplotlib.colors as colors

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
aod_columns = [4, 5, 6]

prime_comb = np.loadtxt("/home/dcmoore/opt_lev/waveforms/rand_wf_primes.txt")
## normalize the prime_comb to have max = 1
prime_comb /= np.max( np.abs(prime_comb) )

prime_freqs = [23,29,31,37,41,
               43,47,53,59,61,67,71, 
               73,79,83,89,97,101,103,107,109,113, 
               127,131,137,139,149,151,157,163,167,173, 
               179,181,191,193,197,199]



chamfil = h5py.File('/home/charles/opt_lev_analysis/scripts/chamsdata/2D_chameleon_force.h5', 'r')
## these don't work if the data is not in ascending order
cham_xforce = interp.RectBivariateSpline(chamfil['xcoord'],\
                                        chamfil['ycoord'], chamfil['xforce'])
cham_yforce = interp.RectBivariateSpline(chamfil['xcoord'],\
                                        chamfil['ycoord'], chamfil['yforce'])
#cham_xforce = interp.interp2d(chamfil['xcoord'],\
#                                        chamfil['ycoord'], chamfil['xforce'])
#cham_yforce = interp.interp2d(chamfil['xcoord'],\
#                                        chamfil['ycoord'], chamfil['yforce'])
cham_dat = np.loadtxt("/home/dcmoore/opt_lev/scripts/cant_force/cham_vs_x.txt", skiprows=9)
cham_dat[:,0] = 0.0015 - cham_dat[:,0] ## distance from cant face in m
cham_dat[:,0] = cham_dat[::-1,0]
cham_dat[:,1] = cham_dat[::-1,1]
sfac = (cham_xforce(1e-5,0)/np.interp(1e-5,cham_dat[:,0],cham_dat[:,1]))
cham_dat[:,1] = cham_dat[:,1]*sfac
cham_spl = interp.UnivariateSpline( cham_dat[:,0], cham_dat[:,1], s=1e-46)

cham_dat = np.loadtxt("/home/dcmoore/opt_lev/scripts/cant_force/cham_vs_x.txt", skiprows=9)
cham_dat[:,0] = 0.0015 - cham_dat[:,0] ## distance from cant face in m
cham_dat[:,0] = cham_dat[::-1,0]
cham_dat[:,1] = cham_dat[::-1,1]
sfac = (cham_xforce(1e-5,0)/np.interp(1e-5,cham_dat[:,0],cham_dat[:,1]))
cham_dat[:,1] = cham_dat[:,1]*sfac
cham_spl = interp.UnivariateSpline( cham_dat[:,0], cham_dat[:,1], s=1e-46)

es_dat = np.loadtxt("/home/dcmoore/comsol/dipole_force.txt", skiprows=9)
gpts = es_dat[:,0] > 15e-6
es_spl_log = interp.UnivariateSpline( es_dat[gpts,0], np.log(np.abs(es_dat[gpts,1])), s=2.5)
def es_spl(x):
    return np.exp(es_spl_log(x))

es_dat_fixed = np.loadtxt("/home/dcmoore/comsol/fixed_dipole_force.txt", skiprows=9)
gpts = es_dat_fixed[:,0] > 15e-6
es_spl_log_fixed = interp.UnivariateSpline( es_dat_fixed[gpts,0], np.log(np.abs(es_dat_fixed[gpts,1])), s=2.5)
def es_spl_fixed(x):
    return np.exp(es_spl_log_fixed(x))

# plt.figure()
# xx = np.linspace(5e-6,1e-3,1e3)
es_dat = np.loadtxt("/home/dcmoore/comsol/dipole_force.txt", skiprows=9)
gpts = es_dat[:,0] > 15e-6
es_spl_log = interp.UnivariateSpline( es_dat[gpts,0], np.log(np.abs(es_dat[gpts,1])), s=2.5)
def es_spl(x):
    return np.exp(es_spl_log(x))

es_dat_fixed = np.loadtxt("/home/dcmoore/comsol/fixed_dipole_force.txt", skiprows=9)
gpts = es_dat_fixed[:,0] > 15e-6
es_spl_log_fixed = interp.UnivariateSpline( es_dat_fixed[gpts,0], np.log(np.abs(es_dat_fixed[gpts,1])), s=2.5)
def es_spl_fixed(x):
    return np.exp(es_spl_log_fixed(x))

# plt.figure()
# xx = np.linspace(5e-6,1e-3,1e3)



## get the shape of the chameleon force vs. distance from Maxime's calculation
#cforce = np.loadtxt("/home/dcmoore/opt_lev/scripts/data/chameleon_force.txt", delimiter=",")
## fit a spline to the data
#cham_spl = interp.UnivariateSpline( cforce[::5,0], cforce[::5,1], s=0 )

## cv2 propId enumeration:
CV_CAP_PROP_POS_FRAMES = 1
CV_CAP_PROP_FPS = 5
CV_CAP_PROP_FRAME_COUNT = 7

## work around inability to pickle lambda functions
class ColFFT(object):
    def __init__(self, vid):
        self.vid = vid
    def __call__(self, idx):
        return np.fft.rfft( self.vid[idx[0], idx[1], :] )

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
            dat = dat / 3276.7 ## hard coded scaling from DAQ
            attribs = dset.attrs

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

    #print attribs["temps"][0]+273
    #norm_rat = (2*kb*(attribs["temps"][0]+273)/(bead_mass)) * 1/bp[0]
    norm_rat = (2*kb*293)/(bead_mass) * 1/bp[0]

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
        return int(re.findall('\d+', fname)[-1])
        
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
        correct_fac = 2.0*n_corr/(n_corr-i) # x2 from empirical tests
        #correct_fac = 1.0*n_corr/(n_corr-i) # 
        corr[i] = np.sum(drive*response[i:i+n_corr])*correct_fac
    return corr

def corr_func(drive, response, fsamp, fdrive, good_pts = [], filt = False, band_width = 1):
    #gives the correlation over a cycle of drive between drive and response.

    #First subtract of mean of signals to avoid correlating dc
    drive = drive-np.mean(drive)
    response  = response-np.mean(response)

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


    #corr_full = good_corr(drive, response, fsamp, fdrive)/(lentrace*drive_amp**2)
    corr_full = good_corr(drive, response, fsamp, fdrive)/(lentrace*drive_amp)
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
            if( "recharge" in f ): continue
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

        cf.close()

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


    return st_fit, st_dat, orth_pars


def get_avg_noise( calib_list, fnums, orth_pars, make_plot=False, norm_by_sum=False ):

    print "Making noise spectrum..."

    tot_vec_x = []
    tot_vec_y = []
    tot_vec_z = []
    dirs = ["x","y","z"]
    cols = data_columns+[drive_column,]

    ntraces = 0.
    fsamp = None
    for f in calib_list:
        fnum = re.findall("\d+.h5",f)
        if( len(fnum)>0 ):
            fnum = int(fnum[0][:-3])
            if( fnum < fnums ): continue
        print "Noise trace: ", f
        dat, attribs, cf = getdata(f)
        if( len(dat) == 0 ): continue

        if( not fsamp ): fsamp = attribs["Fsamp"]

        x_orth, y_orth, z_orth = orthogonalize(dat[:,cols[0]],dat[:,cols[1]],dat[:,cols[2]],
                                               orth_pars[0], orth_pars[1], orth_pars[2] )

        if( norm_by_sum ):
            norm_fac =  np.median( dat[:,7] )
        else: 
            norm_fac = 1.0

        # if( len(tot_vec_x) == 0 ):
        #     tot_vec_x = np.abs( np.fft.rfft( x_orth ) )**2 / norm_fac
        #     tot_vec_y = np.abs( np.fft.rfft( y_orth ) )**2 / norm_fac
        #     tot_vec_z = np.abs( np.fft.rfft( z_orth ) )**2 / norm_fac
        # else:
        #     tot_vec_x += np.abs( np.fft.rfft( x_orth ) )**2 / norm_fac
        #     tot_vec_y += np.abs( np.fft.rfft( y_orth ) )**2 / norm_fac
        #     tot_vec_z += np.abs( np.fft.rfft( z_orth ) )**2 / norm_fac

        if(len(calib_list) < 10 ):
            nfft = 2**13
        else:
            nfft = 2**16

        xpsd, freqs = matplotlib.mlab.psd(x_orth, Fs = fsamp, NFFT = nfft) 
        ypsd, freqs = matplotlib.mlab.psd(y_orth, Fs = fsamp, NFFT = nfft) 
        zpsd, freqs = matplotlib.mlab.psd(z_orth, Fs = fsamp, NFFT = nfft) 
        if( len(tot_vec_x) == 0 ):
            tot_vec_x = xpsd
            tot_vec_y = ypsd
            tot_vec_z = zpsd
        else:
            tot_vec_x += xpsd
            tot_vec_y += ypsd
            tot_vec_z += zpsd
        
        ntraces += 1.

    J = tot_vec_x/ntraces
    J_y = tot_vec_y/ntraces
    J_z = tot_vec_z/ntraces

    Jfreqs = freqs ##np.fft.rfftfreq( len(x_orth), 1/fsamp )

    if( make_plot ):
        fig = plt.figure()
        plt.loglog( Jfreqs, J )
        plt.loglog( Jfreqs, J_y )
        plt.loglog( Jfreqs, J_z )
        plt.show()

    return J, J_y, J_z, Jfreqs
        
def iterstat( data, nsig=2. ):

    good_data = np.ones_like(data) > 0
    last_std = np.std(data)
    cmu, cstd = np.median(data[good_data]), np.std(data[good_data])

    while( np.sum(good_data) > 5 ):
        
        cmu, cstd = np.median(data[good_data]), np.std(data[good_data])
        if( np.abs( (cstd - last_std)/cstd ) < 0.05 ):
            break

        good_data = inrange( data, cmu-nsig*cstd, cmu+nsig*cstd )
        last_std = cstd

    return cmu, cstd

def get_drive_bins( fvec ):
    """ return the bin numbers corresponding to the drive frequencies
        for the input vector fvec """

    bin_list = []
    [bin_list.append( np.argmin( np.abs( fvec - p ) ) ) for p in prime_freqs]

    out_vec = np.zeros_like(fvec) > 1
    out_vec[bin_list] = True

    return out_vec

def get_noise_direction_ratio( noise_list, weight_func ):
    """ take a list of noise files and get the relative amplitude of the spectra
        in the X, Y, and Z directions.  This is used to calibrate the relative
        channel gains """

    Jx, Jy, Jz = get_avg_noise( noise_list, 0, [0,0,0], make_plot=False )
    
    ## now find the weighted average over the frequencies in the drive
    rat_y = np.sum( Jy/Jx * weight_func )/np.sum( weight_func )
    rat_z = np.sum( Jz/Jx * weight_func )/np.sum( weight_func )
    rat_out = [1., rat_y, rat_z]

    print rat_out

    return rat_out

def fsin(x, p0, p1, p2):
    return p0*np.sin(2.*np.pi*p1*x + p2)

def get_mod(dat, mod_column = 3):
    b, a = sp.butter(3, 0.1)
    cdrive = np.abs(dat[:, mod_column] - np.mean(dat[:, mod_column]))
    cdrive = sp.filtfilt(b, a, cdrive)
    cdrive -= np.mean(cdrive)
    xx = np.arange(len(cdrive))
    spars = [np.std(cdrive)*2, 6.1e-4, 0]
    bf, bc = opt.curve_fit( fsin, xx, cdrive, p0 = spars)
    return fsin(xx, bf[0], bf[1], bf[2])/np.abs(bf[0])

def lin(x, m, b):
    return m*x + b

def get_DC_force(path, ind, column = 0, fmod = 6.):
    files = glob.glob(path + "/*.h5")
    n = len(files)
    amps = np.zeros(n)
    DCs = np.zeros(n)
    for i in range(n):
        dat, attribs, cf = getdata(os.path.join(path, files[i]))
        fs = attribs['Fsamp']
        cf.close()
        amps[i] = corr_func(get_mod(dat), dat[:, column], fs, fmod)[0]
        lst = re.findall('-?\d+', files[i])
        DCs[i] = int(lst[ind])
    return amps, DCs

def calibrate_dc(path, charge, dist = 0.01, make_plt = False):
    amps, DCs = get_DC_force(path, -5)
    Fs = DCs*e_charge*charge/dist
    spars = [1e15, 0.]
    bf, bc = opt.curve_fit(lin, Fs, amps, p0 = spars)
    if make_plt:
        
        plt.plot(Fs, lin(Fs, bf[0], bf[1]), 'r', label = 'linear fit', linewidth = 5)
        plt.plot(Fs, amps, 'x', label = 'data', markersize = 10, linewidth = 5)
        plt.legend()
        plt.xlabel('Applied Force [N]')
        plt.ylabel('Measured response [V]')
        plt.show()
    return 1./bf[0]


def get_chameleon_force( xpoints_in ):
    #return np.interp( xpoints_in, cham_dat[:,0], cham_dat[:,1] )
    return cham_spl( xpoints_in )

def get_es_force( xpoints_in, volt=1.0, is_fixed = False ):
    ## set is_fixed to true to get the force for a permanent dipole (prop to
    ## grad E), otherwise gives force on induced dipole (prop to E.(gradE)
    if(is_fixed):
        return es_spl_fixed( xpoints_in )*np.abs(volt)
    else:
        return es_spl( xpoints_in )*volt**2

def get_chameleon_force_chas(xpoints_in, y=0, yforce=False):


    ## Chas's controlling nature forces us to sort the array 
    ## before we can interpolate in order to use his 
    ## fancy 2d function.  I don't want to sort before I pass
    ## to this function!

    sorted_idx = np.argsort(xpoints_in)
    xpoints = xpoints_in[sorted_idx]

    xcomponent_out = cham_xforce(xpoints, y)
    ycomponent_out = cham_yforce(xpoints, y)

    ## now unsort
    xcomponent = np.zeros_like(xcomponent_out)
    ycomponent = np.zeros_like(ycomponent_out)
    xcomponent[sorted_idx] = xcomponent_out
    ycomponent[sorted_idx] = ycomponent_out

    if yforce:
        return xcomponent, ycomponent
    else:
        return xcomponent

def get_color_map( n ):
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=n)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    outmap = []
    for i in range(n):
        outmap.append( scalarMap.to_rgba(i) )
    return outmap

def load_dir_file( f ):

    lines = [line.rstrip('\n') for line in open(f)]

    out_dict = {}
    for l in lines: 
        
        lparts = l.split(";")
        
        if( len( lparts ) < 4 ):
            continue

        idx = int(lparts[0])
        dirs = lparts[1].split(',')
        dirs_list = []
        for cdir in dirs:
            dirs_list.append(cdir.strip())

        out_dict[idx] = [dirs_list, lparts[2].strip(), int(lparts[3])]

    return out_dict

def data_list_to_dict( d ):
    out_dict = {"path": d[0], "label": d[1], "drive_idx": d[2]}
    return out_dict


def make_histo_vs_time(x,y,xlab="File number",ylab="beta",lab="",col="k",axs=[],isbot=False):

    ## take x and y data and plot the time series as well as a Gaussian fit
    ## to the distro

    ## now do the inset plot
    #iax = plt.axes([0.1,0.1,0.5,0.8])
    plt.sca(axs[0])
    fmtstr = 'o-' if( len(y) < 200 ) else '.'
    ms = 4 if( len(y) < 200 ) else 3
    plt.plot( x, y, fmtstr, color=col, mec="none", markersize=ms )
    if(isbot):
        plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    yy=plt.ylim()

    plt.sca(axs[1])
    #iax2=plt.gca()

    crange = [np.percentile(y,5.)-np.std(y), np.percentile(y,95.)+np.std(y)]
    hh, be = np.histogram( y, bins = 30, range=crange )
    bc = be[:-1]+np.diff(be)/2.0
    cmu, cstd = np.median(y), np.std(y)
    amp0 = np.sum(hh)/np.sqrt(2*np.pi*cstd)
    spars=[amp0, cmu, cstd]
    try:
        bp, bcov = opt.curve_fit( gauss_fun, bc, hh, p0=spars )
    except RuntimeError:
        bp = spars
        bcov = np.eye(len(bp))
        bcov[1,1] = cstd/np.sqrt( len(y) )

    if( not np.shape(bcov) ):
        bcov = np.eye(len(bp))
        bcov[1,1] = cstd/np.sqrt( len(y) )        

    xx = np.linspace(crange[0], crange[1], 1e3)

    plt.errorbar( hh, bc, xerr=np.sqrt(hh), yerr=0, fmt='.', color=col, linewidth=1.5 )
   
    if(isbot):
        plt.xlabel("Counts")
        plt.legend(loc=0,prop={"size": 10})


    plt.ylim(yy)

    #plt.subplots_adjust(top=0.96, right=0.99, bottom=0.15, left=0.075)


def simple_sort( s ):
    ## simple sort function that sorts by last number in the file name
    ss = re.findall("\d+.h5", s)
    if( not ss ):
        return 0.
    else:
        return float(ss[0][:-3])

def prev_pow_2( x ):
    return 2**(x-1).bit_length() - 1
