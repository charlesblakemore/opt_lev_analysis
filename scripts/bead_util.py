import h5py, os, matplotlib, re, glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.signal as sp
import scipy.interpolate as interp
import matplotlib.cm as cmx
import matplotlib.colors as colors


#######################################################
# This module has basic utility functions for analyzing bead
# data. In particular, this module has the basic data
# loading function, bead/physical constants and bead
# spectra.
#
# This version has been significantly trimmed from previous
# bead_util in an attempt to force modularization.
# Previous code for millicharge and chameleon data
# can be found by reverting opt_lev_analysis
#######################################################


bead_radius = 2.43e-6 ##m
bead_rho = 2.2e3 ## kg/m^3
kb = 1.3806488e-23 #J/K
bead_mass = 4./3*np.pi*bead_radius**3 * bead_rho
plate_sep = 1e-3 ## m
e_charge = 1.6e-19 ## C

nucleon_mass = 1.67e-27 ## kg
num_nucleons = bead_mass/nucleon_mass

## default columns for data files
data_columns = [0, 1, 2] ## column to calculate the correlation against
drive_column = -1


## work around inability to pickle lambda functions
class ColFFT(object):
    def __init__(self, vid):
        self.vid = vid
    def __call__(self, idx):
        return np.fft.rfft( self.vid[idx[0], idx[1], :] )

    

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




def round_sig(x, sig=2):
    '''Round a number to a certain number of sig figs
           INPUTS: x, number to be rounded
                   sig, number of sig figs

           OUTPUTS: num, rounded number'''
    if x == 0:
        return 0
    else:
        return round(x, sig-int(math.floor(math.log10(x)))-1)

def trend_fun(x, a, b):
    '''Two paramater linear function to de-trend datasets
           INPUTS: x, variable
                   a, param 1 = slope
                   b, param 2 = offset

           OUTPUTS: a*x + b'''
    return a*x + b



def step_fun(x, q, x0):
    '''Single, decreasing step function
           INPUTS: x, variable
                   q, size of step
                   x0, location of step

           OUTPUTS: q * (x <= x0)'''
    xs = np.array(x)
    return q*(xs<=x0)

def multi_step_fun(x, qs, x0s):
    '''Sum of many single, decreasing step functions
           INPUTS: x, variable
                   qs, sizes of steps
                   x0s, locations of steps

           OUTPUTS: SUM_i [qi * (x <= x0i)]'''
    rfun = 0.
    for i, x0 in enumerate(x0s):
        rfun += step_fun(x, qs[i], x0)
    return rfun



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

           OUTPUTS: Lorentzian amplitude'''
    w = 2. * np.pi * f
    w0 = 2. * np.pi * f0
    denom = np.sqrt((w0**2 - w**2)**2 + w**2 * g**2)
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
    return A * np.arctan2(-w * g, w0**2 - w**2) + phase0



def sum_3osc_amp(f, A1, f1, g1, A2, f2, g2, A3, f3, g3):
    '''Fitting function for AMPLITUDE of a sum of 3 damped harmonic oscillators.
           INPUTS: f [Hz], frequency 
                   A1,2,3, amplitude of the three oscillators
                   f1,2,3 [Hz], resonant frequency of the three oscs
                   g1,2,3 [Hz], damping factors

           OUTPUTS: Lorentzian amplitude of complex sum'''
    csum = damped_osc_amp(f, A1, f1, g1)*np.exp(1.j * damped_osc_phase(f, A1, f1, g1) ) \
           + damped_osc_amp(f, A2, f2, g2)*np.exp(1.j * damped_osc_phase(f, A2, f2, g2) ) \
           + damped_osc_amp(f, A3, f3, g3)*np.exp(1.j * damped_osc_phase(f, A3, f3, g3) )
    return np.abs(csum)

def sum_3osc_phase(f, A1, f1, g1, A2, f2, g2, A3, f3, g3, phase0=0.):
    '''Fitting function for PHASE of a sum of 3 damped harmonic oscillators.
       Includes an arbitrary DC phase to fit over out of phase responses
           INPUTS: f [Hz], frequency 
                   A1,2,3, amplitude of the three oscillators
                   f1,2,3 [Hz], resonant frequency of the three oscs
                   g1,2,3 [Hz], damping factors

           OUTPUTS: Lorentzian amplitude of complex sum'''
    csum = damped_osc_amp(f, A1, f1, g1)*np.exp(1.j * damped_osc_phase(f, A1, f1, g1) ) \
           + damped_osc_amp(f, A2, f2, g2)*np.exp(1.j * damped_osc_phase(f, A2, f2, g2) ) \
           + damped_osc_amp(f, A3, f3, g3)*np.exp(1.j * damped_osc_phase(f, A3, f3, g3) )
    return np.angle(csum) + phase0


    
def unwrap_phase(cycles):
    #Converts phase in cycles from ranging from 0 to 1 to ranging from -0.5 to 0.5 
    if cycles>0.5:
        cycles +=-1
    return cycles

def gauss_fun(x, A, mu, sig):
    return A*np.exp( -(x-mu)**2/(2*sig**2) )


def rotate_data(x, y, ang):
    c, s = np.cos(ang), np.sin(ang)
    return c*x - s*y, s*x + c*y









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

    return c_yz, c_xz, c_xy


def orthogonalize(xdat,ydat,zdat,c_yz,c_xz,c_xy):
    ## apply orthogonalization parameters generated by calc_orthogonalize pars
    zorth = zdat - np.median(zdat)
    yorth = ydat - np.median(ydat) - c_yz*zorth
    xorth = xdat - np.median(xdat) - c_xz*zorth - c_xy*yorth
    return xorth, yorth, zorth


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
