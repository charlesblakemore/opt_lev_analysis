## measure the force from the cantilever, averaging over files
## this version takes the difference between the pulled back and
## close position.
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
import scipy.interpolate as interp

data_dir = "/data/20150921/Bead1/chameleons_35um_diff"
remake_files = True
cant_pos_at_5V = 40e-6 ## distance from bead at 5 V, m
cant_step_per_V = 8.e-6 

data_column = 1
drive_column = 19
npts = 250000

NFFT = 2**(npts.bit_length() - 1)

#conv_fac = 1.6e-15/0.11 * (1./0.1) # N to V, assume 10 
conv_fac = 2.5e-14 #* (1./0.2) # N to V 

def get_idx( s ):
    return int(re.findall("\d+.h5", s)[0][:-3])
def get_z_pos(s):
    return float( re.findall("Z\d+nm", s)[0][1:-2] )
def sort_fun( s):
    # sort first by the cantilever position, then sort by 
    # the voltage on the cantilever
    return get_idx(s)*1e6 + get_z_pos(s)

flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)

## get the shape of the chameleon force vs. distance from Maxime's calculation
cforce = np.loadtxt("/home/dcmoore/opt_lev/scripts/data/chameleon_force.txt", delimiter=",")
## fit a spline to the data
spl = interp.UnivariateSpline( cforce[::5,0], cforce[::5,1], s=0 )

# plt.figure()
# plt.plot(cforce[:,0]*1e6, cforce[:,1], 'k.')
# xx = np.linspace(10e-6, 100e-6, 1e3)
# plt.plot( xx*1e6, spl(xx), 'r')
# plt.show()


def make_sig_shape(drive_mon):

    ## first, convert drive mon to meters
    drive_pos = (drive_mon - 5.)*cant_step_per_V + cant_pos_at_5V

    force_vs_time = spl( drive_pos )
    
    return force_vs_time

#make_sig_shape(np.zeros(1e2))

if(remake_files):
    tot_dat = []
    tot_psd = []
    tot_psd_far, tot_psd_near = [], []
    npsd = 0
    ## sorting above ensures files are properly paired
    for file_idx in range( 0,len(flist)/2 ):

        idx1, idx2 = get_idx(flist[file_idx*2]), get_idx(flist[file_idx*2 + 1])

        if( idx1 != idx2 ): 
            print "Warning mismatched files for index %d" % idx1
            continue

        print idx1
        cdat_far, attribs_far, _ = bu.getdata( flist[file_idx*2] )
        cdat_near, attribs_near, _ = bu.getdata( flist[file_idx*2 + 1] )

        f = flist[file_idx*2 + 1]
        attribs = attribs_near 
        cpos = float(re.findall("Z\d+nm", f)[0][1:-2])
        ## get the drive frequency from the attributes
        drive_freq = attribs["stage_settings"][6]

        ##print "Signal freq is: ", drive_freq, " Hz"

        ctime = bu.labview_time_to_datetime(attribs['Time'])

        Fs = attribs['Fsamp']

        cdat_diff = cdat_near[:,data_column] - cdat_far[:,data_column]
        cpsd, freqs = mlab.psd(cdat_diff - np.mean(cdat_diff), Fs = Fs, NFFT = NFFT) 
        cpsd_far, freqs = mlab.psd(cdat_far[:,data_column] - np.mean(cdat_far[:,data_column]), Fs = Fs, NFFT = NFFT) 
        cpsd_near, freqs = mlab.psd(cdat_near[:,data_column] - np.mean(cdat_near[:,data_column]), Fs = Fs, NFFT = NFFT) 

        if( len(tot_psd) == 0 ):
            tot_psd = cpsd
            tot_psd_far = cpsd_far
            tot_psd_near = cpsd_near
        else:
            tot_psd += cpsd
            tot_psd_far += cpsd_far
            tot_psd_near += cpsd_near

        npsd += 1

        cdrive = make_sig_shape( cdat_near[:,drive_column] )

        ## calculate the optimal filter with the drive
        vt = np.fft.rfft( cdat_diff )
        st = np.fft.rfft( cdrive )
        J = np.ones_like( vt )

        of_amp = np.real( np.sum( np.conj(st) * vt / J)/np.sum(np.abs(st)**2/J) )

        freq_idx = np.argmin( np.abs( freqs - drive_freq ) )

        bw = np.sqrt(freqs[1]-freqs[0]) ## bandwidth

        if(False):
            ## plot the individual spectra
            plt.figure()
            plt.semilogy( freqs, cpsd_far, 'r' )
            plt.semilogy( freqs, cpsd_near, 'b' )
            plt.semilogy( freqs, cpsd, 'k' )
            plt.semilogy( freqs[freq_idx], cpsd[freq_idx], 'o', mfc='none', mec='g', label="drive" )
            plt.xlim([np.max([0,drive_freq - 5]), drive_freq + 5])
            plt.title(str(cpos))
            plt.show()

        tot_dat.append( [cpos, np.sqrt(cpsd[freq_idx]*bw), of_amp, ctime] )

    tot_dat = np.array(tot_dat)
    np.save("plots/chameleon_data_by_run.npy", tot_dat)
    tot_psd_dat = np.vstack([freqs,np.ndarray.flatten(tot_psd),np.ndarray.flatten(tot_psd_near),np.ndarray.flatten(tot_psd_far)])
    np.save("plots/chameleon_psd.npy", tot_psd_dat)
else:
    
    tot_dat = np.load("plots/chameleon_data_by_run.npy")
    tot_psd_dat = np.load("plots/chameleon_psd.npy")
    npsd = len(tot_dat[:,0])
    freqs = tot_psd_dat[0,:]
    tot_psd = tot_psd_dat[1,:]
    tot_psd_near = tot_psd_dat[2,:]
    tot_psd_far = tot_psd_dat[3,:]

print "Total traces used: ", npsd


## minimum resolvable force scales down like
## psd [N/rtHz] * 1/sqrt(n) integrations
cpsd = np.sqrt(tot_psd/npsd)
cpsd_near = np.sqrt(tot_psd_near/npsd)
cpsd_far = np.sqrt(tot_psd_far/npsd)

plt.figure()
plt.semilogy( freqs, cpsd_near*conv_fac, 'r', label='Near' )
plt.semilogy( freqs, cpsd_far*conv_fac, 'b', label='Far' )
plt.semilogy( freqs, cpsd*conv_fac, 'k', label='Diff' )
plt.semilogy( freqs[freq_idx], cpsd[freq_idx]*conv_fac, 'o', mfc='none', mec='g', label="drive" )
plt.xlim([0, 2*drive_freq+5])
#plt.gca().set_xscale('log')
#plt.title(str(cpos))

plt.legend(numpoints=1)

#plt.ylim([1e-19, 1e-17])

plt.ylabel("Force PSD [N/rtHz]")    
#plt.savefig("plots/drive_spec.pdf")



plt.show()


