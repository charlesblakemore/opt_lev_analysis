## measure the force from the cantilever
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import scipy.signal as sig
import matplotlib.mlab as mlab

data_dir = "/data/20160320/bead1/cant_sweep_150um_dcsweep13"

using_dc_supply = False

skip_dat = 500

def sort_fun_old( s ):
    if( using_dc_supply ):
        return float(re.findall("dcps-?\d+mVdc", s)[0][4:-4])
    else:
        return float(re.findall("-?\d+mVdc", s)[0][:-4])

def sort_fun( s ):
    return float( re.findall("elec0_-?\d+mV",s)[0][6:-2] )

flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)
## now get the number of unique dc offsets
dclist = []
for f in flist:
    dclist.append(sort_fun( f ))

dclist = np.unique(dclist)
print "AC offsets: ", dclist

elec_list = [0,]
dcol_list = [1,]
tot_dat = np.zeros( [len(flist), 5])
tot_dat_pts = []
for fidx,f in enumerate(dclist):

    cpos = f ##sort_fun(f)

    print "Vdc = ", cpos

    cflist = glob.glob(os.path.join(data_dir, "*elec0_%dmV*.h5"%cpos))

    tot_corr_dr = []
    tot_corr_dr2 = []

    for f2 in cflist:
        cdat, attribs, fhand = bu.getdata( f2 )
        if( len(cdat) == 0 ):
            print "Empty file, skipping: ", f2
            continue

        Fs = attribs['Fsamp']
        drive_freq = attribs['electrode_settings'][16]
        fhand.close()

        NFFT = bu.prev_pow_2( len( cdat[:,1] ) )
        cpsd, freqs = mlab.psd(cdat[:, 1]-np.mean(cdat[:,1]), Fs = Fs, NFFT = NFFT) 

        # plt.figure()
        # plt.loglog(freqs, cpsd)
        # plt.show()

        #freq_idx = np.argmin( np.abs(freqs - drive_freq) )
        #freq_idx2 = np.argmin( np.abs(freqs - 2.0*drive_freq) )

        ## take correlation with drive and drive^2
        elec = elec_list[0]    
        dcol = dcol_list[0]

        response = cdat[skip_dat:-skip_dat, dcol]
        response -= np.mean(response)
        drive = cdat[skip_dat:-skip_dat, 8]
        drive -= np.mean(drive)
        drive2 = drive**2
        drive2 -= np.mean(drive2)

        if(False):
            plt.figure()
            plt.plot(drive)
            #plt.plot(response)
            b,a = sig.butter(3, (2.0*drive_freq+5)/(Fs/2.0) )
            filt_resp = sig.filtfilt(b,a,response)
            plt.plot(filt_resp,'c')
            plt.plot(drive2)
            plt.show()

        dummy_freq = 41 ## freq doesn't matter since we do 0 offset only
        corr_dr = bu.corr_func(drive, response, Fs, dummy_freq)[0]
        corr_dr2 = bu.corr_func(drive2, response, Fs, dummy_freq)[0]

        tot_corr_dr.append(corr_dr)
        tot_corr_dr2.append(corr_dr2)

    tot_dat[fidx, 0] = cpos
    tot_dat[fidx, 1] = np.mean(tot_corr_dr)
    tot_dat[fidx, 2] = np.mean(tot_corr_dr2)
    tot_dat[fidx, 3] = np.std(tot_corr_dr)/np.sqrt( len(tot_corr_dr) )
    tot_dat[fidx, 4] = np.std(tot_corr_dr2)/np.sqrt( len(tot_corr_dr2) )
    tot_dat_pts.append([cpos, tot_corr_dr, tot_corr_dr2])

    if( False and elec == 0 ):
        plt.figure()
        plt.plot( drive )
        plt.plot( response )
        plt.show()

tot_dat = np.array(tot_dat)

frange = [-200, 8000] ## fit range


plt.figure()
for t in tot_dat_pts:
    plt.plot(t[0]*np.ones_like(t[1]), t[1], 'k.', markersize=1)
    plt.plot(t[0]*np.ones_like(t[2]), t[2], 'r.', markersize=1)
plt.errorbar(tot_dat[:,0],tot_dat[:,1], tot_dat[:,3], fmt='ks-', label="f0")
plt.errorbar(tot_dat[:,0],tot_dat[:,2], tot_dat[:,4], fmt='ro-', label="2*f0")
plt.legend(loc=0, numpoints=1)

def ffn(x, p0, p1):
    return p0*x + p1

p = np.polyfit(tot_dat[:,0], tot_dat[:,1], 1)
bp, bcov = sp.curve_fit( ffn, tot_dat[:,0], tot_dat[:,1], p0=p, sigma = tot_dat[:,3] )

xx = np.linspace(np.min(tot_dat),np.max(tot_dat),1e2)
#plt.plot(xx, ffn(xx,bp[0],bp[1]), 'k', linewidth=1.5 )
print "Offset voltage: ", -bp[1]/bp[0]


plt.title("Correlation")

# plt.figure()
# plt.plot(tot_dat[:,0],tot_dat[:,3], 'ks-', label="f0")
# plt.plot(tot_dat[:,0],tot_dat[:,4], 'ro-', label="2*f0")
# plt.legend(loc=0, numpoints=1)
# plt.title("PSD")

plt.show()


