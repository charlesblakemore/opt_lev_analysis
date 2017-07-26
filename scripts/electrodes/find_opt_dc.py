## measure the force from the cantilever
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import scipy.signal as sig
import matplotlib.mlab as mlab

#data_dirs = ["/data/20160320/bead1/pump_down2/dc_zero_elec1_pos3","/data/20160320/bead1/pump_down2/dc_zero_elec1_neg3"]
data_dirs = ["/data/20160325/bead1/elec0_dczero_100um2",]
             #"/data/20160325/bead1/elec0_dczero_60um_neg",]
             #"/data/20160320/bead1/pump_down2/dc_zero_elec2_fine_cant0_0",
             #"/data/20160320/bead1/pump_down2/dc_zero_elec3_fine_cant0_0",
             #"/data/20160320/bead1/pump_down2/dc_zero_elec4_fine_cant0_0",]
             #"/data/20160320/bead1/pump_down2/dc_zero_elec1_fine_cant0_1",
             #"/data/20160320/bead1/pump_down2/dc_zero_elec1_fine_cant-0_4",
             #"/data/20160320/bead1/pump_down2/dc_zero_elec1_fine_cant0_5",
             #"/data/20160320/bead1/pump_down2/dc_zero_elec1_fine_cant1_0",]

leg_list = ['elec0', 'elec2','elec3','elec4','elec5','elec6']
elec_list = [0,2,3,4,5,6]
col = ['k','r','b','g','c','m','y']
cidx = 0

gain = 1./1000.
using_dc_supply = False

skip_dat = 500

def sort_fun( s ):
    if( using_dc_supply ):
        return float(re.findall("dcps-?\d+mVdc", s)[0][4:-4])
    else:
        dcval = float(re.findall("-?\d+mVdc", s)[0][:-4])
        if( 'neg' in s and dcval > 0):
            dcval *= -1
        return dcval

flist = []
for i,data_dir in enumerate(data_dirs):
    cflist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)
    flist += cflist
figd = plt.figure()
figd2 = plt.figure()


## now get the number of unique dc offsets
dclist = []
for f in flist:
    dclist.append(sort_fun( f ))

dclist = np.unique(dclist)
print "DC offsets: ", dclist

elec = elec_list[0]
dcol = 0
tot_dat = np.zeros( [len(dclist), 5])
tot_dat_pts = []
for fidx,f in enumerate(dclist):

    cpos = f ##sort_fun(f)

    print "Vdc = ", cpos

    cflist = []
    for data_dir in data_dirs:
        if( cpos < 0 and 'pos' in data_dir ): continue
        if( cpos > 0 and 'neg' in data_dir ): continue
        #ccflist = glob.glob(os.path.join(data_dir, "*Hz%dmVdc*.h5"%abs(cpos)))
        ccflist = glob.glob(os.path.join(data_dir, "*Hz%dmVdc*.h5"%cpos))
        if( cpos == 0 and len(cflist) == 0):
            ccflist = glob.glob(os.path.join(data_dir, "*Hz-%dmVdc*.h5"%abs(cpos)))
        cflist += ccflist

    tot_corr_dr = []
    tot_corr_dr2 = []
    tot_corr_dr_err = []

    for f2 in cflist:
        cdat, attribs, fhand = bu.getdata( f2 )
        if( len(cdat) == 0 ):
            print "Empty file, skipping: ", f2
            continue

        Fs = attribs['Fsamp']
        drive_freq = attribs['electrode_settings'][16+elec]
        fhand.close()

        NFFT = bu.prev_pow_2( len( cdat[:,dcol] ) )
        cpsd, freqs = mlab.psd(cdat[:, dcol]-np.mean(cdat[:,dcol]), Fs = Fs, NFFT = NFFT) 

        # plt.figure()
        # plt.loglog(freqs, cpsd)
        # plt.show()

        #freq_idx = np.argmin( np.abs(freqs - drive_freq) )
        #freq_idx2 = np.argmin( np.abs(freqs - 2.0*drive_freq) )

        ## take correlation with drive and drive^2

        response = cdat[skip_dat:-skip_dat, dcol]
        response -= np.mean(response)
        if( elec < 3 ):
            drive = cdat[skip_dat:-skip_dat, elec+8]
        else:
            drive = cdat[skip_dat:-skip_dat, elec+9]
        if( cpos < 0 and gain == 0.2): drive *= -1
        drive -= np.mean(drive)
        drive2 = drive**2
        drive2 -= np.mean(drive2)

        dummy_freq = 38 ## freq doesn't matter since we do 0 offset only
        corr_dr = bu.corr_func(drive, response, Fs, dummy_freq)[0]
        corr_dr2 = bu.corr_func(drive2, response, Fs, dummy_freq)[0]

        ## find the error on the correlation from a few dummy frequencies
        tvec = np.linspace(0, (len(response)-1.0)/Fs, len(response))
        rand_phase = np.random.rand()*2.*np.pi
        s_m5 = np.sin(2*np.pi*tvec*(drive_freq-5.0) + rand_phase)
        s_p5 = np.sin(2*np.pi*tvec*(drive_freq+5.0) + rand_phase)
        corr_dr_m5 = bu.corr_func(s_m5, response, Fs, dummy_freq)[0]
        corr_dr_p5 = bu.corr_func(s_p5, response, Fs, dummy_freq)[0]
        ## upper bound on the error


        if(False and elec == 3):
            plt.figure()
            plt.plot(drive)
            #plt.plot(response)
            b,a = sig.butter(3, (2.0*drive_freq+5)/(Fs/2.0) )
            filt_resp = sig.filtfilt(b,a,response)
            plt.plot(filt_resp,'c')
            plt.plot(drive2)
            #plt.plot(s_m5)
            #plt.plot(s_p5)
            plt.show()

        tot_corr_dr.append(corr_dr)
        tot_corr_dr2.append(corr_dr2)
        tot_corr_dr_err.append( np.mean( np.abs([corr_dr_m5, corr_dr_p5])))

    tot_dat[fidx, 0] = cpos*gain
    tot_dat[fidx, 1] = np.mean(tot_corr_dr)
    tot_dat[fidx, 2] = np.mean(tot_corr_dr2)
    tot_dat[fidx, 3] = np.std(tot_corr_dr)/np.sqrt( len(tot_corr_dr) )
    tot_dat[fidx, 4] = np.std(tot_corr_dr2)/np.sqrt( len(tot_corr_dr2) )
    tot_dat_pts.append([cpos*gain, tot_corr_dr, tot_corr_dr2, tot_corr_dr_err])

    if( False and elec == 0 ):
        plt.figure()
        plt.plot( drive )
        plt.plot( response )
        plt.show()

tot_dat = np.array(tot_dat)

frange = [-200, 8000] ## fit range

plt.figure(figd.number)
for t in tot_dat_pts:
    plt.errorbar(t[0]*np.ones_like(t[1]), t[1], yerr=1e-10*np.array(t[3]), fmt=col[cidx]+'.', markersize=3, capsize=0)
    #plt.plot(t[0]*np.ones_like(t[2]), np.sqrt(t[2]), 'r.', markersize=1)
plt.errorbar(tot_dat[:,0],tot_dat[:,1], tot_dat[:,3], fmt=col[cidx]+'s', label=leg_list[cidx])
#plt.errorbar(tot_dat[:,0],np.sqrt(tot_dat[:,2]), np.sqrt(tot_dat[:,4]), fmt='ro-', label="2*f0")
plt.legend(loc=0, numpoints=1)

def ffn(x, p0, p1):
    return p0*x + p1

gpts = tot_dat[:,0] < 1000
fit_pts = tot_dat[:,0] < 1e5 ##np.logical_and(np.abs(tot_dat[:,0])>1.5, np.abs(tot_dat[:,0])<3) ## < 1000
sigvals = tot_dat[:,3]
sigvals[sigvals <1e-10] = np.max( sigvals )
sigvals[np.isnan(sigvals)] = np.max( sigvals )
fit_pts = np.logical_and(fit_pts, np.logical_not(np.isnan(tot_dat[:,1])))
p = np.polyfit(tot_dat[fit_pts,0], tot_dat[fit_pts,1], 1)
bp, bcov = sp.curve_fit( ffn, tot_dat[fit_pts,0], tot_dat[fit_pts,1], p0=p, sigma = sigvals[fit_pts] )

xx = np.linspace(np.min(tot_dat),np.max(tot_dat),1e3)
plt.plot(xx, ffn(xx,*bp), col[cidx], linewidth=1.5 )
intcpt = -bp[1]/bp[0]
yy = plt.ylim()
plt.plot([intcpt,intcpt],yy,col[cidx]+'--') ##,label="%.3f"%intcpt)
print "Offset voltage: ", intcpt
#print "Offset voltage: ", -bp[1]/(2*bp[0])

plt.title("Correlation with drive")


plt.figure(figd2.number)
for t in tot_dat_pts:
    plt.errorbar(t[0]*np.ones_like(t[1]), t[2], yerr=1e-10*np.array(t[3]), fmt=col[cidx]+'.', markersize=3, capsize=0)
    #plt.plot(t[0]*np.ones_like(t[2]), np.sqrt(t[2]), 'r.', markersize=1)
plt.errorbar(tot_dat[:,0],tot_dat[:,2], tot_dat[:,4], fmt=col[cidx]+'s', label=leg_list[cidx])
#plt.errorbar(tot_dat[:,0],np.sqrt(tot_dat[:,2]), np.sqrt(tot_dat[:,4]), fmt='ro-', label="2*f0")
plt.legend(loc=0, numpoints=1)

plt.title("Correlation with drive$^2$")

# plt.figure()
# plt.plot(tot_dat[:,0],tot_dat[:,3], 'ks-', label="f0")
# plt.plot(tot_dat[:,0],tot_dat[:,4], 'ro-', label="2*f0")
# plt.legend(loc=0, numpoints=1)
# plt.title("PSD")

plt.show()


