## measure the force from the cantilever
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import matplotlib.mlab as mlab

data_dir = "/data/20151208/bead1/dc_sweep3"

using_dc_supply = True

NFFT = 2**15

def sort_fun( s ):
    if( using_dc_supply ):
        return float(re.findall("dcps-?\d+mVdc", s)[0][4:-4])
    else:
        return float(re.findall("-?\d+mVdc", s)[0][:-4])

#flist = sorted(glob.glob(os.path.join(data_dir, "*mVdc_stageX*nmY*nmZ*nm.h5")), key = sort_fun)
flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)

avg_dict = {}

elec_list = [3,] ##[1,2,3,4,5,6]
dcol_list = [0,1,2]
tot_dat = np.zeros( [len(flist), len(elec_list), len(dcol_list), 3])
for fidx,f in enumerate(flist):

    cpos = sort_fun(f)

    print("Vdc = ", cpos)

    cdat, attribs, _ = bu.getdata( f )

    Fs = attribs['Fsamp']

    #cpsd, freqs = mlab.psd(cdat[:, 1]-np.mean(cdat[:,1]), Fs = Fs, NFFT = NFFT) 
    
    ## take correlation with drive and drive^2

    for eidx,elec in enumerate(elec_list):
        for didx,dcol in enumerate(dcol_list):

            response = cdat[:, dcol]
            drive = cdat[:, 9+elec]
            drive2 = drive**2
            drive2 -= np.mean(drive2)

            response -= np.mean(response)
            drive -= np.mean(drive)

            dummy_freq = 41 ## freq doesn't matter since we do 0 offset only
            corr_dr = bu.corr_func(drive, response, Fs, dummy_freq)[0]
            corr_dr2 = bu.corr_func(drive2, response, Fs, dummy_freq)[0]
            #cpsd, freqs = mlab.psd(response, Fs = Fs, NFFT = 2**16) 
            #dfidx = np.argmin( np.abs(freqs-dummy_freq) )
            #corr_dr = cpsd[ dfidx ]

            #plt.figure()
            #plt.loglog( freqs, cpsd )
            #plt.loglog( freqs[dfidx], cpsd[dfidx], 'ro' )
            #plt.show()

            tot_dat[fidx, eidx, didx, 0] = cpos
            tot_dat[fidx, eidx, didx, 1] = corr_dr
            tot_dat[fidx, eidx, didx, 2] = corr_dr2

            if( False and elec == 3 ):
                plt.figure()
                plt.plot( drive )
                plt.plot( response )
                plt.show()

            ## save the average
            if( elec == 3 and dcol == 1 ):
                if( cpos in avg_dict ):
                    avg_dict[cpos].append( corr_dr )
                else:
                    avg_dict[cpos] = [corr_dr,]


tot_dat = np.array(tot_dat)

frange = [-200, 8000] ## fit range

avg_vals = []
for cc in avg_dict:
    avg_vals.append( [cc, np.mean( avg_dict[cc] ), np.std( avg_dict[cc] )/len(avg_dict[cc]) ] )
avg_vals = np.array(avg_vals)

print(avg_vals)

def make_plot( x,y,plot=True ):
    if(plot):
        plt.plot(x,y, 'ks', label = "Drive")
    gpts = np.logical_and( x > frange[0], x < frange[1] )
    p = np.polyfit( x[gpts], y[gpts], 1 )
    xx = np.linspace( frange[0], frange[1], 1e3 )
    if(plot):
        plt.plot(xx, np.polyval(p, xx), 'r', linewidth=1.5)
    bestx = -p[1]/p[0]
    if(plot):
        yy = plt.ylim()
        plt.plot( [bestx, bestx], yy, 'r--', linewidth=1.5, label="%.1f mV" % bestx  )
        plt.legend(loc="lower right", numpoints=1)
    return bestx 

pot_arr = np.zeros([len(elec_list), 3])
for eidx, elec in enumerate(elec_list):

    plt.figure()

    # plt.subplot(3,1,1)
    xpot = make_plot(tot_dat[:,eidx,0,0], tot_dat[:,eidx,0,1], plot=False) 
    # plt.title("Electrode %d" % elec)
    # xvals=tot_dat[:,eidx,0,0]
    # plt.xlim(np.min(xvals)-10, np.max(xvals)+10)

    #plt.subplot(3,1,2)
    ypot = make_plot(tot_dat[:,eidx,1,0], tot_dat[:,eidx,1,1]) 
    xvals=tot_dat[:,eidx,1,0]
    plt.xlim(np.min(xvals)-10, np.max(xvals)+10)
    plt.errorbar( avg_vals[:,0], avg_vals[:,1], yerr=avg_vals[:,2], fmt='b.', linewidth=1.5)

    # plt.subplot(3,1,3)
    zpot = make_plot(tot_dat[:,eidx,2,0], tot_dat[:,eidx,2,1], plot=False) 
    # xvals=tot_dat[:,eidx,2,0]
    # plt.xlim(np.min(xvals)-10, np.max(xvals)+10)

    pot_arr[eidx,:] =  [xpot, ypot, zpot]

# plt.figure()
# plt.plot( elec_list, pot_arr[:,0], 'ks', label="x response" )
# plt.plot( elec_list, pot_arr[:,1], 'rs', label="y response" )
# plt.plot( elec_list, pot_arr[:,2], 'gs', label="z response" )
# plt.title( "electrode potentials" )
# plt.xlim([0,6])


# print "X best potentials:"
# print pot_arr[:,0]
# print "Y best potentials:"
# print pot_arr[:,1]

plt.show()


