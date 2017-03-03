## load all files in a directory and plot the correlation of the response
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, time, glob, sys
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

path = "/data/20160325/bead1/chargelp_cal2"
#path = "/data/20160320/bead1/chargelp_cal2"
ts = 10.

fdrive = 29.
make_plot = True
reprocess_file = True
savefig = True

data_columns = [0, 1] ##[0,1] ## column to calculate the correlation against
drive_column = 12 ## column containing drive signal

drive_millivolt = 400.
scale_fac = -0.03

datacut = 5000

force = (drive_millivolt/1000.)*200*bu.e_charge/4e-3 ## N

def keyfunc(s):
	cs = re.findall("_\d+.h5", s)
	return int(cs[0][1:-3])


def getdata(fname, maxv):

	print "Processing ", fname
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))

        fnum = keyfunc( fname )

        if( len(attribs) > 0 ):
            fsamp = attribs["Fsamp"]

        xdat = dat[:,data_columns[1]]

        lentrace = len(xdat)
        ## zero pad one cycle
        corr_full = bu.corr_func( dat[:,drive_column], xdat, fsamp, fdrive)

        # plt.figure()
        # plt.plot(corr_full)
        # plt.show()

        return corr_full[0], np.max(corr_full), fnum


folder_name = path.split("/")
folder_name = "_".join( folder_name[2:] )

best_phase = None
corr_data = []

if make_plot:
    fig0 = plt.figure()
    


if reprocess_file:

    init_list = glob.glob(path + "/*xyzcool*_%dmV*.h5" % drive_millivolt)
    if( len( init_list ) == 0 ):
        print "No files found at %d mV, exiting" % drive_millivolt
        sys.exit(1)

    files = sorted(init_list, key=keyfunc)
    print files
    for f in files[::1]:
        cfile = f
        corr = getdata( cfile, 10. )
        corr_data.append(corr )

    corr_data = np.array(corr_data)

    sing_charge = np.abs(np.abs( corr_data[:,0]/scale_fac ) - 1) < 0.5
    #sing_charge = np.zeros_like(corr_data[:,0])

    if make_plot:
	#nfiles = len(np.array(corr_data)[:,0])
	#t = np.linspace(0, nfiles-1, nfiles) * 10 
        plt.plot(corr_data[:,2], corr_data[:,0]/scale_fac, linewidth=1.5, color='k')
        if( np.sum(sing_charge) > 0):
          plt.plot(corr_data[sing_charge,2], corr_data[sing_charge,0]/scale_fac, 'ro', linewidth=1.5)
	plt.grid()
	#plt.ylim(-5,5)
        #plt.xlabel("Time [s]")
        plt.ylabel("Bead response [# of e$^{-}$]")

        plt.title(folder_name)
        plt.xlabel("File number")

        if(savefig):
          plt.savefig("plots/charge_steps_%s.pdf"%folder_name)

    ## now for each file, do a time domain fit to the filtered trace to
    ## get the amplitude



    if( np.sum(sing_charge) > 0):
            fnum_list = corr_data[sing_charge,2]
            amp_list = []
            for fname in files[::1]:

                fnum = keyfunc( fname )
                if fnum not in fnum_list: continue 

                dat, attribs, cf = bu.getdata(os.path.join(path, fname))
                nyquist = attribs['Fsamp']/2.
                b,a = sp.butter(3, [(fdrive-2)/nyquist, (fdrive+2)/nyquist], btype="bandpass") 
                xdat = dat[:,data_columns[1]]        
                xf = sp.filtfilt(b,a,xdat)

                if(False):
                  plt.figure()
                  plt.plot(xdat-np.mean(xdat))
                  plt.plot(xf)
                  yy=plt.ylim()
                  plt.plot([datacut,datacut],yy,'r')
                  plt.plot([len(xf)-datacut,len(xf)-datacut],yy,'r')
                  plt.show()

                amp_list.append([fnum, np.sqrt(2)*np.std(xf[datacut:-datacut])])


            amp_list = np.array(amp_list)
            plt.figure()
            mean_cal =  np.mean( amp_list[:,1] )
            mean_err =  np.std( amp_list[:,1] )/np.sqrt( len(amp_list[:,1]) )
            ax = plt.gca()
            ax.fill_between( amp_list[:,0], mean_cal-mean_err, mean_cal+mean_err, facecolor='r', edgecolor='None', alpha=0.5 )
            plt.plot( amp_list[:,0], mean_cal*np.ones_like(amp_list[:,0]), 'r--' )
            plt.plot( amp_list[:,0], amp_list[:,1], 'ko')

            plt.xlim([amp_list[0,0], amp_list[-1,0]])

            force_err = force*( 1./(mean_cal-mean_err) - 1./(mean_cal+mean_err) )

            force_str = "Cal fac: %.2e +/- %.3e [N/V] " % (force/mean_cal, force_err)
            plt.title( force_str )
            print force_str

            plt.xlabel("File number")
            plt.ylabel("Response [V]")
            plt.title(folder_name)

            if(savefig):
                    plt.savefig("plots/charge_cal_%s.pdf"%folder_name)

    plt.show()
