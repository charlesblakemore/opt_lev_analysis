## load all files in a directory and plot the correlation of the resonse
## with the drive signal versus time

import numpy as np
import matplotlib, calendar
import matplotlib.pyplot as plt
import os, re, time, glob
import bead_util as bu
import scipy.signal as sp
import scipy.optimize as opt
import cPickle as pickle

path = r"/data/20141212/Bead2/charge_cantilever_leave"
fdrive = 307.
make_plot = True

data_columns = [0, 1] ## column to calculate the correlation against
drive_column = -1 ## column containing drive signal

def getphase(fname):
        print "Getting phase from: ", fname 
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
        fsamp = attribs["Fsamp"]
        xdat = dat[:,data_columns[0]]

        xdat = np.append(xdat, np.zeros( int(fsamp/fdrive) ))
        corr2 = np.correlate(xdat,dat[:,drive_column])
        maxv = np.argmax(corr2) 

        cf.close()

        print maxv
        return maxv


def getdata(fname, maxv):

	print "Processing ", fname
        dat, attribs, cf = bu.getdata(os.path.join(path, fname))

        if( len(attribs) > 0 ):
            fsamp = attribs["Fsamp"]

        xdat = dat[:,data_columns[0]]

        lentrace = len(xdat)
        ## zero pad one cycle
        xdat = np.append(xdat, np.zeros( fsamp/fdrive ))
        corr_full = np.correlate( xdat-np.median(xdat), dat[:,drive_column]-np.median(dat[:,drive_column]))/lentrace
        cout = corr_full[ len(corr_full)/2 ]
        cm_out = np.max( np.abs(corr_full))

        return cout, cm_out

def get_most_recent_file(p):

    filelist = os.listdir(p)
    
    mtime = 0
    mrf = ""
    for fin in filelist:
        if( fin[-3:] != ".h5" ):
            continue
        f = os.path.join(path, fin) 
        if os.path.getmtime(f)>mtime:
            mrf = f
            mtime = os.path.getmtime(f)     

    return mrf


best_phase = None
corr_data = []
corr_max_data = []

def sort_fun( s ):
  return int(re.findall("\d+.h5",s)[0][:-3])

flist = sorted(glob.glob(path + "/*.h5"), key=sort_fun)
#while( True ):
    ## get the most recent file in the directory and calculate the correlation
for cfile in flist:
    print cfile

    if( not best_phase ):
        best_phase = getphase( cfile )

    corr, corr_max = getdata( cfile, best_phase )
    corr_data.append(corr)
    corr_max_data.append(corr_max)


plt.figure()
plt.plot(corr_data)
plt.plot(corr_max_data)
plt.show()

    
