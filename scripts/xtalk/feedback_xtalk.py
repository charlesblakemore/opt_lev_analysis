import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import sys, glob, os

# Calculate the cross talk between the z-feedback and the y
# response, then use this to correct a data file for this xtalk
#
###############################################################

idx_for_xtalk = 81 ## index from dir file below to use
idx_for_data = 81 ##

data_column = 1 ## data to plot, x=0, y=1, z=2
buffer_pts = 0 ## number of points at beginning and end of file to drop
xtalk_column = 16

force_remake_file = False ## force recalculation of the ffts saved to disk

## load the list of data from a text file into a dict
ddict = bu.load_dir_file( "/home/dcmoore/opt_lev/scripts/cant_force/dir_file.txt" )
###############################################################


data_dir = ddict[str(idx_for_data)]
data_dict = bu.data_list_to_dict( data_dir )

xtalk_dir = ddict[str(idx_for_xtalk)]
xtalk_dict = bu.data_list_to_dict( xtalk_dir )

## first make xtalk template
xtalk_files = sorted( glob.glob( os.path.join( xtalk_dict["path"], "*.h5" )),  key=bu.simple_sort )

tfz, cpsdz, ntf = [], [], 0

for f in xtalk_files:

    print "Loading: ", f
    cdat, attribs, fhand = bu.getdata( f )

    Fs = attribs['Fsamp']

    resp_data = cdat[:,data_column]
    drive_data = cdat[:,xtalk_column]

    # plt.figure()
    # plt.plot(resp_data)
    # plt.plot(drive_data)
    # plt.show()

    ctfz = np.fft.rfft( resp_data ) / np.fft.rfft( drive_data )

    if( len(tfz) == 0 ):
        tfz = ctfz
        cpsdz = np.abs(ctfz)**2
    else:
        tfz += ctfz
        cpsdz += np.abs(ctfz)**2

    ntf += 1

tfz /= ntf
cpsdz /= ntf

freqs = np.fft.rfftfreq( len(resp_data), 1./Fs )

plt.figure()
plt.loglog( freqs, np.abs(tfz) )

plt.figure()
plt.loglog( freqs, np.sqrt(cpsdz) )
plt.show()
