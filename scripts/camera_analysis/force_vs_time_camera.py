import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import cv2, sys, glob, os
#from multiprocessing import Pool
#from itertools import product

# Calculate the position of the bead vs time from the camera
# and compare it to the PDP
#
###############################################################

idx_to_plot = 199 ## index from dir file below to use
plot_pulse_locs = True ## plot the location of the laser flashes
plot_fft_image = True ## plot the image at the drive freq

image_smoothing = 31 ## pixels, set to 0 to disable
drive_type = 'stage' ## electrode, stage, or none

data_column = 1 ## data to plot, x=0, y=1, z=2
buffer_pts = 0 ## number of points at beginning and end of file to drop

offset_freq = [3.,1.] ## number of Hz offset from drive, and bandwidth

force_remake_file = False ## force recalculation of the ffts saved to disk

## load the list of data from a text file into a dict
ddict = bu.load_dir_file( "/home/dcmoore/opt_lev/scripts/cant_force/dir_file.txt" )
###############################################################

#pool = Pool() ## Use all CPUs by default for parallel FFTs
#chunksize = 20

if(image_smoothing % 2 == 0): image_smoothing+=1 ## must be odd

cdir = ddict[str(idx_to_plot)]
ddict = {"path": cdir[0], "drive_idx": cdir[2]}

dfiles = sorted( glob.glob( os.path.join( ddict["path"], "*.h5" )),  key=bu.simple_sort )
vfiles = glob.glob( os.path.join( ddict["path"], "*.avi" ) )

if( len(dfiles) == 0 ):
    print "Couldn't find any data"
    sys.exit(1)
if( len(vfiles) == 0 ):
    print "Couldn't find any videos"
    sys.exit(1)
if( len(vfiles) > 1 ):
    print "Warning, found more than 1 video, using first file"



def find_flash( vdict ):

    tot_amp = np.zeros( vdict['nframes'] )
    for n in range( vdict['nframes'] ):

        ret, frame = vh.read()
        if( not ret ):
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tot_amp[n] = np.sum(gray_frame)

    thresh = np.mean(tot_amp) + 4.*np.std(tot_amp)
    above_thresh = tot_amp > thresh
    goes_above_thresh = np.argwhere(np.logical_and( above_thresh, np.logical_not(np.roll( above_thresh, 1)) ))
    
    if( plot_pulse_locs ):
        plt.figure()
        plt.plot( range( vdict['nframes'] ), tot_amp )
        xx = plt.xlim()
        plt.plot(xx, [thresh, thresh], 'r')
        plt.plot( goes_above_thresh, tot_amp[goes_above_thresh], 'ro' )
        plt.show()

    if( len(goes_above_thresh) == 0 ):
        return np.array([0,])

    return goes_above_thresh

def get_subvid( vdict, vidx ):

    ## go two frames back since we'll read one junk frame to get the size,
    ## and then read in loop advances one more before giving the desired frame
    vdict['handle'].set(bu.CV_CAP_PROP_POS_FRAMES, np.max([vidx[0]-2,0]) )

    ## preallocate array for speed
    ret, frame = vdict['handle'].read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #out_arr = np.zeros( (np.shape(frame)[0], np.shape(frame)[1], vidx[1]), dtype=np.uint8 )
    out_arr = np.zeros( (np.shape(frame)[0], np.shape(frame)[1], vidx[1]) )

    for n in range(vidx[1]):
        ret, frame = vdict['handle'].read()
        if( not ret ):
            print "Error reading frame %d" % n
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_arr[:,:,n] = frame

    return out_arr


def get_fft_image( vdict, vid ):

    nx, ny, nz = np.shape(vid)
    fft_len = nz/2+1 if nz%2==0 else (nz+1)/2
    output = np.zeros( (nx,ny,fft_len), dtype=np.float32 ) + np.zeros( (nx,ny,fft_len), dtype=np.float32 )*1j
    
    print "Taking FFTs...., %d total"%nx

    for xidx in range(nx):
        if( xidx % 10 == 0 ): print xidx
        for yidx in range(ny):
            output[xidx,yidx,:] = np.fft.rfft( vid[xidx,yidx,:].flatten() )

    return output

def make_image_plot( zin, tit="", clim=[] ):

    if( image_smoothing > 0 ):
        z = cv2.GaussianBlur(zin,(image_smoothing,image_smoothing),0)

    fig=plt.figure()
    plt.imshow( z )
    if( clim ):
        plt.clim(clim)
    else:
        plt.clim([np.percentile(z.flatten(), 2), np.percentile( z.flatten(), 98 )])
    #plt.clim([0, np.max(z)])
    plt.title(tit)
    plt.colorbar()
    return fig

## load the video
vfile = vfiles[0]
vh = cv2.VideoCapture(vfile)
vdict = {'handle': vh, 'nframes': int(vh.get(bu.CV_CAP_PROP_FRAME_COUNT)), 'fps': vh.get(bu.CV_CAP_PROP_FPS)}

## first, get the indices of any flashes
print "Finding location of flashes"
pulse_locs = find_flash( vdict )

print "Found %d pulse(s) at indices: "%len(pulse_locs), pulse_locs.T

if( len(pulse_locs) != len(dfiles) ):
    print "Error, number of data files doesn't match video chunks, exiting"
    print "%d data files, %d video flashes"%(len(pulse_locs),len(dfiles))
    #sys.exit(1)

fft_image = []
niter = 0.

data_file_path = ddict["path"].replace("/data/","/home/dcmoore/analysis/")
## make directory if it doesn't exist
if(not os.path.isdir(data_file_path) ):
    os.makedirs(data_file_path)
fft_imag_file = os.path.join( data_file_path, "vid_fft_file.npy" )
file_exists = os.path.isfile( fft_imag_file ) and not force_remake_file

for df,vidx in zip(dfiles, pulse_locs):

    print "Loading: ", df
    cdat, attribs, fhand = bu.getdata( df )

    stagemon = cdat[:,ddict["drive_idx"]]
    truncdata = cdat[:,data_column]
    if( buffer_pts > 0 ):
        stagemon, truncdata = stagemon[buffer_pts:-buffer_pts], truncdata[buffer_pts:-buffer_pts]

    if( drive_type == 'electrode'):
        drive_freq = attribs['electrode_settings'][16]
    elif( drive_type == 'stage' ):
        drive_freq = attribs['stage_settings'][6]
    else:
        print "No drive, assuming default freq = 10 Hz"
        drive_freq = 10
    print "Drive frequency is %f Hz"%drive_freq
        
    Fs = attribs['Fsamp']        

    nbufferframes = int(np.round((1.*buffer_pts/Fs)*vdict['fps']))
    nvidframes = int(np.round((1.*len(truncdata)/Fs)*vdict['fps']))

    cvidx = [vidx+nbufferframes, nvidframes]

    print "Loading corresponding video chunk"
    if(not file_exists):
        cvid = get_subvid( vdict, cvidx )

        if(plot_fft_image):
            ## This takes the fft of the time stream in each pixel
            ## if desired
            curr_fft_image = np.abs( get_fft_image( vdict, cvid ) )**2

            if( len(fft_image) == 0):
                fft_image = curr_fft_image
            else:
                fft_image += curr_fft_image

            niter+=1.0

if(not file_exists):
    fft_image = np.sqrt( fft_image/niter )
    np.save( fft_imag_file, fft_image)
else:
    fft_image = np.load( fft_imag_file )


if( plot_fft_image ):

    fft_freqs = np.linspace( 0, vdict['fps']/2, np.shape(fft_image)[2] )

    ## first at drive freq
    drive_idx = np.argmin( np.abs(fft_freqs - drive_freq))
    drive_idx2 = np.argmin( np.abs(fft_freqs - 2*drive_freq))
    f1, f2 = offset_freq[0]-offset_freq[1]/2.0, offset_freq[0]+offset_freq[1]/2.0
    offdrive_idx0 = np.argmin( np.abs(fft_freqs - drive_freq - f1))
    offdrive_idx1 = np.argmin( np.abs(fft_freqs - drive_freq - f2))

    plt.figure()
    #for idx in range(0,np.shape(fft_image)[1],10):
    #    plt.loglog( fft_freqs, np.abs( fft_image[0,idx,:].flatten() ) )

    curr_image = fft_image ##[30:70,30:70,:]
    curr_arr = np.reshape( curr_image, (np.shape(curr_image)[0]*np.shape(curr_image)[1], np.shape(curr_image)[2]) )
    plt.loglog(fft_freqs, np.sqrt(np.mean( np.abs(curr_arr)**2, axis=0 )) )
    
    plt.show()


    f1=make_image_plot(np.abs(fft_image[:,:,drive_idx]), 
                       "Bead response, drive freq, %f Hz"%drive_freq)
    cclim = plt.gci().get_clim()
    #f2=make_image_plot(np.abs(fft_image[:,:,drive_idx2]), 
    #                   "Bead response, 2*drive freq, %f Hz"%(2*drive_freq))
    z1 = np.abs(fft_image[:,:,drive_idx])
    z2 = np.mean(np.abs(fft_image[:,:,offdrive_idx0:offdrive_idx1]), 2)
    f3=make_image_plot(z2, 
                       "Bead response, off drive, %f Hz"%(drive_freq+offset_freq[0]), clim=cclim)
    f4=make_image_plot( (z1-z2)/z2,
                       "Bead response, [(drive freq) - (off drive freq)]/(off drive freq)")

    vdict['handle'].set(bu.CV_CAP_PROP_POS_FRAMES, 0)
    ret, frame = vdict['handle'].read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f5=make_image_plot( gray_frame )

    plt.show()
