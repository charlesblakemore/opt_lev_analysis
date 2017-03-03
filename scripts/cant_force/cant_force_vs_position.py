## measure the force from the cantilever, averaging over files
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.signal as signal
import scipy.interpolate as interp
import scipy.optimize as opt
import cPickle as pickle
<<<<<<< HEAD
=======
from mpl_toolkits.mplot3d import Axes3D
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959

###################################################################################
do_mean_subtract = True  ## subtract mean position from data
do_poly_fit = False  ## fit data to 1/r^2 (for DC bias data)
<<<<<<< HEAD
do_2d_fit = True ## fit data vs position and voltage to 2d function
sep_forward_backward = False ## handle forward and backward separately
idx_to_plot = [217,218, 225] #[217,218,219,221,222] ## indices from dir file below to use
diff_dir = None ##'Y' ## if set, take difference between two positions

=======
do_2d_fit = False ## fit data vs position and voltage to 2d function
sep_forward_backward = False ## handle forward and backward separately
match_overlap_region = True ## for multiple overlapping files, match together
#idx_to_plot = [293,296,299,302,305,308] ## neg drives
#idx_to_plot = [292,295,298,301,304,309] ## pos drives
#idx_to_plot = [291,294,297,300,303,306,307] ## 0V
idx_to_plot = [311,]
diff_dir = None ##'Y' ## if set, take difference between two positions

sig_dir = 'y' ## Direction of the expected signal
pos_offset = 60. ## um, distance of closest approach (needed to make voltage template)

>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959
data_columns = [0,1,2] ## data to plot, x=0, y=1, z=2
mon_columns = [3,7]  # columns used to monitor position, empty to ignore
plot_title = 'Force vs. position'
nbins = 8 ##40  ## number of bins vs. bead position

<<<<<<< HEAD
max_files = 10 ## max files to process per directory
=======
max_files = 50 ## max files to process per directory
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959
force_remake_file = True ## force recalculation over all files

buffer_points = 1000 ## number of points to cut from beginning and end of file

make_opt_filt_plot = True
plot_psds = False
dirs_to_plot=['x','y','z']

## load the list of data from a text file into a dict
ddict = bu.load_dir_file( "/home/arider/opt_lev/scripts/cant_force/dir_file.txt" )
###################################################################################

cant_step_per_V = 8. ##um
colors_yeay = ['b', 'r', 'g', 'k', 'c', 'm', 'y', [0.5,0.5,0.5], [0, 0.5, 1], [1, 0.5, 0], [0.5, 1,0]]
colors_yeay = colors_yeay + colors_yeay + colors_yeay + colors_yeay

dirs = []
# dir, label, drive_column, numharmonics, monmin, monmax, closest_app, cal_fac
for idx in idx_to_plot:
    dirs.append( ddict[str(idx)] )
print dirs

sbins = 4  # number of bins to either side of drive_freq to integrate

def sort_fun( s ):
    cc = re.findall("-?\d+.h5", s)
    if( len(cc) > 0 ):
        return float(cc[0][:-3])
    else:
        return -1.

def bin(xvec, yvec, binmin=0, binmax=10, n=300):
    bin_edges = np.linspace(binmin, binmax, n+1)
    inds = np.digitize(xvec, bin_edges, right = False)
    bins = bin_edges[:-1] + np.diff(bin_edges)/2.0
    avs = np.zeros(n)
    ers = np.zeros(n)
    for i in range(len(bins)):
        cidx = inds == i
        if( np.sum(cidx) > 0 ):
            avs[i] = np.median(yvec[cidx])
            ers[i] = np.std(yvec[cidx])/np.sqrt(len(yvec[cidx]))

    return avs, ers, bins 

def get_stage_dir_pos( s, d ):
    if( d == 'X' ):
        coord = re.findall("stageX\d+nm", s)
        if( len(coord) == 0 ):
            return None
        else:
            return int(coord[0][6:-2])

    if( d == 'Y' ):
        coord = re.findall("nmY\d+nm", s)
        if( len(coord) == 0 ):
            return None
        else:
            return int(coord[0][3:-2])

<<<<<<< HEAD
def process_files(data_dir, num_files, dc_val=-999999, pos_at_10V=0., 
                  monmin=20., monmax=100., conv_fac =1., drive_indx=19):

    out_dict = {}

    if( dc_val > -999999 ):
        print "Data with DC bias (V): ", dc_val
        flist = sorted(glob.glob(os.path.join(data_dir, "*Hz%dmVdc*.h5"%dc_val)), key = sort_fun)
        if( len( flist ) == 0 ):
            ## must be the dc supply
            flist = sorted(glob.glob(os.path.join(data_dir, "*dcps%dmVdc*.h5"%dc_val)), key = sort_fun)
        flist1 = []
    else:
        flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)
        flist1 = []

    tempdata, tempattribs, temphandle = bu.getdata(flist[0])
    drive_freq = tempattribs['stage_settings'][-2]
    temphandle.close()  


    for fidx,f in enumerate(flist[:num_files]):

        print f 
        ## Load the data
        cdat, attribs, fhand = bu.getdata( f )
        if( len(cdat) == 0):
            print "Skipping: ", f
            continue         

        Fs = attribs['Fsamp']        

        ## get the data and cut off the beginning and the end to avoid edge effects
        cmonz = cdat[:,drive_indx][buffer_points:-buffer_points] 
        truncdata_x = cdat[:,data_columns[0]][buffer_points:-buffer_points]
        truncdata_y = cdat[:,data_columns[1]][buffer_points:-buffer_points]
        truncdata_z = cdat[:,data_columns[2]][buffer_points:-buffer_points]

        ## Subtract the mean to compensate for long time drift
        if(do_mean_subtract):
            truncdata_x = truncdata_x - np.mean(truncdata_x)
            truncdata_y = truncdata_y - np.mean(truncdata_y)
            truncdata_z = truncdata_z - np.mean(truncdata_z)

        truncdata_dict = {'x': truncdata_x, 
                          'y': truncdata_y, 
                          'z': truncdata_z}

        ## Consider stage travel direction separately
        ## filter the monitor around the drive freq
        b,a = signal.butter(3,(drive_freq+2.)/(Fs/2.), btype='lowpass')
        cmonz_filt = signal.filtfilt( b, a, cmonz )
        monderiv = np.gradient(cmonz_filt)
        posmask = monderiv >= 0
        negmask = monderiv < 0 
        allmask = np.logical_or(posmask, negmask)

        ## optimal filter 
        cpos = pos_at_10V + cant_step_per_V*(10. - cmonz)
        cdrive = bu.get_chameleon_force( cpos*1e-6 )
        cdrive -= np.mean(cdrive)
        ## convert newtons to V
        cdrive /= conv_fac

        st = np.fft.rfft( cdrive.flatten() )
        J = np.ones_like( st )
        norm_fac = np.real(np.sum(np.abs(st)**2/J))

        ## now bin the data, separating into forward and backward
        for col in ['x','y','z']:
            for mask, sdir in zip([posmask,negmask,allmask],['pos','neg','both']):

                btrace, cerr, bins = bin( cpos[mask], truncdata_dict[col][mask]*conv_fac, 
                                          binmin=monmin, binmax=monmax, n=nbins)
                cname = 'binned_dat_' + col + '_' + sdir

                if( sdir == 'both' ):
                    cpsd,cfreq = mlab.psd(  truncdata_dict[col], 
                                            NFFT=len(truncdata_dict[col]), Fs=Fs )
                    cpsd *= conv_fac**2

                    vt = np.fft.rfft( truncdata_dict[col][mask] )
                    of_amp = np.real( np.sum( np.conj(st)*vt/J ) / norm_fac )

                else:
                    cpsd, cfreq = [],[]
                    of_amp = 0.

                if( not cname in out_dict ):
                    out_dict[ cname ] = [[btrace,], [cerr,], bins, [of_amp,],[cpsd,],cfreq]
                else:
                    out_dict[ cname ][0].append(btrace)
                    out_dict[ cname ][1].append(cerr)
                    out_dict[ cname ][3].append(of_amp)
                    out_dict[ cname ][4].append(cpsd)

    ## we've now looped through all the files, so average everything down
    for col in ['x','y','z']:
        for sdir in ['pos','neg','both']:
            cname = 'binned_dat_' + col + '_' + sdir

            bavg = np.mean( np.array(out_dict[cname][0]),axis=0)
            berr = np.sqrt( np.sum(np.array(out_dict[cname][1])**2,axis=0)/len(out_dict[cname][1]) )
            
            tot_psd = np.sqrt( np.sum( out_dict[ cname ][4],axis=0)/len( out_dict[ cname ][4] ) )

            out_dict[cname + "_avg"] = [out_dict[cname][2], bavg, berr, tot_psd, cfreq]
    
    return out_dict
=======
def get_pos_from_mon(cmonz, pos_at_10V ):
    return pos_at_10V + cant_step_per_V*(10. - cmonz)

def process_files(data_dir, num_files, dc_val=-999999, pos_at_10V=0., 
                  monmin=0., monmax=10., conv_fac =1., drive_indx=19):
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959

    out_dict = {}

<<<<<<< HEAD
def process_files_old(data_dir, num_files, numharmonics, \
                  monmin, monmax, drive_indx=19, dc_val=-1, pos_at_10V=0., conv_fac=1.):
    ## Load a series of files, acausal filter the cantilever drive and 
    ## some number of harmonics then bin the data and plot position/force
    ## as a function of cantilever position
    global sbins
    global nbins
    
    if( diff_dir ):
        ## figure out what values of the dir position exist
        init_list = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)
        dir_coords = np.unique([get_stage_dir_pos(f,diff_dir) for f in init_list])
        
        flist = sorted(glob.glob(os.path.join(data_dir, "*%s%dnm*.h5"%(diff_dir,dir_coords[0]))), key = sort_fun)
        flist1 = sorted(glob.glob(os.path.join(data_dir, "*%s%dnm*.h5"%(diff_dir,dir_coords[1]))), key = sort_fun)
        ## make sure we have exactly the same number of files
        flist = flist[:len(flist1)]

    elif( dc_val > -999999 ):
        print dc_val
        flist = sorted(glob.glob(os.path.join(data_dir, "*Hz%dmVdc*.h5"%dc_val)), key = sort_fun)
=======
    if( dc_val > -999999 ):
        print "Data with DC bias (V): ", dc_val
        flist = sorted(glob.glob(os.path.join(data_dir, "*Hz%dmVdc*.h5"%abs(dc_val))), key = sort_fun)
        if( len( flist ) == 0 ):
            ## probably wasn't abs valued
            flist = sorted(glob.glob(os.path.join(data_dir, "*Hz%dmVdc*.h5"%dc_val)), key = sort_fun)
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959
        if( len( flist ) == 0 ):
            ## must be the dc supply
            flist = sorted(glob.glob(os.path.join(data_dir, "*dcps%dmVdc*.h5"%abs(dc_val))), key = sort_fun)
        flist1 = []
    else:
        flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)
        flist1 = []

    tempdata, tempattribs, temphandle = bu.getdata(flist[0])
    drive_freq = tempattribs['stage_settings'][-2]
    temphandle.close()  


    for fidx,f in enumerate(flist[1:num_files]):

        print f 
        ## Load the data
        cdat, attribs, fhand = bu.getdata( f )
        if( len(cdat) == 0):
            print "Skipping: ", f
            continue         

        Fs = attribs['Fsamp']        

        ## get the data and cut off the beginning and the end to avoid edge effects
        cmonz = cdat[:,drive_indx][buffer_points:-buffer_points] 
        truncdata_x = cdat[:,data_columns[0]][buffer_points:-buffer_points]
        truncdata_y = cdat[:,data_columns[1]][buffer_points:-buffer_points]
        truncdata_z = cdat[:,data_columns[2]][buffer_points:-buffer_points]

        ## Subtract the mean to compensate for long time drift
        if(do_mean_subtract):
            bm,am = signal.butter(3,(drive_freq-4.)/(Fs/2.), btype='highpass')
            #truncdata_x = truncdata_x - np.mean(truncdata_x)
            #truncdata_y = truncdata_y - np.mean(truncdata_y)
            #truncdata_z = truncdata_z - np.mean(truncdata_z)
            truncdata_x = signal.filtfilt(bm,am,cdat[:,data_columns[0]])[buffer_points:-buffer_points]
            truncdata_y = signal.filtfilt(bm,am,cdat[:,data_columns[1]])[buffer_points:-buffer_points]
            truncdata_z = signal.filtfilt(bm,am,cdat[:,data_columns[2]])[buffer_points:-buffer_points]
            
        truncdata_dict = {'x': truncdata_x, 
                          'y': truncdata_y, 
                          'z': truncdata_z}

        ## Consider stage travel direction separately
        ## filter the monitor around the drive freq
        b,a = signal.butter(3,(drive_freq+2.)/(Fs/2.), btype='lowpass')
        cmonz_filt = signal.filtfilt( b, a, cmonz )
        monderiv = np.gradient(cmonz_filt)
        posmask = monderiv >= 0
        negmask = monderiv < 0 
        allmask = np.logical_or(posmask, negmask)

        ## optimal filter 
<<<<<<< HEAD
        cpos = pos_at_10V + cant_step_per_V*(10. - cmonz)
        cdrive = bu.get_chameleon_force( cpos*1e-6 )
        #cdrive = np.ones_like(cpos)
=======
        cpos = get_pos_from_mon(cmonz, pos_at_10V)
        cdrive = bu.get_chameleon_force( cpos*1e-6 )
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959
        cdrive -= np.mean(cdrive)
        ## convert newtons to V
        cdrive /= conv_fac

        st = np.fft.rfft( cdrive.flatten() )
        J = np.ones_like( st )
        norm_fac = np.real(np.sum(np.abs(st)**2/J))

<<<<<<< HEAD
        ## add some fake signal in
        #truncdata += cdrive*0.00001

        #btrace, cerr, bins = bin(cmonz, ctrace, \
        #                         binmin=monmin, binmax=monmax, n=300)
        btracef, cerrf, binsf = bin(cmonz[posmask], truncdata[posmask], \
                                    binmin=monmin, binmax=monmax, n=nbins)
        btracer, cerrr, binsr = bin(cmonz[negmask], truncdata[negmask], \
                                    binmin=monmin, binmax=monmax, n=nbins)
        btrace, cerr, bins = bin(cmonz, truncdata, \
                                    binmin=monmin, binmax=monmax, n=nbins)
        bmon, monerr, monbins = bin(cmonz, monderiv, binmin=monmin, \
                                    binmax=monmax, n=nbins)

        vt = np.fft.rfft( truncdata )
        #vt_diff = np.fft.rfft( truncdata_diff )
        st = np.fft.rfft( cdrive.flatten() )
        J = np.ones_like( vt )

        ## look at max opt filt output
        # xvals = np.arange(-20000,10000,10)
        # efac = -2*np.pi*1j*(xvals)/len(st)
        # of_vec = []
        # for ei in efac:
        #     of_vec.append(np.real( np.sum( np.conj(st)*vt*np.exp(ei)/J ) / np.sum(np.abs(st)**2/J) ))
        # of_vec = np.array(of_vec)
        # plt.figure()
        # plt.plot(xvals, of_vec)
        # plt.show()

        of_amp = np.real( np.sum( np.conj(st)*vt/J ) / np.sum(np.abs(st)**2/J) )
        #of_amp_diff = np.real( np.sum( np.conj(st)*vt_diff/J ) / np.sum(np.abs(st)**2/J) )
        of_amp_list.append(of_amp)

        # plt.figure()
        # plt.plot(  truncdata, '.' )
        # dl, dh = (drive_freq-1.)/(Fs/2.), (drive_freq+1.)/(Fs/2.)
        # b2,a2 = signal.butter(1,[dl,dh], btype='bandpass')
        # tf = signal.filtfilt(b2,a2,truncdata)
        # plt.plot(  tf, 'c.' )
        # plt.plot( cdrive * of_amp, 'r' )
        # plt.show()

        binned_tracesf.append(btracef)
        binned_tracesr.append(btracer)
        binned_traces.append(btrace)

        binned_errorsf.append(cerrf)
        binned_errorsr.append(cerrr)
        binned_errors.append(cerr)

        ## Add to the PSDs
        bw_fac = 1. ##2.0/(len(cfft)*Fs)
        if( len(tot_psdi) == 0 ):
            p1,f1 = mlab.psd(  truncdata, NFFT=len(truncdata), Fs=Fs )
            tot_psdi = p1 * conv_fac**2
            #tot_psdi = bw_fac*cfft * cfft.conj() * conv_fac**2
            tot_psdf = bw_fac*fft_filt * fft_filt.conj() * conv_fac**2
        else:
            p2,f2 = mlab.psd(  truncdata, NFFT=len(truncdata), Fs=Fs )
            tot_psdi += p2*conv_fac**2
            #tot_psdi += bw_fac * cfft * cfft.conj() * conv_fac**2
            tot_psdf += bw_fac * fft_filt * fft_filt.conj() * conv_fac**2

        fhand.close()

    binned_tracesf = np.array(binned_tracesf)
    binned_errorsf = np.array(binned_errorsf)
    binned_tracesr = np.array(binned_tracesr)
    binned_errorsr = np.array(binned_errorsr)
    binned_traces = np.array(binned_traces)
    binned_errors = np.array(binned_errors)
    of_amp_list = np.array(of_amp_list)

    avsf = np.mean(binned_tracesf, axis=0)
    ersf = np.sqrt(np.sum(binned_errorsf**2, axis=0) \
                   / np.shape(binned_errorsf)[0])
    avsr = np.mean(binned_tracesr, axis=0)
    ersr = np.sqrt(np.sum(binned_errorsr**2, axis=0) \
                   / np.shape(binned_errorsr)[0])
    avs = np.mean(binned_traces, axis=0)
    ers = np.sqrt(np.sum(binned_errors**2, axis=0) \
                   / np.shape(binned_errors)[0])

    tot_psdi = tot_psdi * (1. / ntrace)
    tot_psdf = tot_psdf * (1. / ntrace)

    return binsf, binsr, avsf, avsr, ersf, ersr, freqs, tot_psdi, tot_psdf, avs, ers, bins, of_amp_list
=======
        physmin = get_pos_from_mon(monmax, pos_at_10V)
        physmax = get_pos_from_mon(monmin, pos_at_10V)

        ## now bin the data, separating into forward and backward
        for col in ['x','y','z']:
            for mask, sdir in zip([posmask,negmask,allmask],['pos','neg','both']):

                btrace, cerr, bins = bin( cpos[mask], truncdata_dict[col][mask]*conv_fac, 
                                          binmin=physmin, binmax=physmax, n=nbins)
                cname = 'binned_dat_' + col + '_' + sdir

                if( sdir == 'both' ):
                    cpsd,cfreq = mlab.psd(  truncdata_dict[col], 
                                            NFFT=len(truncdata_dict[col]), Fs=Fs )
                    cpsd *= conv_fac**2

                    vt = np.fft.rfft( truncdata_dict[col][mask] )
                    of_amp = np.real( np.sum( np.conj(st)*vt/J ) / norm_fac )

                else:
                    cpsd = []
                    of_amp = 0.

                if( not cname in out_dict ):
                    out_dict[ cname ] = [[btrace,], [cerr,], bins, [of_amp,],[cpsd,]]
                else:
                    out_dict[ cname ][0].append(btrace)
                    out_dict[ cname ][1].append(cerr)
                    out_dict[ cname ][3].append(of_amp)
                    out_dict[ cname ][4].append(cpsd)

    ## we've now looped through all the files, so average everything down
    for col in ['x','y','z']:
        for sdir in ['pos','neg','both']:
            cname = 'binned_dat_' + col + '_' + sdir

            bavg = np.median( np.array(out_dict[cname][0]), axis=0)
            berr = np.sqrt( np.sum(np.array(out_dict[cname][1])**2,axis=0)/len(out_dict[cname][1]) )
            
            tot_psd = np.sqrt( np.sum( out_dict[cname][4],axis=0)/len( out_dict[ cname ][4] ) )

            out_dict[cname + "_avg"] = [out_dict[cname][2], bavg, berr, tot_psd]
    
            ## zero out individual PSDs at this point to minimize file size
            if( sdir == 'both' ):
                out_dict[ cname ][4] = []

    out_dict['freq_list'] = cfreq
                
    return out_dict

## function to fit data vs position
def ffn(x,A,B):
    #return A * (1./(x+pos_offset))**2 + B
    #return A * (1./(x+50.))**2 + B
    return A * (1./(x+pos_offset))**1 + B

## function to fit force vs voltage
def ffn2(x,A,x0,C):
    return A * (x-x0)**2 + C

## don't correct for the gain, when getting the file name
def get_dc_offset_orig(s):  
    dcstr = re.findall("-?\d+mVdc", s)
    if( len(dcstr) == 0 ):
        return -999999
    else:
        curr_str = int(dcstr[0][:-4])
        if( "neg" in s and curr_str > 0):
            curr_str *= -1
        return curr_str
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959

def get_dc_offset(s):
    dcstr = re.findall("-?\d+mVdc", s)
    if( len(dcstr) == 0 ):
        return -999999
    else:
        curr_str = int(dcstr[0][:-4])
        if( "neg" in s and curr_str > 0):
            curr_str *= -1
        if( abs(curr_str) > 500 ):
            curr_str = int(curr_str/200.)
        return curr_str

data = []
# dir, label, drive_column, numharmonics, monmin, monmax
#  process_files(data_dir, num_files, numharmonics, monmin, monmax,
#                   drive_indx=19):
last_pos_at_10V = 0

for cdir in dirs:

    data_file_path = cdir[0].replace("/data/","/home/arider/analysis/")
    ## make directory if it doesn't exist
    if(not os.path.isdir(data_file_path) ):
        os.makedirs(data_file_path)
    proc_file = os.path.join( data_file_path, "cant_force_vs_position.pkl" )
    file_exists = os.path.isfile( proc_file ) and not force_remake_file


    ## first get a list of all the dc offsets in the directory
    #print cdir
    clist = glob.glob( os.path.join( cdir[0], "*.h5") )
    dc_list = []
    dc_list_orig = []
    for cf in clist:
        dcoffset = get_dc_offset( cf )
        dcoffset_orig = get_dc_offset_orig( cf )
        dc_list.append( dcoffset  )
        dc_list_orig.append( dcoffset_orig  )
    dc_list, idx = np.unique(dc_list, return_index=True)
    dc_list_orig = np.array(dc_list_orig)[idx]
    print "List of dc offsets: ", dc_list


    if(not file_exists):

        curr_data = []
        for dc_val,dc_val_orig in zip(dc_list,dc_list_orig):
            print dc_val

<<<<<<< HEAD
            #binsf, binsr, avsf, avsr, ersf, ersr, freqs, psdi, psdf, avs, ers, bins, of_amp_list = \
            #        process_files(cdir[0], max_files, cdir[3], cdir[4], cdir[5], drive_indx=cdir[2], dc_val=dc_val,pos_at_10V=cdir[6],conv_fac=cdir[7])

            curr_dict = process_files(cdir[0],max_files,drive_indx=cdir[2],dc_val=dc_val,
=======
            curr_dict = process_files(cdir[0],max_files,drive_indx=cdir[2],dc_val=dc_val_orig,
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959
                                      pos_at_10V=cdir[6],conv_fac=cdir[7])

            if( dc_val > -999999 and True):
                clab = str(dc_val) + " mV DC"
            else:
                clab = cdir[1]
            
            curr_dict['label'] = clab

            curr_data.append( curr_dict )

<<<<<<< HEAD

        out_file = open( proc_file, 'wb')
        pickle.dump(curr_data, out_file)
        out_file.close()
    else:
        print "Loading previously processed data from: %s" % proc_file
        curr_data = pickle.load( out_file )

    data += curr_data
=======

        out_file = open( proc_file, 'wb')
        pickle.dump(curr_data, out_file)
        out_file.close()
    else:
        print "Loading previously processed data from: %s" % proc_file
        out_file = open( proc_file, 'rb')
        curr_data = pickle.load( out_file )
        out_file.close()
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959

    data += curr_data

<<<<<<< HEAD
## power spectra
plt.figure(1)
for i,d in enumerate(data):
    plt.subplot(3,1,1)
    label = d['label']
    plt.loglog(d['binned_dat_x_both_avg'][4],d['binned_dat_x_both_avg'][3], label=label, color=colors_yeay[i])
    plt.subplot(3,1,2)
    plt.loglog(d['binned_dat_y_both_avg'][4],d['binned_dat_y_both_avg'][3], color=colors_yeay[i])
    plt.subplot(3,1,3)
    plt.loglog(d['binned_dat_z_both_avg'][4],d['binned_dat_z_both_avg'][3], color=colors_yeay[i])
    plt.xlim([1,100])
    plt.ylabel("Force PSD [N/rtHz]")   
    plt.xlabel("Freq [Hz]")
   
plt.legend(loc=0)

plt.show()

of_fig = plt.figure()
for i in range(len(data)):
    label = data[i]['label']
    of_amps = data[i]['binned_dat_y_both'][3]

    if( len(of_amps) > 2 and make_opt_filt_plot):
        bu.make_histo_vs_time( range(len(of_amps)), of_amps,lab=label,col=colors_yeay[i] )
    else:
        plt.plot(range(len(of_amps)), of_amps, 'o-', label=label, color=colors_yeay[i])
        

    ## make a sideways histogram
plt.ylabel('beta value')
plt.legend(loc=0)

plt.figure(2)
g = plt.gcf()
plot = plt.subplot(111)
plt.ylabel("Beta")
#ax1.set_xlabel('file number')
#ax2.set_ylabel('beta value')
plot.tick_params(axis='both', labelsize=16)
plt.show()
## function to fit data vs position
def ffn(x,A,B):
    #return A * (1./x)**2 + B
    return A * (1./(x+50.))**2 + B
    ##return A * (1./(x+15.))**1 + B
=======
## make total list of all dc offsets
def get_dcvolt_from_label(label):
    try:
        dc_volt = float(label[:-5])/1000.
    except ValueError:
        dc_volt = 0.
    return dc_volt

tot_dc_list = []
for d in data:
    tot_dc_list.append( get_dcvolt_from_label( d['label'] ) )
tot_dc_list = np.unique(tot_dc_list)

## make color list for given number of files
colors_yeay = bu.get_color_map( len(data) )
colors_dc = bu.get_color_map( len(tot_dc_list) )

## power spectra
if( plot_psds ):
    plt.figure(1)
    xlims = [1,100]
    for i,d in enumerate(data):
        label = d['label']
        for j,v in enumerate(dirs_to_plot):
            plt.subplot(len(dirs_to_plot),1,j+1)
            cxdat = d['freq_list']
            cydat = d['binned_dat_'+v+'_both_avg'][3]

            plt.loglog(cxdat, cydat, label=label, color=colors_yeay[i])
            plt.ylabel(v+" PSD [N/rtHz]")   
            plt.xlim(xlims)

    plt.xlabel("Freq [Hz]")   
    plt.legend(loc=0)

## Optimal filter amplitudes
of_fig = plt.figure(11)
ax_list = []
for i,d in enumerate(data):
    label = d['label']

    for j,v in enumerate(dirs_to_plot):
        if( i == 0 ):
            plt.subplot(len(dirs_to_plot),1,j+1)
            iax1 = plt.gca()
            cax_pos_arr = plt.gca().get_position().splitx(0.7)
            iax1.set_position(cax_pos_arr[0])
            iax2 = plt.axes(cax_pos_arr[1])
            iax2.yaxis.set_visible(False)
            ax_list.append([iax1,iax2])

        of_amps = d['binned_dat_'+v+'_both'][3]

        if( len(of_amps) > 2 and make_opt_filt_plot):
            is_bot = False
            if( v == 'y' ): is_bot = True
            bu.make_histo_vs_time( range(len(of_amps)), of_amps,lab=label,col=colors_yeay[i],axs=ax_list[j],isbot=is_bot )
        else:
            plt.plot(range(len(of_amps)), of_amps, 'o-', label=label, color=colors_yeay[i])


        ## also find the direction that maximizes the signal amplitude

def set_max( ax_list ):
    max_val = 0
    for a in ax_list:
        for b in a:
            cmax = np.max( np.abs( b.get_ylim() ) )
            if cmax > max_val:
                max_val = cmax
    for a in ax_list:
        for b in a:
            b.set_ylim([-max_val,max_val])
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959

## now maximize the axes to be equal everywhere
set_max( ax_list )

 
mag_list = []
data_vs_volt = []
<<<<<<< HEAD
for i in range(len(data)):
    #label = dirs[i][1]
    #print data[i][0], data[i][1], data[i][2]
    label = data[i]['label']
    if( sep_forward_backward ):
                plt.errorbar(data[i]['binned_dat_y_pos_avg'][0], data[i]['binned_dat_y_pos_avg'][1], data[i]['binned_dat_y_pos_avg'][2], fmt='o-', label=label, color=colors_yeay[i])
                plt.errorbar(data[i]['binned_dat_y_neg_avg'][0], data[i]['binned_dat_y_neg_avg'][1], data[i]['binned_dat_y_neg_avg'][2], fmt='o-', label=label, color=colors_yeay[i])
    
    else:
        plt.errorbar(data[i]['binned_dat_y_neg_avg'][0], data[i]['binned_dat_y_neg_avg'][1], data[i]['binned_dat_y_neg_avg'][2], fmt='o-', label=label, color=colors_yeay[i])
    
    #data_vs_volt.append( [float(label[:-5])/1000., data[i][10][gpts], data[i][11][gpts]] )
    ## fit 1/r^2 to the dipole response
    if( do_poly_fit ):
        if(sep_forward_backward):
            xdat, ydat = data[i][0][gpts], data[i][2][gpts]
        else:
            xdat, ydat = data[i][10][gpts], data[i][11][gpts]
        A, Aerr = opt.curve_fit( ffn, xdat, ydat, p0=[1.,0] )

        try:
            dc_volt = float(label[:-5])/1000.
        except ValueError:
            dc_volt = 0.
        
        mag_list.append([dc_volt,A[0],np.sqrt(Aerr[0,0])])
        xx = np.linspace( np.min(xdat), np.max(xdat), 1e3 )
        plt.plot( xx, ffn(xx,A[0],A[1]), color=colors_yeay[i], linewidth=1.5)

        print "Fit to %.2fV: A[0]=%e, A[1]=%e"%(dc_volt, A[0], A[1]) 

plt.xlabel('Distance From Bead [um]', fontsize='16')
if( do_mean_subtract ):
    plt.ylabel('Force [N]', fontsize='16')
else:
    plt.ylabel('Force[N]', fontsize='16')
plt.title(plot_title, fontweight='bold', fontsize='16', y=1.05)
#plt.xlim(30,110)
plt.legend(loc=0, numpoints=1)

g.set_size_inches(8,6)
#plt.savefig('force-v-date.pdf')
#plt.ylim(-1.4e-15, 1e-15)
#plt.savefig('force-v-pos_multipressure2.pdf')
=======
ax_list2 = []
tot_vdat, tot_bdat, tot_fdat = [],[],[]
sub_dat = []
ofig = plt.figure(111)
if( do_poly_fit ):
    pfig = plt.figure(222)

old_offsets = {}
for i,d in enumerate(data):
    label = d['label']
    curr_dat = []
    for j,v in enumerate(dirs_to_plot):
        plt.figure(ofig.number)
        plt.subplot(len(dirs_to_plot),1,j+1)
        if( i == 0 ):
            ax_list2.append( [plt.gca(),] )
        if( sep_forward_backward ):
            cd = d['binned_dat_'+v+'_pos_avg']
            bins, dat, err = cd[0], cd[1], cd[2]
            gpts = dat != 0
            plt.errorbar(bins[gpts], dat[gpts], err[gpts], fmt='o-',label=label, color=colors_yeay[i])

            cd = d['binned_dat_'+v+'_neg_avg']
            bins, dat, err = cd[0], cd[1], cd[2]
            gpts = dat != 0
            plt.errorbar(bins[gpts], dat[gpts], err[gpts], fmt='s-',label=label, color=colors_yeay[i])
            if( do_poly_fit ):
                print "Poly fit requires not to separate forward and back, skipping"
        else:
            ## if there are previous files in this region, make
            ## sure to match the mean in the overlap region
            cd = d['binned_dat_'+v+'_both_avg']
            bins, dat, err = cd[0], cd[1], cd[2]

            is_first_pos = True
            offset = 0.
            curr_dcv = get_dcvolt_from_label( d['label'] )
            if( match_overlap_region and do_mean_subtract):
                mean_list = []
                for iprev, dprev in enumerate( data[:i] ):
                    od = dprev['binned_dat_'+v+'_both_avg']
                    old_dcv = get_dcvolt_from_label( dprev['label'] )                    
                    if( np.abs(curr_dcv - old_dcv) > 1e-5 ): continue
                    old_bins, old_dat = od[0], od[1]
                    if ( np.abs(np.min(old_bins) - np.min(bins))<5 ): continue
                    
                    match_list = []
                    match_list_new = []
                    for bb,dd in zip(bins,dat):
                        if( bb >= np.min( old_bins ) and bb <= np.max( old_bins ) and abs(dd) > 0 ):
                            match_list_new.append(dd)
                            is_first_pos = False
                    for bb,dd in zip(old_bins,old_dat):
                        if( bb >= np.min( bins ) and bb <= np.max( bins ) and abs(dd) > 0):
                            match_list.append(dd)
                            is_first_pos = False
                    if( len(match_list) > 0 ):
                        mean_list.append( np.mean( match_list_new ) - np.mean(match_list) )

                min_idx = int( np.min( bins ) )
                curr_key = str(min_idx)+"_"+v+"_"+str(curr_dcv*1000)
                if( is_first_pos ):
                    #if( min_idx < 160 ):
                    #    raw_input('e')
                    curr_offset = -dat[-1]
                    if( curr_key in old_offsets ):
                        old_offsets[curr_key].append( curr_offset )
                    else:
                        old_offsets[curr_key] = [curr_offset,]

                tot_cum_offset = 0.
                if( len(mean_list) > 0):
                    curr_offset = -np.mean( mean_list )
                    if( curr_key in old_offsets ):
                        old_offsets[curr_key].append( curr_offset )
                    else:
                        old_offsets[curr_key] = [curr_offset,]

                    ## find the sum of all offsets below this
                    for k in old_offsets.keys():
                        cvals = k.split( "_" )
                        kval = cvals[0]
                        dirval = cvals[1]
                        dcval = cvals[2]
                        if( int(kval) > min_idx 
                            and dirval == v 
                            and np.abs(float(dcval) - curr_dcv*1000)< 1):
                            
                            tot_cum_offset += np.mean(old_offsets[k])
                #if(is_first_pos): print "first: ", curr_key
                offset = curr_offset + tot_cum_offset

            gpts = dat != 0
            if( len(tot_dc_list) > 1 ): 
                coll = colors_dc[ np.argwhere( tot_dc_list == curr_dcv)[0] ]
            else:
                coll = colors_yeay[i]
            
            plt.errorbar(bins[gpts], dat[gpts]+offset, err[gpts], fmt='o-',label=label, color=coll)
            curr_dat.append(dat[gpts])

            if( do_poly_fit and v == 'y'):
                plt.figure(pfig.number)
                plt.errorbar(bins[gpts], dat[gpts]+offset, err[gpts], fmt='o-',label=label, color=coll)
                dc_volt = get_dcvolt_from_label(label)
                xdat, ydat = bins[gpts], dat[gpts]
                tot_vdat.append( dc_volt*np.ones_like(xdat) )
                tot_bdat.append( xdat )
                A, Aerr = opt.curve_fit( ffn, xdat, ydat, p0=[1.,0] )
                mag_list.append([dc_volt,A[0],np.sqrt(Aerr[0,0])])
                tot_fdat.append( ydat - ffn(np.array(bins[gpts])[-1],*A) )
                
                xx = np.linspace( np.min(xdat), np.max(xdat), 1e3 )
                plt.plot( xx, ffn(xx,A[0],A[1]), color=coll, linewidth=1.5)
                fval = ffn(20.,A[0],0)
                print "Fit to %.2fV: A[0]=%e, A[1]=%e, Force[20 um]=%e"%(dc_volt, A[0], A[1],fval)

    sub_dat.append(curr_dat)

set_max( ax_list2 ) 

if( False ):

    sub_dat = np.array(sub_dat)
    plt.close('all')
    plt.figure()
    plt.plot( bins[gpts], sub_dat[5,1,:], 's-', color='k' )
    for j,scale_fac in enumerate(np.arange(-18, -8, 1)):
        for i in range( np.shape(sub_dat)[0] ):
        
            # if(i==0):
            #     plt.plot( bins[gpts], sub_dat[i,1,:]-scale_fac*sub_dat[i,2,:], 's-', color=colors_yeay[j], label=str(scale_fac) )
            # else:
            #     plt.plot( bins[gpts], sub_dat[i,1,:]-scale_fac*sub_dat[i,2,:], 's-', color=colors_yeay[j])
            yvals = sub_dat[i,1,:]-sub_dat[-1,1,:]
            plt.plot( bins[gpts], yvals-yvals[-1], 's-', color=colors_yeay[i] )

    plt.legend()
    plt.show()

def find_volt_for_beta(beta, pos, volts, force ):
    force = np.array(force)
    v_mesh = np.linspace(0,10,1e3)
    pos_list = get_pos_from_mon(v_mesh, pos_offset )

    ## now get the chameleon force at this beta
    cham_force = bu.get_chameleon_force( pos_list * 1e-6 ) * beta

    ## now invert the measured data to get the voltage
    #force_2d_dat = interp.RectBivariateSpline(pos[0],force,volts)

    #out_volts = np.zeros_like( v_mesh )
    
    # out_volts = interp.griddata( np.column_stack([pos.flatten(),np.array(force).flatten()]),
    #                              volts.flatten(),
    #                              np.column_stack([pos_list,cham_force]))

    ## step through each position in the input array, interpolate
    pos_volt = []
    for i in range( np.shape(pos)[1] ):
        cpos = pos[0,i]
        cforce = force[:,i]
        cvolts = volts[:,i]
        
        pos_volt.append( [cpos, np.interp( bu.get_chameleon_force( cpos*1e-6 ) * beta, 
                                           cforce, cvolts )] )
    pos_volt = np.array(pos_volt)

    pos_volt_fit = np.polyfit( pos_volt[:,0], pos_volt[:,1], 4 )
    out_vals = np.polyval( pos_volt_fit, pos_list )

    # plt.close('all')
    # plt.figure()
    # #plt.plot( v_mesh, cham_force )
    # plt.plot( pos_volt[:,0], pos_volt[:,1], 'b.')
    # plt.plot( pos_list, out_vals)

    # plt.figure()
    # plt.plot( v_mesh,  out_vals)
    # plt.show()

    return np.column_stack([v_mesh, out_vals])

if( do_2d_fit ):
    ## fit the total data set
    tot_xdat, tot_ydat, tot_zdat = np.array(tot_bdat), np.array(tot_vdat), np.array(tot_fdat)
    
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter( tot_xdat, tot_ydat, tot_zdat, c=tot_zdat )
    #plt.colorbar()

    # ## now figure out the voltages vs position that we need for a few 
    # ## values of beta
    # beta_list = [3e8,]

    # for b in beta_list:
    #     curr_volt = find_volt_for_beta( b, tot_xdat, tot_ydat, tot_fdat ) 
        
    #     np.savetxt( data_file_path + "/cham_force_beta_%.0e.txt"%b, curr_volt, delimiter=',' )
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959

if( do_poly_fit ):
    mag_list = np.array(mag_list)
    fit_fig = plt.figure()
    plt.errorbar( mag_list[:,0], mag_list[:,1], yerr=mag_list[:,2], fmt='ko' )
    A, Aerr = opt.curve_fit( ffn2, mag_list[:,0], mag_list[:,1], p0=[1.,0,0] )    
    xx = np.linspace( np.min(mag_list[:,0]), np.max(mag_list[:,0]), 1e3 )
    plt.plot(xx, ffn2(xx, A[0], A[1], A[2]), 'r', linewidth=1.5)

    plt.xlabel("Cantilever DC bias [V]")
    plt.ylabel("Force from fit at 1um [N]")

<<<<<<< HEAD

#if( do_2d_fit ):

=======
>>>>>>> d87af76a3f9ea33789e9ad3ce281174f5b4e0959
plt.show()
