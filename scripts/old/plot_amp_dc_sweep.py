## measure the force from the cantilever
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import matplotlib.mlab as mlab
import matplotlib.cm as cmx
import matplotlib.colors as colors

data_dir_list = ["/data/20151208/bead1/amp_sweep_dc_man", "/data/20151208/bead1/amp_sweep_dc_auto", "/data/20151208/bead1/amp_sweep_dc_man_2",]


name_list = []
color_list = ['k','r','g','b','c','m','y']

savefig = True

NFFT = 2**17

conv_fac = 3e-13 ## N/V

def sort_fun( s ):
    ## sort by pressure, then index
    cs = re.findall("_\d+.h5", s)
    if( len(cs) == 0 ):
        idx = 0
    else:
        idx = int(cs[0][1:-3])

    ## get pressure
    # _,ssplit = os.path.split(s)
    # cs = re.findall(".*mbar", ssplit)
    # cs = cs[0][:-4]
    # if(cs == "UR" ):
    #     press = 1e-6
    # else:
    #     cs = cs.replace("_", ".")
    #     press = float(cs)
    press = 1e-6

    ## get z_pos
    cs = re.findall("Z\d+nm", s)
    if( len(cs) == 0 ):
        zpos = 0
    else:
        zpos = int(cs[0][1:-2])

    return press*1e30 + zpos*1e10 + idx

def get_curr_freq_auto( s ):
    return float(re.findall( "\d+mVdc.h5", s )[0][:-7])

def get_curr_freq_man( s ):
    return float(re.findall( "\d+VDC", s )[0][:-3])

df_fig = plt.figure()
df2_fig = plt.figure()
for ddid,data_dir in enumerate(data_dir_list):

    use_man = "man" in data_dir
    if( use_man ):
        get_curr_freq = lambda s: get_curr_freq_man(s)
    else:
        get_curr_freq = lambda s: get_curr_freq_auto(s)

    flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = get_curr_freq)

    if( ddid == 0):
        fig=plt.figure()

    folder_name = data_dir.split("/")
    folder_name = "_".join( folder_name[2:] )

    ## make color list same length as flist
    freq_list = []
    for f in flist:
        cf = get_curr_freq(f)
        if( cf not in freq_list ):
            freq_list.append(cf)

    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=len(freq_list))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    amp_dat = []
    for i,f in enumerate(flist):

        #if( "elec" in f ): continue

        #if( len(re.findall("morning", f)) == 0): continue
        print f 

        cdat, attribs, _ = bu.getdata( f )
        _,fex = os.path.split(f)

        if( name_list ):
            cpress = name_list[i][1]
            fex = "%s mbar" % cpress

        Fs = attribs['Fsamp']

        cpsd, freqs = mlab.psd(cdat[:, 1]-np.mean(cdat[:,1]), Fs = Fs, NFFT = NFFT) 

        fex = str(get_curr_freq( f ))
        y = np.sqrt(cpsd)*conv_fac
        if( use_man ):
            fex = float(fex)
            color_idx = freq_list.index( float(fex) )
            cv = fex
        else:
            fex = attribs['electrode_settings'][27]
            color_idx = freq_list.index( float(fex)*1000 )
            cv = fex*200            

        cf_idx = re.findall("\d+.h5", f )
        if( len(cf_idx) > 0 ):
            fidx = int(cf_idx[0][:-3])
        else:
            fidx = -1
        if( ddid == 0 ):
            if( fidx==0 ):
                plt.semilogy( freqs, y, label = str(cv)+" V", color=scalarMap.to_rgba(color_idx) )
            else:
                plt.semilogy( freqs, y, color=scalarMap.to_rgba(color_idx) )

        fex = attribs['electrode_settings'][19]
        freq_idx = np.argmin( np.abs( freqs - float( fex ) ) )
        if( ddid == 0 ):
            plt.semilogy( freqs[freq_idx], y[freq_idx], 'o', mfc='none', markeredgecolor=scalarMap.to_rgba(color_idx) )
        camp_f = y[freq_idx]
        freq_idx = np.argmin( np.abs( freqs - 2.0*float( fex ) ) )
        if( ddid == 0 ):
            plt.semilogy( freqs[freq_idx], y[freq_idx], 's', mfc='none', markeredgecolor=scalarMap.to_rgba(color_idx) )
        camp_2f = y[freq_idx]

        amp_dat.append( [cv, camp_f, camp_2f] )

    if( ddid == 0 ):
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={"size": 11})
        plt.xlim([0, 210])
        plt.ylim([1e-17, 1e-13])

        plt.xlabel('Freq [Hz]')
        plt.ylabel('Force [N Hz$^{-1/2}$]')


    amp_dat = np.array(amp_dat)

    bw = np.sqrt(50.)
    plt.figure(df_fig.number)
    plt.plot( amp_dat[:,0], amp_dat[:,1]/bw, 'o-', color=color_list[ddid] )
    plt.xlabel( "DC offset voltage [V]" )
    plt.ylabel( "Response [N]/V" )
    plt.title("Response at f")

    plt.figure(df2_fig.number)
    plt.plot( amp_dat[:,0], amp_dat[:,2]/bw, 'o-', color=color_list[ddid] )
    plt.xlabel( "DC offset voltage [V]" )
    plt.ylabel( "Response [N]" )
    plt.title("Response at 2*f")

# fig.set_size_inches(12,6)

# if(savefig):
#     plt.savefig("plots/calibrated_spec_%s.png"%folder_name)

plt.show()


