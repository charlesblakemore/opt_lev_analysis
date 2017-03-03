## measure the force from the cantilever
import glob, os, re
import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import scipy.optimize as sp
import matplotlib.mlab as mlab
import matplotlib.cm as cmx
import matplotlib.colors as colors

#data_dir = "/data/20150908/Bead2/cant_mod"
data_dir = "/data/20151208/bead1/freq_sweep3"

name_list = []

savefig = True

NFFT = 2**17

conv_fac = 3e-14 ## N/V

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

def get_curr_freq( s ):
    return float(re.findall( "\d+Hz", s )[0][:-2])

if( name_list ):
    flist = [os.path.join(data_dir,n[0]) for n in name_list]    
else:
    #flist = sorted(glob.glob(os.path.join(data_dir, "*xyzcool*.h5")), key = sort_fun)
    flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = get_curr_freq)

print flist

fig=plt.figure()
tot_dat = []

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

def get_curr_freq( s ):
    return re.findall( "\d+Hz", s )[0][:-2]

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
    color_idx = freq_list.index( float(fex) )
    fidx = int(re.findall("\d+.h5", f )[0][:-3])
    if( fidx==0 ):
        plt.semilogy( freqs, y, label = fex+" Hz", color=scalarMap.to_rgba(color_idx) )
    else:
        plt.semilogy( freqs, y, color=scalarMap.to_rgba(color_idx) )

    fex = attribs['electrode_settings'][19]
    freq_idx = np.argmin( np.abs( freqs - float( fex ) ) )
    plt.semilogy( freqs[freq_idx], y[freq_idx], 'o', mfc='none', markeredgecolor=scalarMap.to_rgba(color_idx) )
    freq_idx = np.argmin( np.abs( freqs - 2.0*float( fex ) ) )
    plt.semilogy( freqs[freq_idx], y[freq_idx], 's', mfc='none', markeredgecolor=scalarMap.to_rgba(color_idx) )

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={"size": 11})
plt.xlim([0, 210])
plt.ylim([1e-17, 1e-14])

plt.xlabel('Freq [Hz]')
plt.ylabel('Force [N Hz$^{-1/2}$]')

#plt.title(folder_name)

fig.set_size_inches(12,6)

if(savefig):
    plt.savefig("plots/calibrated_spec_%s.png"%folder_name)

plt.show()


