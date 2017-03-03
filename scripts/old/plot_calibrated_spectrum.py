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
data_dir = "/data/20160320/bead1/cant_sweep_150um_dcsweep"

# name_list = [['URmbar_xyzcool_cantiout_stageX0nmY0nmZ0nm.h5', "~5e-6"],
#              ['URmbar_xyzcool_after_lunch.h5', "~1e-5"],
#              ['7e-4mbar_xyzcool_after_lunch.h5', "7e-4"],
#              ['1e-3mbar_xyzcool_after_lunch.h5', "1e-3"],
#              ['5e-3mbar_xyzcool_after_lunch.h5', "5e-3"],
#              ['2e-2mbar_xyzcool_after_lunch.h5', "0.02"],
#              ['1_5mbar_xyzcool.h5', "1.5"]]
name_list = []

savefig = True

NFFT = 2**17

conv_fac = 2.7e-14 ## N/V

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



if( name_list ):
    flist = [os.path.join(data_dir,n[0]) for n in name_list]    
else:
    #flist = sorted(glob.glob(os.path.join(data_dir, "*xyzcool*.h5")), key = sort_fun)
    flist = sorted(glob.glob(os.path.join(data_dir, "*.h5")), key = sort_fun)

print flist

fig=plt.figure()
tot_dat = []

folder_name = data_dir.split("/")
folder_name = "_".join( folder_name[2:] )

## make color list same length as flist
jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=len(flist))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

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
    
    plt.loglog( freqs, np.sqrt(cpsd)*conv_fac, label = fex, color=scalarMap.to_rgba(i) )

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.7), prop={"size": 11})
plt.xlim([1, 2.5e3])
plt.ylim([1e-18, 5e-15])
plt.xlabel('Freq [Hz]')
plt.ylabel('Force [N Hz$^{-1/2}$]')

#plt.title(folder_name)

fig.set_size_inches(12,6)

if(savefig):
    plt.savefig("plots/calibrated_spec_%s.png"%folder_name)

plt.show()


