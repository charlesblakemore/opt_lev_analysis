import numpy as np
import matplotlib.pyplot as plt
import cant_utils as cu

dat_dir = "/data/20160429/nobead/ysweepoverz_75_65"
load = True

dobj = cu.Data_dir(dat_dir, [0, 0, 0])
lfun = lambda fname, sep: cu.pos_loader(fname, sep, cant_axis = 1)



if load == True:
    dobj.load_dir(lfun)
    dobj.save_dir()

else:
    dobj.load_from_file()

labeler = lambda fobj: str(80 - np.mean(fobj.binned_cant_data[1, 2]))

def plt_sweeps(dobj, axis = 1, ecal = 5e-14):
    #plots all of the binned data for axis.
    for fobj in dobj.fobjs:
        plt.errorbar(fobj.binned_cant_data[1, 1], fobj.binned_pos_data[1, 1]*ecal, fobj.binned_data_errors[1, 1]*ecal, fmt = 'o-', label = labeler(fobj))
    
    plt.xlabel("Cantilever x position")
    plt.ylabel("Apparent z background force")
    plt.legend()
    plt.show()

