import cant_utils as cu
import numpy as np
import matplotlib.pyplot as plt
import glob 
import bead_util as bu
import Tkinter
import tkFileDialog
import os, sys
from scipy.optimize import curve_fit
import bead_util as bu
from scipy.optimize import minimize_scalar as minimize 

dirs = [3,4,5,6,7,8]
cal = 5.0e-14

ddict = bu.load_dir_file( "/home/charles/opt_lev_classy/scripts/cant_force/dir_file.txt" )

load_from_file = False

colors_yeay = bu.get_color_map( len(dirs) )
i = 0

def proc_dir(d):
    global i
    dv = ddict[d]

    dir_obj = cu.Data_dir(dv[0], [0,0,dv[-2]])
    dir_obj.avg_force_v_p()
    x, y, errs = dir_obj.ave_force_vs_pos['0.0']

    #for direction in [0,1,2]:
    #    plt.subplot(3,1,1+direction)
    plt.errorbar(x, y*dv[-1], errs*dv[-1], fmt='o-', label=dv[1], color=colors_yeay[i])

    i += 1
    
plt.figure()
dir_objs = map(proc_dir, dirs)

plt.xlabel('Cantilever Position [um]')
plt.legend()
plt.show()
