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

dirs = [13,14,15,16]
cal = 5.0e-14

ddict = bu.load_dir_file( "/home/charles/opt_lev_analysis/scripts/dirfiles/dir_file_june2017.txt" )

load_from_file = False

new_obj = cu.Data_dir('shit_path', [0,0,0], "wheeeee")

new_obj.load_H("optphase2_Hout.p")
#new_obj.plot_H(phase=True, label=True)

Hmat = new_obj.build_avgH(fthresh = 60)

plt.imshow(np.abs(Hmat))
plt.show()

Hmat_diag = np.linalg.inv(Hmat)

plt.imshow(np.abs(Hmat_diag))
plt.show()

print np.abs(Hmat)

print np.abs(Hmat_diag)

print np.abs(np.einsum('ij,jk -> ik', Hmat, Hmat_diag))
