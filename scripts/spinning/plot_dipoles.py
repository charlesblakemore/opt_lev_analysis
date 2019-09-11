import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu

path = '/data/old_trap_processed/spinning/wobble/20190905/before_pramp/dipoles/' 
#path = '/home/cblakemore/processed_data_s/spinning/wobble/20190905/before_pramp/dipoles/' 

files, zero = bu.find_all_fnames(path, ext = '.dipole')

dipole_arr = []
dipole_err_arr = []
times = []

for i, fil in enumerate(files):
	arr = np.load(fil)
	dipole_arr.append(arr[0])
	dipole_err_arr.append(arr[1])	

plt.plot(dipole_arr)

x = np.arange(0, len(dipole_arr))

plt.errorbar(x, dipole_arr, yerr = dipole_err_arr, fmt = 'o')
plt.show()
