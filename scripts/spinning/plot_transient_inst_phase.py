from hs_digitizer import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import window_func as window
import bead_util as bu
from amp_ramp_3 import bp_filt, lp_filt, hp_filt
from scipy import signal
from transfer_func_util import damped_osc_amp
from scipy.optimize import curve_fit
from memory_profiler import profile
from memory_profiler import memory_usage
from plot_phase_vs_pressure_many_gases import build_full_pressure


import gc

mpl.rcParams['figure.figsize'] = [7,5]
mpl.rcParams['figure.dpi'] = 150

directory = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_2/base_press/change_phi_offset_redo/'
directory = '/home/dmartin/Desktop/analyzedData/20200130/spinning/series_3/base_press/change_phi_offset/'

files, zeros = bu.find_all_fnames(directory, ext='.npz')

damp_avg = []
damp_std = []
dg_arr = []
for i, f in enumerate(files):
    print(files[i])
    data = np.load(files[i], allow_pickle=True)
    
    fit_params = data['fit_params']
    dg_arr.append(data['dg'])
    
    print(fit_params)
    avg = np.mean(fit_params[:], axis=0)
    std = np.std(fit_params[:], axis=0)

    damp_avg.append(-avg[2])
    damp_std.append(std[2])    
    
plt.errorbar(dg_arr,damp_avg, yerr=damp_std, fmt='o')
plt.yscale('log')
plt.ylabel(r'$\gamma$ [Hz]')
plt.xlabel('dg scale factor [arb.]')
plt.grid(b=True, which='minor', axis='both')
plt.grid(b=True ,which='major', axis='both')
plt.show()






