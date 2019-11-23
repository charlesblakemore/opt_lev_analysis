import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import bead_util as bu
from hs_digitizer import *
from scipy import signal
from scipy.optimize import curve_fit

base_path = '/home/dmartin/Desktop/analyzedData/20191105/phase_mod/forced_libration/no_window/'

files, zero = bu.find_all_fnames(base_path, ext='.npy')
fil = '/data/old_trap/20191105/bead4/phase_mod/deriv_feedback_1/turbombar_powfb_xyzcool_mod_no_dg_1_0.h5'

mpl.rcParams['figure.figsize'] = [7,5]
mpl.rcParams['figure.dpi'] = 150

print(files)

data_ind = 0
drive_ind = 1


obj = hsDat(fil)

Ns = obj.attribs['nsamp']
Fs = obj.attribs['fsamp']
freqs = np.fft.rfftfreq(Ns, 1./Fs)

fft = np.fft.rfft(obj.dat[:,data_ind])

z = signal.hilbert(obj.dat[:,data_ind])
phase = signal.detrend(np.unwrap(np.angle(z)))

phase_fft = np.fft.rfft(phase)

data = np.load(files[0])
data1 = np.load(files[1])
data2 = np.load(files[2])

labels = ['large frequency step', 'small frequency step 1', 'small frequency step 2']

dfs = [data,data1,data2]
def lorenztian(f, A, f0, g, c):
    w = 2*np.pi*f
    w0 = 2*np.pi*f0

    denom = ((w0**2 - w**2)**2 + (g*w)**2)**(1./2.)

    return A/denom

def fit(df):
    p0=[1e4,280,0.1,0.5]
    popt, pcov = curve_fit(lorenztian, df[0], df[1],p0=p0)
    print(popt)
    freqs = np.linspace(df[0][0],df[0][-1], 1000)
    

    plt.loglog(freqs, lorenztian(freqs, *popt))
    
    plt.scatter(df[0],df[1], label=labels[i])
    
    
    
#plt.loglog(data[0],data[1])
#plt.scatter(data2[0],data2[1])

for i in range(len(dfs)):
    fit(dfs[i])

plt.legend()
plt.ylabel('Amplitude [arb.]')
plt.xlabel('Frequency [Hz]')
plt.show()

data = np.load('/home/dmartin/Desktop/analyzedData/20191105/phase_mod/forced_libration/forced_libration_3_fine.npy')
data1 = np.load('/home/dmartin/Desktop/analyzedData/20191105/phase_mod/forced_libration/forced_libration_3.npy')
data2 = np.load('/home/dmartin/Desktop/analyzedData/20191105/phase_mod/forced_libration/forced_libration_4_fine.npy')

x = data[0]
y = data[1]

x1 = data1[0]
y1 = data1[1]

x2 = data2[0]
y2 = data2[1]

plt.scatter(x2,y2)
plt.scatter(x1,y1)
plt.scatter(x,y)
plt.show()
