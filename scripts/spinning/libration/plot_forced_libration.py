import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import bead_util as bu
from hs_digitizer import *
from scipy import signal
from scipy.optimize import curve_fit

base_path = '/home/dmartin/Desktop/analyzedData/20191105/phase_mod/forced_libration/small_amp/'

files, zero = bu.find_all_fnames(base_path, ext='.npy')
fil = '/data/old_trap/20191105/bead4/phase_mod/deriv_feedback/deriv_feedback_0/turbombar_powfb_xyzcool_mod_0.h5'

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

data = np.load(files[1])
print(files[1])
#data1 = np.load(files[1])

labels = ['']#'large frequency step', 'small frequency step 1']

dfs = [data]
def lorenztian(f, A, f0, g, c):
    w = 2*np.pi*f
    w0 = 2*np.pi*f0

    denom = ((w0**2 - w**2)**2 + (g*w)**2)**(1./2.)

    return (A/denom) + c

def fit(df):
    mask = df[0] > 310
    
    x = df[0]
    y = df[1]
    
    print(x)

    p0=[1e4,280,100,0.]
    popt, pcov = curve_fit(lorenztian, x, y,p0=p0)
    
    print(popt)
    freqs = np.linspace(x[0],x[-1], 10000)
    

    plt.plot(freqs, lorenztian(freqs, *popt))
    
    plt.scatter(x,y, label=labels[i])
    
    
    
#plt.loglog(data[0],data[1])
#plt.scatter(data2[0],data2[1])

for i in range(len(dfs)):
    fit(dfs[i])

plt.legend()
plt.ylabel('Amplitude [arb.]')
plt.xlabel('Frequency [Hz]')
plt.show()
