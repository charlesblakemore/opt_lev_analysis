import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.signal as ss
from scipy.optimize import curve_fit
from scipy import constants

path = "/data/20181129/bead1/weigh/high_pressure_0.5Hz_4pp"

files = bu.find_all_fnames(path)
g_monitor = 100.
s_electrode = 0.004 
elec_top = 1
elec_bottom = 2
power_column = 4
q = 25*constants.e

def sin_fun(x, a, f, phi, const):
    return a*np.sin(x*2.*np.pi*f + phi) + const

def lin_fun(x, m, x0):
    return m*(x-x0)

def dec2(arr):
    return ss.decimate(ss.decimate(arr, 13), 13)

df = bu.DataFile()
df.load(files[-1])
df.load_other_data()

E = g_monitor*(df.other_data[elec_top]-df.other_data[elec_bottom])/s_electrode
optical = df.other_data[power_column]

E = dec2(E)
optical = dec2(optical)
p0 = [-1e12, 2.5e-12]
popt, pcov = curve_fit(lin_fun, q*E, optical, p0 = p0)
x_plt_fit = np.linspace(np.min(q*E), popt[-1], 100)
plt.plot(q*E, optical, 'o')
plt.plot(x_plt_fit, lin_fun(x_plt_fit, *popt), 'r', linewidth = 5, label = "m = " + str(popt[-1]/9.81))
plt.legend()
plt.xlabel("Electric Force [N]")
plt.ylabel("Optical Power [arb]")
plt.show()



plt.plot(q*E, optical-lin_fun(q*E, *popt), 'o')
#plt.plot(x_plt_fit, lin_fun(x_plt_fit, *popt), 'r', linewidth = 5, label = "m = " + str(popt[-1]/9.81))
#plt.legend()
plt.xlabel("Electric Force [N]")
plt.ylabel("residual optical Power [arb]")
plt.show()
