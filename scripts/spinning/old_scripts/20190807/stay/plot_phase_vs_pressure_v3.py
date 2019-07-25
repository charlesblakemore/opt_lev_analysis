import numpy as np
import matplotlib.pyplot as plt
from piecewise_line import *
from scipy.optimize import curve_fit
import matplotlib
import re

base_path = "/home/dmartin/analyzedData/20190514/pramp3/N2/N2_4Vpp_50kHz_1/"
in_fs = ['N2_4Vpp_50kHz_1_']#, 'He_4Vpp_50kHz_3_','He_4Vpp_50kHz_4_']#,'Kr_4Vpp_50kHz_2_','N2_4Vpp_50kHz_1_',\
		 #'N2_4Vpp_50kHz_2_','Ar_4Vpp_50kHz_2_']

#base_path = "/home/dmartin/analyzedData/20190514/"
#in_fs = ['N2_4Vpp_50kHz_1_']

regex = re.compile(r'\w\w')
gases = []

for i in range(len(in_fs)):
	gas = regex.match(in_fs[i])
	gases.append(gas.group())

cal = 0.66

def get_phi(fname):
    phi = np.load(base_path + fname + "phi.npy")
    return phi/2

def get_pressure(fname):
    pressures = np.load(base_path + fname + "pressures.npy")
    return pressures

def pressure_model(pressures, break_ind = 885 , p_ind = 1, plt_press = True):
    ffun = lambda x, y0, m0, m1,m2: \
            pw_line(x, break_ind, 1e8, 1e7, y0, m0, m1, m2, 0.)

    n = np.shape(pressures)[0]
    inds = np.arange(n)
    popt, pcov = curve_fit(ffun, inds, pressures[:, p_ind])
    pfit = ffun(inds, *popt)

    if plt_press:
        p_dict = {0:"Pirani",1:"1 Torr Baratron", -1:"Baratron"}
        plt.plot(1.33322*pressures[:, p_ind], '-', label = p_dict[p_ind])
        plt.plot(1.33322*pfit, label = "piecewise linear fit")
        plt.xlabel("File [#]")
        plt.ylabel("Pressure [mbar]")
        plt.legend()
        plt.show()

    return pfit

def constline(x,m,b):
	return m*x + b
def phi_ffun(p, k, phinot):
    return -1*np.arcsin(np.clip(p/k, 0., 1)) + phinot

phases = np.array(map(get_phi, in_fs))
pressures = np.array(map(get_pressure, in_fs))
p_fits = np.array(map(pressure_model, pressures))

p_fits *= 1.33322
phases = -1 * phases

upper_bound =  0.08

#Finds where the phase lag becomes random with pressure. Skip will bypass any issues with unwrapping in the phase
def	find_random_phase(phases,phase_diff_max, skip): 
	
	for i in range(len(phases[skip:])):
		#takes diff between phase at at index i+skip and 1 index away
		phase_diff = phases[i+skip] - phases[i+skip+1]
		
		if phase_diff > phase_diff_max:
			phase_diff_max = phase_diff
			max_ind = i+skip 
			break
	
	return max_ind


max_diff_ind = find_random_phase(phases[0],0.5,350)
max_diff_ind -= 10

plt.scatter(p_fits[0][max_diff_ind],phases[0][max_diff_ind])
plt.plot(p_fits[0],phases[0],'r')
plt.show()

print(p_fits[0][max_diff_ind])

p_new_interval = p_fits[0]<p_fits[0][max_diff_ind]

unwrap_cut = [p_fits[0][max_diff_ind],0]
for i in range(len(p_fits)):
	mask = p_fits[i] < unwrap_cut[i]
	phases[i][mask] = np.unwrap(2.0*phases[i][mask])/2.0
	#phases[i] += np.pi
	phases[i] = np.unwrap(phases[i])

popts = []
pcovs = []

plt.scatter(p_fits[0],phases[0])
plt.show()
for i, p in enumerate(p_fits):
    p0 = [p_fits[0][max_diff_ind], 0.0]
    pphi, covphi = curve_fit(phi_ffun, p_fits[i][p_new_interval], phases[i][p_new_interval], p0 = p0)
    popts.append(pphi)
    pcovs.append(covphi)

plt.plot(p_fits[0],phi_ffun(p_fits[0],*popts[0]),'r')
plt.scatter(p_fits[0],phases[0])
plt.show()

plt.plot(p_fits[0],(phi_ffun(p_fits[0],*popts[0])-popts[0][1])/np.pi,'r')

plt.scatter(p_fits[0][p_new_interval],(phases[0][p_new_interval]-popts[0][1])/np.pi,alpha=0.25)
	
print(popts[0][0])
plt.show()
