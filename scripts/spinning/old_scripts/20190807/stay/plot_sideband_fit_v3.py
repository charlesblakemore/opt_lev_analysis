import numpy as np
import matplotlib.pyplot as plt
import os
import re

from scipy.optimize import curve_fit
save = False 
no_offset = False 

if no_offset:
	name = 'no_offset'
else:
	name = 'offset'
 
rho = 1.55e3 #kg/m^3
m = 85.0e-15 #kg
#m = 88.5e-15 #kg



I = (0.4) * m * (0.75 * (m/(np.pi*rho)))**(0.6666666)
print(I)
def sqrt(E, d, c):
	func = np.sqrt(E*d) + c

	if no_offset:
		func = np.sqrt(E*d)
	return func


#path = "/processed_data/spinning/wobble/20190626/slow/"
path = "/processed_data/spinning/wobble/20190626/"
#out_path = "/home/dmartin/analyzedData/20190626/wobble/"
#path = "/processed_data/spinning/wobble/20190514/"
split = re.split(r'\W+', path)
 
meas_names = []
for root, dirs, filenames in os.walk(path):
	
	for i, filename in enumerate(filenames):
		data = np.load(path + filename)
		print(data[0])	
		
		E = data[0] * 1e-3 #kV/m
		wobble_freq = 2 * np.pi * data[2] #rad*Hz
		
		popt, pcov = curve_fit(sqrt, E, wobble_freq) 
		
		dipole = popt[0] *1e-3* I * (1/1.602e-19) * 1e6 #e * um

		 
		if no_offset:
			plt.plot(E ,sqrt(E,*popt),label="meas {}, {} $e\cdot\mu m$".format(i+1,round(dipole,3)))
			plt.scatter(E,wobble_freq)
		else:
			plt.plot(E ,sqrt(E,*popt),label="meas {}, {} $e\cdot\mu m$, offset = {} rad Hz".format(i+1,round(dipole,3),round(popt[1],3)))		
			plt.scatter(E,wobble_freq)

plt.xlabel("E [kV/m]")
plt.ylabel("$\omega_{\phi}$")
plt.legend()
if save:
	plt.savefig(out_path + 'wobble_meas_20190626_{}_{}.png'.format(split[-2],name),dpi = 300)

plt.show()
