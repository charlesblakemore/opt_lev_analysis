import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
#Need to average masses or what is necessary now. Chas has changed the all_data.p file
save = False 
File = '/processed_data/spinning/pramp_data/20190626/all_data.p'
out_path = '/home/dmartin/analyzedData/20190626/pramp/'

data = np.load(File)

def inv_sqrt(x,a):
	return a * (1/np.sqrt(x)) 
 
mean_scaled_pmaxes = []
std_scaled_pmaxes = []

masses = []
masses_dict = {'He': 4.022,'N2': 28.014, 'Ar': 39.948, 'Kr': 83.798, 'Xe': 131.29, 'SF6': 146.06}


for gas in data.iterkeys():
	if not data[gas]['dipole']:
		continue

	dipoles = np.array(data[gas]['dipole'])
	pmaxes = np.array(data[gas]['pmax'])
	
	scaled_pmaxes = pmaxes/dipoles
	
	mean_scaled_pmaxes.append(np.mean(scaled_pmaxes))
	std_scaled_pmaxes.append(np.std(scaled_pmaxes)/len(scaled_pmaxes)) #Divide each std by number of measurements
	#masses.append(data[gas]['mass'])
	masses.append(masses_dict[gas])
	
print(len(mean_scaled_pmaxes),len(masses))
popt, pcov = curve_fit(inv_sqrt,masses,mean_scaled_pmaxes,[1],sigma=std_scaled_pmaxes)

print(popt)

a_coeff = popt[0]
a_coeff_err = np.sqrt(np.diag(pcov)[0])

mass_vec = np.linspace(1,max(masses) +20,1000)


plt.figure(figsize=(10,6), dpi = 150)

plt.plot(mass_vec,inv_sqrt(mass_vec,*popt),label=r"$\frac{E}{\omega_{0}k'}$" + " = {}".format(round(a_coeff*1e6,0)) + r"$\pm$" + "{}".format(round(a_coeff_err*1e6,0)) + r"$\times10^{-6}\frac{\sqrt{amu} \; Torr}{e\cdot\mu m}$")

#plt.errorbar(masses,mean_scaled_pmaxes,yerr=std_scaled_pmaxes,fmt='o',ms=5)

plt.scatter(masses,mean_scaled_pmaxes)

plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))

plt.ylabel(r'$P_{\pi/2}/d$ [$Torr/e\cdot\mu m$]',fontsize = 10)
plt.xlabel(r'Mass [amu]',fontsize=10)

plt.legend()

plt.show()

if save:
	np.savez(out_path + 'pmax_vs_mass_fit', masses=masses, scaled_pmaxes=mean_scaled_pmaxes,\
		 	 std_scaled_pmaxes=std_scaled_pmaxes, a_coeff=a_coeff, a_coeff_err=a_coeff_err)

