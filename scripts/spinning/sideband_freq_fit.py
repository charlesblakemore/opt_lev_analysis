import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import os 

from scipy.optimize import curve_fit

plt.rcParams.update({'font.size':14})

#path = '/processed_data/spinning/wobble/20190829/long_wobble_before_pramp'
path = '/processed_data/spinning/wobble/20190905/before_pramp'
out_path = '/processed_data/spinning/wobble/20190905/before_pramp/dipoles/'


bu.make_all_pardirs(out_path)
files, zero = bu.find_all_fnames(path, ext='.npy', sort_time=True)

files = files[-24:]

date = path.split('/')[-2]

for root, dirs, filenames in os.walk('/calibrations/masses'):
	for i, filename in enumerate(filenames):
		if date in filename:
			calibration_file = root + '/'+ filename

print calibration_file
###Microsphere Properties and Errors###

mass_arr = np.load(calibration_file)
ms_mass = mass_arr[0]
ms_mass_err = mass_arr[1]

print mass_arr

rho = 1550.
rho_err = 80.

ms_radius = ((3./4.) * (ms_mass/rho))**(1./3.)
ms_radius_err = np.sqrt( ((1./3.) * ((3./4.) * (ms_mass/rho))**(-2./3.)\
						* (3./4.) * (1/rho) * ms_mass_err)**2 + ((1./3.) * ((3./4.)  \
						* (ms_mass/rho))**(-2./3.) * (-3./4.) * (ms_mass/rho**2) * rho_err)**2 )
ms_radius_sys_err = 0.


I = 0.4 * ms_mass * ms_radius**2.
I_err = np.sqrt( (0.4 * ms_radius**2. * ms_mass_err)**2. + (0.4 * 2. \
				* ms_mass * ms_radius * ms_radius_err)**2. )
I_sys_err = 0. 

print ms_mass, ms_mass_err 
########################################

save = True 
plot = False
separate_by_name = False 
names = ['init','after']

if not separate_by_name:
	names = []


dipoles = [[] for i in range(len(names))]
dipole_errs = [[] for i in range(len(names))]
weights = [[] for i in range(len(names))]

def sqrt(x, a):
	return a * np.sqrt(x)

colors = bu.get_color_map(len(files), cmap='plasma')


errors = []
dic = {'dipoles': dipoles,'errors': dipole_errs, 'weights': weights,  'names': names}


efield_arr = []

totals = np.zeros_like(names)
num_files = np.zeros_like(totals)
	
if plot:
	plt.figure(dpi=150)
for i, fil in enumerate(files):
	data = np.load(fil)

	meas_name = fil.split('/')[-1].split('.')[0]
		
	efields = data[0]
	spin_freqs = data[2] * 2 * np.pi
	spin_freq_errs = data[3] * 2 * np.pi

	mask = (spin_freqs >= 2 * np.pi * 80) & (spin_freqs <=  2 * np.pi * 3000) 

	
	efield_arr.append(efields[1])
		
	#plt.scatter(efields[mask],spin_freqs[mask], c = colors[i])
	
	popt, pcov = curve_fit(sqrt, efields[mask], spin_freqs[mask], sigma=spin_freq_errs[mask])	
	fit_err = np.sqrt(np.diag(pcov))			

	x_axis = np.linspace(0, efields[0], len(spin_freqs) * 2 )
	
	d = popt[0]**2. * I * (1/1.602e-19) * (1.e6)
	d_err = np.sqrt( (2. * popt[0] * I * fit_err[0])**2.\
					+ (popt[0]**2. * I_err)**2. ) * (1./1.602e-19) * (1.e6) 

		
	#Organize dipoles by names
	if separate_by_name:
		for j in range(len(names)):
			if dic[names][j] in meas_name:
				dic['dipoles'][j].append(d)
				dic['errors'][j].append(d_err)
				dic['weights'][j].append(1./(d_err)**2)

	#Else just store all dipole moments together
	else: 
		dic['dipoles'].append(d)
		dic['errors'].append(d_err)
		dic['weights'].append(1./(d_err)**2)
		dic['names'].append(meas_name)

		if save:
			np.save(open(out_path + meas_name + '.dipole', 'wb'), [d , d_err])

	if plot:
		plt.scatter(efields[mask],spin_freqs[mask], c = colors[i])
		plt.plot(x_axis, sqrt(x_axis, *popt), c = colors[i], label=meas_name +\
			 ', d = {:.1f} $\pm$ {:.3f} $e \cdot \mu m$'.format(d, d_err))
	
	
	print(fil, popt, d )





if separate_by_name:	


	weighted_avgs = [[] for i in range(len(dic['dipoles']))]
	sum_of_weights = [[] for i in range(len(dic['dipoles']))]
	avgs = [[] for i in range(len(dic['dipoles']))]
	stds = [[] for i in range(len(dic['dipoles']))]
	
	for i in range(len(avgs)):
		weighted_avgs[i], sum_of_weights[i] = np.average(dic['dipoles'][i], weights=dic['weights'][i], returned=True )
		avgs[i] = np.mean(dic['dipoles'][i])
		stds[i] = np.std(dic['dipoles'][i], ddof = 1) #ddof=1 for sample standard deviation
		
	std_means = stds/np.sqrt(len(dic['dipoles'][0]))
	title_str = ''
	
	for i, name in enumerate(dic['names']):
		title_str += '{}: {:0.1f} $\pm$ {:0.1f} '.format(name, weighted_avgs[i], 1./np.sqrt(sum_of_weights[i]))
	if plot:
		plt.title(title_str)

	print sum_of_weights
	print dic['errors'][0] 

if plot:
	plt.legend()
	plt.xlabel('E-Field [V/m]')
	plt.ylabel('$\omega_{\phi} [rad/s]$')
	plt.show()


#fig, ax = plt.subplots(2,1)
#ax[0].plot(dic['dipoles'])
#
#ax[1].plot(efield_arr)
#plt.show()
