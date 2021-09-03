import numpy as np
import matplotlib.pyplot as plt
import os
import re
from iminuit import Minuit
from scipy.optimize import curve_fit
save = False 
no_offset = True 

if no_offset:
	offset_state = 'no_offset'
else:
	offset_state = 'offset'
 
rho = 1.55e3 #kg/m^3
m = 85.0e-15 #kg

r = (0.75 * m/(np.pi*rho))**(0.3333333)

I = (0.4) * m * (0.75 * (m/(np.pi*rho)))**(0.6666666)

def sqrt(E, d, c):
	func = np.sqrt(E*d) + c

	if no_offset:
		func = np.sqrt(E*d)
	return func


#path = "/processed_data/spinning/wobble/20190626/slow/"
#path = ["/processed_data/spinning/wobble/20190626/Kr_pramp_1/"]#,\
		#"/processed_data/spinning/wobble/20190626/N2_pramp_2/",\
		#"/processed_data/spinning/wobble/20190626/N2_pramp_3/"]	
#path = ["/home/dmartin/analyzedData/20190626/pramp/wobble_many_wobble_0000data_arr.npy"]

#path = ["/processed_data/spinning/wobble/20190626/initial_slow/",\
#		"/processed_data/spinning/wobble/20190626/after-highp_slow_later/",\
#		"/processed_data/spinning/wobble/20190626/after-highp_slow/"]#,\
		#"/processed_data/spinning/wobble/20190626/slow_later/"]

path = ["/home/dmartin/analyzedData/20190805/wobble/init_dipole_2kHz/"]
path = ['/home/dmartin/Desktop/analyzedData/20200924/dipole_meas/trial_0000/']
out_path = "/home/dmartin/analyzedData/20190626/wobble/new/"
#filenames_ = ['reset_dipole_3_wobble_2.npy','reset_dipole_2wobble_0.npy','reset_dipole_3_wobble_1.npy','reset_dipole_3_wobble_0.npy']
filenames_ = ['trial_0000.npy']

multiple = False 

d_arr = []
meas_names = []
for j, folder in enumerate(path):
	meas_names.append(folder.split('/')[-2])
	#/date = folder.split('/')[-3]  

	print(meas_names)

	if multiple:	
		#for root, dirs, filenames in os.walk(folder):
		for i, filename in enumerate(filenames_):
			print(folder)#+ filename)
			data = np.load(folder+filenames_[i])# + filename)
			
			meas_name_2 = filename.split('/')[-1]	
			#E = data[0] * 50 * (1/4e-3) * 0.66
			E = data[0] * 1e-3 #kV/m
			E_err = data[1] * 1e-3
			wobble_freq = 2 * np.pi * data[2] #rad*Hz
			wobble_freq_err = 2 * np.pi* data[3]
			


				
			popt, pcov = curve_fit(sqrt, E, wobble_freq)#,sigma = wobble_freq_err) 
			
			dipole_1 = popt[0] * I * 1e-3
			dipole = popt[0] *1e-3* I * (1/1.602e-19) * 1e6 #e * um
			d_arr.append(dipole_1)
			
			x = np.linspace(0,max(E),3000)
                        
                        label = "meas {}, {} $e\cdot\mu m$".format(meas_name_2,round(dipole,3)) 
			label = r"$\frac{d}{I}$ = " + "{}".format(round(popt[0],3))
                        if no_offset:
				plt.plot(x ,sqrt(x,*popt),label=label)
				plt.scatter(E,wobble_freq)
			else:
				plt.plot(x ,sqrt(x,*popt),label="meas {}, {} $e\cdot\mu m$, offset = {} rad Hz".format(meas_name_2,round(dipole,3),round(popt[1],3)))		
				plt.scatter(E,wobble_freq)
		d_mean = np.mean(d_arr)
		d_err = np.std(d_arr)
		if save:	
			np.save(out_path + '{}'.format(meas_names[j]),[d_mean,d_err])
	else:
		print(path[0]+filenames_[0])#+ filename)
		data = np.load(path[0]+filenames_[0], allow_pickle=True)# + filename)
		
		#meas_name_2 = filename.split('/')[-1]	
		#E = data[0] * 50 * (1/4e-3) * 0.66
		E = data[0] #* 1e-3 #kV/m
		print(E * 1e3)
		#E_err = data[1] * 1e-3
		wobble_freq =  data[2] #rad*Hz
		wobble_freq_err = data[3]
		print(wobble_freq)	
			
		popt, pcov = curve_fit(sqrt, E, wobble_freq)#,sigma = wobble_freq_err) 
		print(popt[0])
		dipole_1 = popt[0] * I * 1e-3
		dipole = popt[0] *1e-3* I * (1/1.602e-19) * 1e6 #e * um
		d_arr.append(dipole_1)
		
		x = np.linspace(0,max(E),3000)
                
                label = "{} $e\cdot\mu m$".format(round(dipole,3))

                label = r"$\frac{d}{I}$ = " + r"{} ".format(round(popt[0],3)) +\
                        r"$\frac{C}{kg m}$"

		if no_offset:
			plt.plot(x ,sqrt(x,*popt),label=label)
			plt.scatter(E,wobble_freq)
		else:
			plt.plot(x ,sqrt(x,*popt),label="{} $e\cdot\mu m$, offset = {} rad Hz".format(round(dipole,3),round(popt[1],3)))		
			plt.scatter(E,wobble_freq)
	d_mean = np.mean(d_arr)
	d_err = np.std(d_arr)

	plt.xlabel("E [kV/m]")
	plt.ylabel("$\omega_{\phi}$")
	plt.legend()
	if save:
		plt.savefig(out_path + 'wobble_meas_{}_{}.png'.format(meas_names[j],offset_state),dpi = 250)
plt.show()	
