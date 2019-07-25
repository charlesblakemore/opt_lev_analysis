import rga_util as ru
from rga_util import *
import re

#filename = 'pre-He-leak_1_000001.txt'
#directory = '/daq2/20190514/bead1/rga_scans/'

#filename = 'meas2_He-leak_5_rerun_2_000001.txt'

m_N = 14
m_N2 = 28.01
m_He = 4.002
m_O2 = 32
m_H20 =  18.0153
mass_gas = [m_He,m_N2,m_O2,m_H20]
mass_name = ['He','N2','O2','H2O']


directory = ['/daq2/20190626/bead1/spinning/pramp/He/rga/']#'/daq2/20190514/bead1/spinning/pramp3/He/rga_scans/He_20190607_measurement_1/', '/daq2/20190514/bead1/spinning/pramp3/He/rga_scans/He_20190609_measurement_3/','/daq2/20190514/bead1/spinning/pramp3/He/rga_scans/He_20190609_measurement_4/']


back_files = ['before_1_000001.txt','before_2_000001.txt','before_3_000001.txt']
leak_files = ['flush_1_000001.txt','flush_2_000001.txt','flush_3_000001.txt']

#back_files = ['meas1_N2_preleak_1_000001.txt']#'meas1_pre-He-leak_1_000001.txt','meas3_He_preleak_1_000001.txt','meas4_He_preleak_1_000001.txt']
#leak_files = ['meas1_N2_leak_1_000001.txt']#'meas1_He-leak_1_000001.txt','meas3_He_leak_1_000001.txt','meas4_He_leak_1_000001.txt']
meas_gas = 'He'


num_files = len(back_files)

background_data = np.zeros(num_files)

#fig, ax = plt.subplots(num_files,sharex=True,sharey=True)

meas_num_regex = re.compile(r'\d')
meas_type_regex = re.compile(r'preleak')

def name_search(f):
	print(f)
	meas_num = meas_num_regex.search(f)
	meas_type = f.find('preleak')
	
	return meas_num.group(0), meas_type




def plot_rga_data(lines, skip_ind = 0):
	mass_data = ru.extract_mass_vec(lines)
	mass_vec = mass_data['mass_vec']
	
	scan_data = ru.extract_scans(lines, mass_data['mass_line_ind'],
	mass_data['nscans'])
	
	
	part_press = scan_data['scans']
	tot_pressure = scan_data['pressures']
	
	nscans = len(part_press)
	diffs = np.array([])
	for i in range(nscans):
		diff = np.abs(part_press[i]-part_press[-1])#/(np.sum(part_press[-1]))
		
			
		diffs = np.append(diffs,np.sum(diff)/np.sum(part_press[-1]))

		
	
	mask = diffs < 0.75
	part_press = part_press[mask,:]	
	
	#plt.plot()
	#plt.show()
	
	nscans = len(part_press)	
	colors = bu.get_color_map(nscans, cmap='inferno')
	
	
	fig, ax = plt.subplots(1,1,figsize=(10,3))
	for i in range(nscans):
		ax.semilogy(mass_vec,part_press[i],color=colors[i])
	
	plt.tight_layout()
	#plt.show()
	
	print(len(tot_pressure))
	print(len(diffs))

	mean_part_press = np.mean(part_press)
	pressure = tot_pressure[mask]
	mean_pressure = np.mean(pressure)
	
	return mean_part_press, mean_pressure, mass_vec

file_obj = open(directory[0]+leak_files[0])
lines = file_obj.readlines()

leak_part_press, leak_pressure, leak_mass_vec = plot_rga_data(lines, skip_ind = 0)

#back_part_press, back_pressure, back_mass_vec = plot_rga_dat(

'''
for i in range(num_files):
	meas_num, meas_type = 0, ''
	
	print(i)
	background_data = ru.get_rga_data(directory[0]+back_files[i], plot = True)
	leak_data = ru.get_rga_data(directory[0]+leak_files[i], plot=True)

	back_x = background_data['mass_vec']
	back_y = background_data['partial_pressures']

	leak_x = leak_data['mass_vec']
	leak_y = leak_data['partial_pressures']
		
	if num_files > 1:
		#ax[i].plot(leak_x,(leak_y-back_y)/background_data['pressure'])
		ax[i].semilogy(leak_x,leak_y/background_data['pressure'])
		ax[i].semilogy(leak_x,back_y/background_data['pressure'])
		ax[i].set_title("RGA Scan {} measurement {}".format(meas_gas,meas_num))
		#ax[i].set_yticks(np.arange(0,1.0,0.3))
		
	else:
		#ax.semilogy(leak_x,(leak_y-back_y)/background_data['pressure'])
		ax.semilogy(leak_x,leak_y/background_data['pressure'])
		ax.semilogy(leak_x,back_y/background_data['pressure'])
		ax.set_title("RGA Scan {} leak measurement {}".format(meas_gas,meas_num))
		ax.set_xlabel('Mass [amu]')
		ax.set_ylabel(r'$(P_{leak} - P_{background})/P_{total}$')
	

#ax.set_ylabel(r'$(P_{leak} - P_{preleak})P_{tot}$',fontsize =10)
#ax[2].set_xlabel('Mass [amu]')
fig.tight_layout()
plt.show()
'''
