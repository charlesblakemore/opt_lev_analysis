import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import rga_util as ru
import bead_util_funcs as buf
plt.rcParams.update({'font.size': 14})


#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/background2000001.txt'
#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/background_nextday000001.txt'

#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/preleak000001.txt'
#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/leak000001.txt'
#rga_data_file2 = '/daq2/20190514/bead1/rga_scans/postleak000001.txt'

#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/pre_Ar-leak_3_000001.txt'
#rga_data_file1 = '/daq2/20190514/bead1/rga_scans/Ar-leak_2_000001.txt'
#rga_data_file2 = '/daq2/20190514/bead1/rga_scans/post_Ar-leak_4_000001.txt'

#base = '/daq2/20190626/bead1/spinning/pramp/N2_2/rga/'

#rga_data_file1 = base + 'He_20190607_measurement_1/meas1_pre-He-leak_2_000001.txt'
#rga_data_file2 = base + 'He_20190607_measurement_1/meas1_He-leak_1_000001.txt'

#rga_data_file1 = base + 'He_20190607_measurement_2/meas2_He-leak_2_000001.txt'
#rga_data_file2 = base + 'He_20190607_measurement_2/meas2_He-leak_3_000001.txt'


#base1 = '/daq2/20190514/bead1/spinning/pramp3/He/rga_scans/'
#rga_data_file1 = base1 + 'He_20190607_measurement_1/meas1_pre-He-leak_2_000001.txt'

#base2 = '/daq2/20190625/rga_scans/'
#rga_data_file2 = base2 + 'reseat_with-grease_000002.txt'

gas = 'Ar'
rga_data_file_list1 = ['/daq2/20190626/bead1/spinning/pramp/Ar/rga/before_1_000001.txt',\
				  '/daq2/20190626/bead1/spinning/pramp/Ar/rga/before_2_000001.txt',\
				  '/daq2/20190626/bead1/spinning/pramp/Ar/rga/before_3_000001.txt']
rga_data_file_list2 = ['/daq2/20190626/bead1/spinning/pramp/Ar/rga/flush_1_000001.txt',\
				  '/daq2/20190626/bead1/spinning/pramp/Ar/rga/flush_2_000001.txt',\
				  '/daq2/20190626/bead1/spinning/pramp/Ar/rga/flush_3_000001.txt']

plot_scan = False 
plot_all_scans = False 
plot_together = True

scans_from_end = 4

title = 'RGA Scan Evolution'#: Reseating Window'

filename = '/home/dmartin/analyzedData/20190626/pramp/Ar/rga'
save = True 

gases_to_label = {'He': 3.9, \
                  'H$_2$O': 18, \
                  'N$_2$': 28, \
                  'O$_2$': 32, \
                  'Ar': 40, \
                  #'Kr': 84, \
                  #'Xe': 131, \
                  #'SF$_6$': 146, \
                  }


#arrow_len = 0.02
#arrow_len = 0.2
arrow_len = 0.2


		
######################################################
def RGA_scan_comparison(rga_data_file1,rga_data_file2,list_ind=0,end_ind=1):
	filename1 = filename + '/{}'.format(rga_data_file1.split('/')[-1]) + '_and_{}'.format(rga_data_file2.split('/')[-1]) 
	pp1 = []
	pp2 = []
	
	p1 = []
	p2 = []
	
	#print(rga_data_file1[list_ind])	
	#print(rga_data_file2[list_ind])
	for i in range(end_ind):
		#i-end_ind will take only the last #(end_ind) scans
		dat1 = ru.get_rga_data(rga_data_file1, all_scans=False, scan_ind=(i-end_ind), \
		                       plot=plot_scan, plot_many=plot_all_scans)
		dat2 = ru.get_rga_data(rga_data_file2, all_scans=False, scan_ind=(i-end_ind), \
		                       plot=plot_scan, plot_many=plot_all_scans)
		
			
		pp1.append(dat1['partial_pressures'])
		pp2.append(dat2['partial_pressures'])
		p1.append(dat1['pressure'])
		p2.append(dat2['pressure'])
	
	m1 = dat1['mass_vec']
	m2 = dat2['mass_vec']
	
	p1 = np.mean(p1)
	p2 = np.mean(p2)
	
	e1 = np.std(pp1,axis=0)
	e2 = np.std(pp2,axis=0)
	
	pp1 = np.mean(pp1,axis=0)
	pp2 = np.mean(pp2,axis=0)
	
	print(e1)
	
	
	'''
	m1 = dat1['mass_vec']
	m2 = dat2['mass_vec']
	
	pp1 = dat1['partial_pressures']
	pp2 = dat2['partial_pressures']
	
	p1 = dat1['pressure']
	p2 = dat2['pressure']
	
	e1 = dat1['errs']
	e2 = dat2['errs']
	'''
	
	diff = pp2 - pp1
	diff_err = np.sqrt(e1**2 + e2**2)
	
	pp_mean = 0.5 * (pp1 + pp2)
	p_mean = 0.5 * (p1 + p2)
	
	#p_tot = p_mean
	p_tot = dat1['pressure']
	
	
	
	if plot_together:
	
	    fun = lambda x: '{:0.3g}'.format(x)
	    title_str = 'Scan Comparison'
	
	    fig, ax = plt.subplots(1,1,dpi=300,figsize=(10,3))
	    ax.errorbar(m1, pp1, yerr=e1, color='C0')
	    ax.fill_between(m2, pp1, np.ones_like(pp1)*1e-10,\
	                    alpha=0.5, color='C0', label=('Before: ' + fun(p1) + ' torr'))
	    ax.errorbar(m2, pp2, yerr=e2, color='C1')
	    ax.fill_between(m2, pp2, np.ones_like(pp2)*1e-10,\
	                    alpha=0.5, color='C1', label=('After: ' + fun(p2) + ' torr'))
	    ax.set_ylim(1e-9, 2*np.max([np.max(pp1), np.max(pp2)]) )
	    ax.set_xlim(0,int(np.max([np.max(m1), np.max(m2)])))
	    ax.set_yscale('log')
	    ax.set_xlabel('Mass [amu]')
	    ax.set_ylabel('Partial Pressure [torr]')
	    fig.suptitle(title_str)
	    plt.tight_layout()
	    plt.legend()
	    plt.subplots_adjust(top=0.87)
	    #plt.show()
		
        if save:
            fig.savefig(filename1 + '_scan_comp.png')
	
	
	
	
	
	
	fig, ax = plt.subplots(1,1,dpi=300,figsize=(10,4))
	
	ax.errorbar(m1, diff/p_tot, yerr=diff_err/p_tot)
	#ax.errorbar(m1, diff/pp1, yerr=diff_err/p_tot)
	
	gas_keys = gases_to_label.keys()
	labels = []
	neg = False
	negmax = 0
	pos = False
	posmax = 0
	for gas in gas_keys:
	    mass = gases_to_label[gas]
	    mass_ind = np.argmin( np.abs(m1 - mass) )
	
	    val_init = diff[mass_ind]
	    if val_init > 0:
	        pos = True
	        val = np.max((diff/p_tot)[mass_ind-5:mass_ind+5]) + \
	                np.max((diff_err/p_tot)[mass_ind-5:mass_ind+5])
	        if val > posmax:
	            posmax = val
	    if val_init < 0:
	        neg = True
	        val = np.min((diff/p_tot)[mass_ind-5:mass_ind+5]) - \
	                np.max((diff_err/p_tot)[mass_ind-5:mass_ind+5])
	        if val < negmax:
	            negmax = val
	
	    labels.append(ax.annotate(gas, (mass, val), \
	                              xytext=(mass, val + np.sign(val)*arrow_len), \
	                              ha='center', va='center', \
	                              arrowprops={'width': 2, 'headwidth': 3, \
	                                          'headlength': 5, 'shrink': 0.0}))
	
	ax.set_xlabel('Mass [amu]')
	ax.set_ylabel('$(\Delta P \, / \, P_{\mathrm{init}})$ [abs]')
	ax.set_xlim(0,150)
	
	y1, y2 = ax.get_ylim()
	if pos:
	    y2 = posmax + 2.0*arrow_len
	if neg:
	    y1 = negmax - 2.0*arrow_len
	ax.set_ylim(y1, y2)
	
	fig.suptitle(title)
	
	plt.tight_layout()
	plt.subplots_adjust(top=0.89)
	
	if save:
	    fig.savefig(filename1 + '_scan_evol.png')
	
	#plt.show()

if __name__ == '__main__':
     for i in range(len(rga_data_file_list1)):
          RGA_scan_comparison(rga_data_file_list1[i],rga_data_file_list2[i],i,scans_from_end)

