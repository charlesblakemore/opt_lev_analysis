import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import glob
import scipy.signal as ss
from scipy.optimize import curve_fit
from sideband_analysis_amp_hilbert import find_efield_amp
from amp_ramp_3 import bp_filt
import re, sys

save = True

#path = "/data/20181204/bead1/high_speed_digitizer/pramp/50k_zhat_8vpp_3"

#path = "/daq2/20190408/bead1/spinning/49kHz_200Vpp_pramp-N2_1"
#path = '/daq2/20190417/bead1/pramps/N2_3vpp_0'
#path = "/daq2/20190430/bead1/spinning/he/1vpp_50kHz_0"
#path = "/daq2/20190514/bead1/spinning/pramp/Xe/50kHz_4Vpp_1"
#path = "/daq2/20190514/bead1/spinning/pramp/He/50kHz_4Vpp_3"
#path_list = ['/daq2/20190626/bead1/spinning/pramp/N2/50kHz_4Vpp',\
#		'/daq2/20190626/bead1/spinning/pramp/N2/50kHz_4Vpp_2',\
#		'/daq2/20190626/bead1/spinning/pramp/N2/50kHz_4Vpp_3']	
path_list = ['/data/old_trap/20191017/bead1/spinning/pramp/He/50kHz_4Vpp_1/']#,\
	     #'/daq2/20190905/bead1/spinning/pramp/He/50kHz_4Vpp_5']
path_list = ['/data/old_trap/20200330/gbead3/spinning/pramp_2/25kHz_3Vpp_xy_1/',\
             ]#'/data/old_trap/20200330/gbead3/spinning/pramp/50kHz_4Vpp_xy_1/',\
             #'/data/old_trap/20200330/gbead3/spinning/pramp/50kHz_6Vpp_xy_1/',\
             #'/data/old_trap/20200330/gbead3/spinning/pramp/50kHz_8Vpp_xy_1/']
drive_ax = 1
data_ax = 0

#out_f = "/processed_data/spinning/pramp_data/49k_200vpp"
#out_f = "/home/dmartin/analyzedData/20190430/20190430_1vpp_50kHz_0"
#out_f = "/home/dmartin/analyzedData/20190417/N2/N2_3vpp_0"
#out_f = "/home/dmartin/analyzedData/20190514/Xe_4Vpp_50kHz_1"
#out_f = "/home/dmartin/analyzedData/20190514/pramp/He/50kHz_4Vpp_3"
out_f = "/home/dmartin/Desktop/analyzedData/20200330/gbead3/spinning/pramp_2/"


spinning_freq = 25e3
plot_dat = False 
 
def line(x, m, b):
    return m*x + b

def dec2(arr, fac):
    return ss.decimate(ss.decimate(arr, fac), fac)
corr = []
for i, path in enumerate(path_list):
    bu.make_all_pardirs(out_f)
    
    files, zero = bu.find_all_fnames(path, sort_time=True)
    
    #files = files[0:500]
    
    init_file = 0
    final_file = len(files)
    n_file = final_file-init_file
    
    bw = 10
    obj0 = hsDat(files[init_file])
    t0 = obj0.attribs['time']
    Ns = obj0.attribs['nsamp']
    Fs = obj0.attribs['fsamp']
    
    tarr0 = np.linspace(0, (Ns-1)/Fs, Ns)
    
    
    freqs = np.fft.rfftfreq(Ns, d = 1./Fs)
    times = np.zeros(n_file)
    phases = np.zeros(n_file)
    dphases = np.zeros(n_file)
    pressures = np.zeros((n_file, 3))
    
    fc = 2.*spinning_freq
    bfreq = np.abs(freqs-fc)>bw/2.
    bfreq2 = np.abs(freqs-fc/2.)>bw/2.
    
    for i, f in enumerate(files[init_file:final_file]):
        bu.progress_bar(i, n_file)
        sys.stdout.flush()
        #print f
        try:
        
            obj = hsDat(f)
           
            Ns = obj.attribs['nsamp']
            Fs = obj.attribs['fsamp']
            
            efield_arr = find_efield_amp(obj, spinning_freq)

            filt_crossp = bp_filt(obj.dat[:, data_ax], 2*spinning_freq, Ns, Fs, bw)
            filt_drive = bp_filt(obj.dat[:,drive_ax], spinning_freq, Ns, Fs, bw )
            
            fft = np.fft.rfft(filt_crossp)
            fft2 = np.fft.rfft(filt_drive)

            #fft = np.fft.rfft(obj.dat[:, data_ax])
            #fft2 = np.fft.rfft(obj.dat[:, drive_ax])
            #fft[bfreq] = 0.
            #fft2[bfreq2] = 0.
            if plot_dat:
            	plt.loglog(freqs, np.abs(fft))
            	plt.loglog(freqs, np.abs(fft2))
            	plt.show()
            phases[i] = np.angle(np.sum(fft))
            dphases[i] = np.angle(np.sum(fft2))
            pressures[i, :] = obj.attribs["pressures"]
            times[i] = obj.attribs['time']
        
        except:
            print "bad file"
        
#        amp = np.abs(fft[freqs == 100e3])
#        damp = np.abs(fft2[freqs == 50e3])
        
#        corr.append(amp/damp)
        
    dphases[dphases<0.]+=np.pi
    corr = np.array(corr)
    
    phi = phases - 2.*dphases
    phi[phi>np.pi]-=2.*np.pi
    phi[phi<-np.pi]+=2.*np.pi
    
    meas_name = path.split('/')[-2]
    save_name = out_f + meas_name
    print('save name', save_name)
    if save:
        np.save(save_name + '_efield.npy', efield_arr)
        np.save(save_name + '_phi.npy', phi)
        np.save(save_name + "_pressures.npy", pressures)
        np.save(save_name + "_time.npy", times)
        np.save(save_name + "_corr.npy", corr)	
