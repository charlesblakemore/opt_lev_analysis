import numpy, h5py
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu

refname = "URmbar_xyzcool_elec0_10000mV41Hz0mVdc_stageX0nmY6000nmZ4000nm.h5"
fname0 = ""
path = "/data/20150827/Bead3/cant_bias2/"
d2plt = 1
if fname0 == "":
	filelist = os.listdir(path)

	mtime = 0
	mrf = ""
	for fin in filelist:
		f = os.path.join(path, fin) 
		if os.path.getmtime(f)>mtime:
			mrf = f
			mtime = os.path.getmtime(f) 
 
	fname0 = mrf		


		 

Fs = 5e3  ## this is ignored with HDF5 files
NFFT = 2**17

def getdata(fname):
	print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
		
		#dat = 1.0*dat*max_volt/nbit

	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	xpsd, freqs = matplotlib.mlab.psd(dat[:, 0]-numpy.mean(dat[:, 0]), Fs = Fs, NFFT = NFFT) 
	ypsd, freqs = matplotlib.mlab.psd(dat[:, 1]-numpy.mean(dat[:, 1]), Fs = Fs, NFFT = NFFT)
        zpsd, freqs = matplotlib.mlab.psd(dat[:, 2]-numpy.mean(dat[:, 2]), Fs = Fs, NFFT = NFFT)

	norm = numpy.median(dat[:, 2])
        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,2])**2
	return [freqs, xpsd, ypsd, dat, zpsd]

data0 = getdata(os.path.join(path, fname0))

def rotate(vec1, vec2, theta):
    vecn1 = numpy.cos(theta)*vec1 + numpy.sin(theta)*vec2
    vecn2 = numpy.sin(theta)*vec1 + numpy.cos(theta)*vec2
    return [vec1, vec2]


if refname:
	data1 = getdata(os.path.join(path, refname))
Fs = 10000
b, a = sp.butter(1, [2*5./Fs, 2*10./Fs], btype = 'bandpass')

if d2plt:	

        fig = plt.figure()
        plt.plot(data0[3][:, 0])
        plt.plot(data0[3][:, 1])
        plt.plot(data0[3][:, 3])
       # plt.plot(np.abs(data0[3][:, 3])-np.mean(np.abs(data0[3][:, 3])))
       



fig = plt.figure()
plt.subplot(3, 1, 1)
plt.loglog(data0[0], data0[1],label="Data")
#if refname:
#	plt.loglog(data1[0], data1[1],label="Ref")
plt.ylabel("V$^2$/Hz")
plt.legend()
plt.subplot(3, 1, 2)
plt.loglog(data0[0], data0[2])
#if refname:
#	plt.loglog(data1[0], data1[2])
plt.subplot(3, 1, 3)
plt.loglog(data0[0], data0[4])
#if refname:
#	plt.loglog(data1[0], data1[4])
plt.ylabel("V$^2$/Hz")
plt.xlabel("Frequency[Hz]")
plt.close()

refname = "/data/20150827/Bead3/1_5mbar_zcool.h5"
r,b,bc = bu.get_calibration(refname, [10,500], make_plot=True, 
                    data_columns = [0,1], drive_column=11, NFFT=2**14, exclude_peaks=False)

f = h5py.File(refname,'r')
dset = f['beads/data/pos_data']
dat = numpy.transpose(dset)
#max_volt = dset.attrs['max_volt']
#nbit = dset.attrs['nbit']
Fs = dset.attrs['Fsamp']
ypsd, freqs = matplotlib.mlab.psd(dat[:, 1]-numpy.mean(dat[:, 1]), Fs = Fs, NFFT = NFFT)

plt.figure()
plt.loglog( freqs, np.sqrt(ypsd)*r )
plt.loglog( data0[0], np.sqrt(data0[2])*r, 'r')

sfac = 4e-9/1e-5 * bu.bead_mass * (2*np.pi*150)**2

fig = plt.figure()
plt.plot([41, 41], [1e-17, 1e-14], 'r--', linewidth=1.5)
plt.plot([82, 82], [1e-17, 1e-14], 'r--', linewidth=1.5)
plt.text(35,1e-13,r"$\omega$",color="r")
plt.semilogy(data0[0], np.sqrt(data0[2])*r*sfac, 'k', linewidth=1.5)
plt.xlim([10, 100])
plt.ylim([1e-17, 1e-14])
plt.xlabel("Frequency [Hz]")
plt.ylabel("Force [N Hz$^{-1/2}$]")

fig.set_size_inches(8,6)
plt.savefig("dipole_resp.pdf")

plt.show()
