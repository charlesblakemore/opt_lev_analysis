import matplotlib, os, h5py, glob
import matplotlib.pyplot as plt
import scipy.signal as sp
import numpy as np
import bead_util as bu

matplotlib.rc('font', family='serif') 

path = "/data/20140717/Bead5/reduce_feedback"

#file_list = glob.glob( os.path.join(path, "*.h5") )
file_list = [#"../no_charge_41Hz_vramp/_50mV_41Hzurmbar_xyzcool.h5",
             "urmbar_xyzcool_xgain_1_ygain_1_50mV_40Hz.h5",
             "urmbar_xyzcool_xgain_0_5_ygain_1_50mV_40Hz.h5",
             "urmbar_xyzcool_xgain_0_2_ygain_1_50mV_40Hz.h5",
             "urmbar_xyzcool_xgain_0_ygain_0_1_50mV_40Hz.h5"]

print file_list

# fg = sorted(glob.glob("/data/20140717/Bead5/no_charge_41Hz_vramp/*.h5"))
# f1=plt.figure()
# #plt.hold(False)
# f2=plt.figure()
# #plt.hold(False)
# for f in fg:
#     d,a,fh = bu.getdata(f)

#     psd,freqs = matplotlib.mlab.psd( d[:,0], NFFT=2**13, Fs=5000)

#     plt.figure(f1.number)
#     plt.plot( d[:,0] )
#     plt.title(f[-70:])

#     plt.figure(f2.number)
#     plt.loglog( freqs,psd )
#     plt.title(f[-70:])
#     plt.draw()
#     plt.pause(1)

#     fh.close()

# raw_input('e')

col_list = ['k', [0, 0.75, 0.75]]
m_list = ['.', 's']
damp_list = [10e-3, 500]

## first calibrate into physical units
ref_2mbar = "/data/20140717/Bead5/_50mV_1000Hz2mbar_zcool_100s.h5"
abs_cal, fit_bp, fit_cov = bu.get_calibration(ref_2mbar, [75,300],
                                              make_plot=True,
                                              NFFT=2**14,
                                              exclude_peaks=False)
print "Orig fit:"
print fit_bp


fig = plt.figure()
if(False):
    dat, attribs, cf = bu.getdata( ref_2mbar )

    fsamp = attribs["Fsamp"]

    xdat = dat[:, 0]
    xdat -= np.median(xdat)

    NFFT = 2**12

    xpsd, freqs = matplotlib.mlab.psd(xdat, Fs = fsamp, NFFT = NFFT) 

    xpsd = np.sqrt( xpsd ) * abs_cal * 1e9

    plt.semilogy(freqs, xpsd, markerfacecolor='None', markeredgecolor='b', marker='^', linestyle='None', markeredgewidth=1.5, markersize=4)

    xx = np.linspace(75, 300, 1e3)
    plt.plot( xx, bu.bead_spec_rt_hz( xx, fit_bp[0], fit_bp[1], fit_bp[2])*abs_cal*1e9, linewidth=2.5, color='r', linestyle='--' )

alist = []
for i,f in enumerate([file_list[-1], file_list[0]]):


    dat, attribs, cf = bu.getdata( os.path.join(path,f) )

    fsamp = attribs["Fsamp"]

    xdat = dat[:, 0]
    xdat -= np.median(xdat)

    if( i == 0):
        NFFT = 2**20
        wind_fac = (1.0*NFFT/len(xdat))
        xpsd_coarse, freqs_coarse = matplotlib.mlab.psd(xdat, Fs = fsamp, NFFT = 2**20) 
        xpsd_coarse = np.sqrt(xpsd_coarse)*( abs_cal * 1e9)
    else:
        NFFT = 2**12

    xpsd, freqs = matplotlib.mlab.psd(xdat, Fs = fsamp, NFFT = NFFT) 

    xpsd = np.sqrt( xpsd ) * abs_cal * 1e9

    if(i == 0):
        fit_points = [125,138.]
        exc = [[126, 129],
               [134, 138]]
    else:
        fit_points = [50,150.]
        exc = False

    print i
    plt.figure()
    abs_calc, fit_bpc, fit_covc = bu.get_calibration(os.path.join(path,f), fit_points,
                                                  make_plot=False,
                                                  NFFT=NFFT,
                                                     exclude_peaks=exc, 
                                                     spars=[fit_bp[0]*10, 131.888, damp_list[i]])    
    xx = np.linspace(fit_points[0], fit_points[1], 1e3)
    plt.figure( fig.number )
    if( i == 0 ):
        fit_bpc[0]/=2
        fit_bpc[2]*=10
        fit_covc[0,0]*=4**2
        plt.semilogy(freqs_coarse[::], xpsd_coarse[::], markerfacecolor='None', markeredgecolor=[0.7,0.7,0.7], marker=m_list[i], linestyle='None', markeredgewidth=1.5, markersize=1.5)
        plt.plot( xx, bu.bead_spec_rt_hz( xx, fit_bpc[0], fit_bpc[1], fit_bpc[2])*abs_cal*1e9, linewidth=1.5, color=[0,0,1] )
    else:
        plt.semilogy(freqs, xpsd, markerfacecolor='None', markeredgecolor='k', marker=m_list[i], linestyle='None', markeredgewidth=1.5, markersize=4)
        plt.plot( xx, bu.bead_spec_rt_hz( xx, fit_bpc[0], fit_bpc[1], fit_bpc[2])*abs_cal*1e9, linewidth=2.5, color='r', linestyle='--' )

    alist.append([ fit_bpc[0], np.sqrt(fit_covc[0,0]) ])
    print "Fit pars:"
    print fit_bpc, fit_covc

    plt.xlim([75, 150])
    plt.ylim([5e-2, 5e3])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"Radial position PSD [nm Hz$^{-1/2}$]")


fig.set_size_inches(4.5,3)
plt.subplots_adjust(top=0.96, right=0.99, bottom=0.15, left=0.15)
plt.savefig("feedback_cooling.pdf")

print "Effective temp:"
T0 = 297.4
alist = np.array(alist)
print alist[1,0]/alist[0,0]*T0*1000, " +/- ", alist[1,1]/alist[0,0]*T0*1000

##print alist[1,0]/fit_bp[0]*T0*1000, " +/- ", alist[1,1]/fit_bp[0]*T0*1000

plt.show()
