import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import cant_util as cu
import grav_util as gu
import scipy.signal as signal
import scipy.optimize as optimize 

show_final_stitching = True


### Load theoretical modified gravity curves

gpath = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/'
filname = 'attractorv2_1um_manysep_fullthrow_force_curves.p'

fcurve_obj = gu.Grav_force_curve(path=gpath+filname, make_splines=True, spline_order=3)

SEP = 5.0e-6
ALPHA = 1.e8
YUKLAMBDA = 10.0e-6
RBEAD = 2.43e-6



limitdata_path = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160928_datathief_nodecca2.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/home/charles/opt_lev_analysis/scripts/gravity_sim/data/limitdata_20160914_datathief.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'




background_data = cu.Force_v_pos()
background_data.load('/force_v_pos/20170718_grav_background.p')

bins = background_data.bins
force = background_data.force
errs = background_data.errs

diagbins = background_data.diagbins
diagforce = background_data.diagforce
diagerrs = background_data.diagerrs

wvnum = np.fft.rfftfreq(len(force), d=(bins[1]-bins[0]))
datfft = np.fft.rfft(force)
datasd = np.abs(datfft)

diagwvnum = np.fft.rfftfreq(len(diagforce), d=(diagbins[1]-diagbins[0]))
diagdatfft = np.fft.rfft(diagforce)
diagdatasd = np.abs(diagdatfft)





lambdas = fcurve_obj.lambdas
alphas = np.zeros_like(lambdas)
diagalphas = np.zeros_like(lambdas)

for ind, yuklambda in enumerate(lambdas):
    fcurve = fcurve_obj.mod_grav_force(final_dat.bins*1e-6, sep=SEP, alpha=1., \
                                       yuklambda=yuklambda, rbead=RBEAD, nograv=True)
    diagfcurve = fcurve_obj.mod_grav_force(final_dat.diagbins*1e-6, sep=SEP, alpha=1., \
                                           yuklambda=yuklambda, rbead=RBEAD, nograv=True)

    fcurve = signal.detrend(fcurve)
    diagfcurve = signal.detrend(diagfcurve)

    fft = np.fft.rfft(fcurve)
    asd = np.sqrt(fft.conj() * fft)
    diagfft = np.fft.rfft(diagfcurve)
    diagasd = np.sqrt(diagfft.conj() * diagfft)

    maxind = np.argmax(asd)
    diagmaxind = np.argmax(diagasd)

    alpha = datasd[maxind] / asd[maxind]
    diagalpha = diagdatasd[diagmaxind] / diagasd[maxind]

    alphas[ind] = alpha
    diagalphas[ind] = diagalpha

    







f, axarr = plt.subplots(1,2,sharex='all',sharey='all',figsize=(10,5),dpi=100)

#print final_dat.bins
#print final_dat.force

axarr[0].errorbar(final_dat.bins, final_dat.force*1e15, final_dat.errs*1e15, \
               fmt='.-', ms=10, color = 'r') #, label=lab)
axarr[0].plot(final_dat.bins, grav_dat*1e15, '-', color='k')

axarr[0].set_ylabel('Force [fN]')
axarr[0].set_xlabel('Position along cantilever [um]')

axarr[1].errorbar(final_dat.diagbins, final_dat.diagforce*1e15, final_dat.diagerrs*1e15, \
               fmt='.-', ms=10, color = 'r') #, label=lab)
axarr[1].plot(final_dat.diagbins, diag_grav_dat*1e15, '-', color='k')

axarr[1].set_xlabel('Position along cantilever [um]')






f2, axarr2 = plt.subplots(1,2,sharex='all',sharey='all',figsize=(10,5),dpi=100)

axarr2[0].loglog(wvnum, np.sqrt(datfft.conj() * datfft) )
axarr2[0].loglog(wvnum, np.sqrt(gravfft.conj() * gravfft) )

axarr2[0].set_ylabel('ASD [N / rt(um$^{-1}$)]')
axarr2[0].set_xlabel('Wavenumber [um$^{-1}$]')

axarr2[1].loglog(diagwvnum, np.sqrt(diagdatfft.conj() * diagdatfft) )
axarr2[1].loglog(diagwvnum, np.sqrt(diaggravfft.conj() * diaggravfft) )

axarr2[1].set_xlabel('Wavenumber [um$^{-1}$]')






f3, axarr3 = plt.subplots(1,2,sharex='all',sharey='all',figsize=(10,5),dpi=100)

axarr3[0].loglog(lambdas, alphas, label='Raw Data: Sig = Noise')
axarr3[0].loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
axarr3[0].loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')

axarr3[0].set_ylabel('alpha [arb]')
axarr3[0].set_xlabel('lambda [m]')

axarr3[1].loglog(lambdas, diagalphas, label='Diagonalized Data: Sig = Noise')
axarr3[1].loglog(limitdata[:,0], limitdata[:,1], '--', label=limitlab, linewidth=3, color='r')
axarr3[1].loglog(limitdata2[:,0], limitdata2[:,1], '--', label=limitlab2, linewidth=3, color='k')

axarr3[1].set_xlabel('lambda [m]')

axarr3[0].legend(numpoints=1)
axarr3[1].legend(numpoints=1)






plt.show()
