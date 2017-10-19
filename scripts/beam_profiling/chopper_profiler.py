import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize as opti

import bead_util as bu
import configuration as config


def gauss_intensity(x, A, xo, w):
    '''Formula for the intensity of a gaussian beam
    with an arbitrary amplitude and mean, but a waist
    parameter defined as usual, and correctly for the
    intensity (rather than E-field).'''
    return A * np.exp( -2.0 * (x - xo)**2 / (w**2) )


#xfile = '/data/20171018/chopper_profiling/xprof.h5'
xfile = '/data/20171018/chopper_profiling/xprof_bright_fast.h5'

#yfile = '/data/20171018/chopper_profiling/yprof.h5'
yfile = '/data/20171018/chopper_profiling/yprof_bright_fast.h5'

guess = 3.0e-3    # m, guess to help fit


xfilobj = bu.DataFile()
xfilobj.load(xfile)
xfilobj.load_other_data()

yfilobj = bu.DataFile()
yfilobj.load(yfile)
yfilobj.load_other_data()



rawx = xfilobj.other_data[-1]
rawy = yfilobj.other_data[-1]

numpoints = len(rawx)
fsamp = xfilobj.fsamp
dt = 1.0 / fsamp
t = np.linspace( 0, (numpoints - 1) * dt, numpoints ) 

xpsd, freqs = mlab.psd(rawx, NFFT=len(rawx), Fs=fsamp)
chopfreq = freqs[np.argmax(xpsd)]

numsamp = int(fsamp / chopfreq)

offset = np.argmax(rawx[:numsamp]) - int(numsamp / 2)

profx = xfilobj.other_data[-2]
profy = yfilobj.other_data[-2]

gradx = np.gradient(rawx)
grady = np.gradient(rawy)

plt.plot(t[offset:numsamp+offset], gradx[offset:numsamp+offset])
plt.show()


dt_chop = 1.0 / chopfreq
numchops = int(t[-1] / dt_chop)
twidth = (guess / (2.0 * np.pi * 10.0e-3)) * dt_chop

for ind in range(numchops):
    if ind != 0:
        continue
    profs = gradx[offset + numsamp * ind : offset + numsamp * (1 + ind)]
    lenprof = int(len(profs) * 0.5)
    pos_prof = profs[:lenprof]
    pos_t = np.linspace( 0, (len(pos_prof) - 1) * dt, len(pos_prof) ) 
    neg_prof = -1.0 * profs[lenprof:]
    neg_t = np.linspace( 0, (len(neg_prof) - 1) * dt, len(neg_prof) ) 

    pos_p0 = [0.1, pos_t[np.argmax(pos_prof)], twidth] 
    neg_p0 = [0.1, neg_t[np.argmax(neg_prof)], twidth] 

    pos_popt, pos_pcov = opti.curve_fit(gauss_intensity, pos_t, pos_prof, \
                                        p0=pos_p0, maxfev=10000)

    neg_popt, neg_pcov = opti.curve_fit(gauss_intensity, neg_t, neg_prof, \
                                        p0=neg_p0, maxfev=10000)

    numpoints_in_prof = 500  # make sure  (num) %2 = 0
    ### BAD BAD AWFUL HARDCODED NUMBER DETERMINED EMPIRICALLY
    ### FROM THE COMMENTED PLOTTING
    #plt.figure()
    #plt.plot(pos_prof)
    #plt.show()

    pos_cent_bin = np.argmin( np.abs(pos_t - pos_popt[1]) )
    neg_cent_bin = np.argmin( np.abs(neg_t - pos_popt[1]) )

    new_pos_bins = (pos_cent_bin+pos_cent_bin/2, pos_cent_bin+pos_cent_bin/2)
    new_neg_bins = (neg_cent_bin+neg_cent_bin/2, neg_cent_bin+neg_cent_bin/2)

    new_pos_t = pos_t[new_pos_bins[0]:new_pos_bins[1]] - pos_popt[1]
    new_neg_t = neg_t[new_neg_bins[0]:new_neg_bins[1]] - neg_popt[1]

    new_pos_prof = pos_prof[new_pos_bins[0]:new_pos_bins[1]]
    new_neg_prof = neg_prof[new_neg_bins[0]:new_neg_bins[1]]

    plt.figure()

    plt.plot(pos_t, pos_prof, pos_t, gauss_intensity(pos_t, *pos_popt) )
    plt.plot(neg_t, neg_prof, neg_t, gauss_intensity(neg_t, *neg_popt) )
    plt.show()





# Plots to make sure data isn't cray
'''
plt.plot(rawx)
plt.plot(rawy)

plt.figure()
plt.plot(profx*2+0.0125)
plt.plot(gradx)

plt.figure()
plt.plot(profy*2+0.0125)
plt.plot(grady)

plt.show()
'''
