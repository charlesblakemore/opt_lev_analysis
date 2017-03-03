import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as sp

nmc = 10000

out_list = []
for n in range(nmc):

    cdat = np.random.randn(2**11)

    xx = np.arange(2**11)
    cdat += 0.02*np.sin( 2*np.pi*(xx*0.1 + np.random.rand()) )

    cpsd, freqs = mlab.psd(cdat, Fs = 5000, NFFT = 2**11) 

    # plt.figure()
    # plt.loglog(freqs, cpsd)
    # plt.show()

    if(n == 0):
        tot_psd = cpsd
    else:
        tot_psd += cpsd

tot_psd = tot_psd/n

plt.figure()
plt.loglog(freqs, np.sqrt(tot_psd)*5000./2**11)
plt.show()

#hh, be = np.histogram( np.sqrt(out_list), bins=50)

#bc = be[:-1] + np.diff(be)/2.

#cc = sp.chi.pdf(bc, nmc)
#cc *= np.max(hh)/np.max(cc)

#plt.figure()
#plt.step(be[:-1], hh, where='post')
#plt.plot(bc, cc, 'r')
#plt.show()
