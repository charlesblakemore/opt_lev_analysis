import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

fsamp = 1e3
t = np.linspace(0,100,100*fsamp)

def f1(x):
    return 2./(x+5)**2
def f2(x):
    return 0.01*np.ones_like(x)

f = f1

f_d = 5.
f_m = 3.
cant_pos = np.sin( 2*np.pi*f_d*t )
mod_pos = np.sin( 2*np.pi*f_m*t )

xvt = f( cant_pos )/(mod_pos*0.1 + 1)
xf = np.fft.rfft( xvt )
freqs = np.fft.rfftfreq( len(xvt) ) * fsamp

fvt = xvt * (mod_pos*0.5 + 1)
ff = np.fft.rfft( fvt )

## now cut to modulated frequencies
## make stupid comb
fcomb = []
for n in range(1,10):
    fcomb.append( np.argmin( np.abs( freqs - (n*f_m + f_d) ) ) )

print fcomb
fvf_filt = np.zeros_like(xf)+1e-15
for idx in fcomb:
    fvf_filt[idx] = xf[idx]
fvf_filt = xf

ft = np.fft.irfft( fvf_filt )
b,a = sig.butter( 3, np.array([f_m+f_d-2, f_m+f_d+2])/(2*fsamp), btype='bandpass')
ftf = sig.filtfilt( b, a, xvt )
print ftf


# plt.figure()
# plt.plot(t, xvt)
# plt.plot( t, cant_pos)
# plt.plot( t, mod_pos)

plt.figure()
#plt.loglog( freqs, np.abs(xf) )
plt.loglog( freqs, np.abs(fvf_filt) )
#plt.loglog( freqs, np.abs(ff) )

xbins = np.linspace(np.min(cant_pos), np.max(cant_pos), 1e2)
xcent = xbins[:-1] + np.diff(xbins)/2.0
mu, sg = np.zeros_like(xbins[:-1]), np.zeros_like(xbins[:-1])
for i,(xl,xh) in enumerate(zip(xbins[:-1], xbins[1:])):
    gpts = np.logical_and( cant_pos > xl, cant_pos <= xh )
    mu[i], sg[i] = np.mean( ft[gpts] ), np.std( ft[gpts] )/np.sqrt(np.sum(gpts))

# plt.figure()
# plt.plot( cant_pos, ftf )
# plt.errorbar(xcent, mu, yerr=sg, fmt='r')
#plt.plot(t, ft)
#plt.plot(t, ft)

plt.show()
