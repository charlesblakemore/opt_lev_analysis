import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab
import bead_sim_funcs as bsfuncs
from scipy import constants

def whiteNoise(nsamp, Fs):
    '''Generates white noise with PSD of 1. rounds nsamp down to
       nearest even number necessary for irfft'''
    dt = 1./Fs
    nfft = int(np.ceil((nsamp)/2) + 1)
    re = np.random.randn(nfft)/np.sqrt(2)
    im = np.random.randn(nfft)/np.sqrt(2)
    fft = re + 1.j*im
    return np.fft.irfft(np.sqrt(nsamp)*fft/np.sqrt(2*dt))

def noise_test():
    '''plots noise with different parameters to test normalization'''
    noise0 = whiteNoise(10000, 1e-3)
    noise1 = whiteNoise(100000, 1e-4)
    noise2 = whiteNoise(1000000, 1e-5)
    psd0, freqs0  = matplotlib.mlab.psd(noise0, Fs = 1e3)
    psd1, freqs1  = matplotlib.mlab.psd(noise1, Fs = 1e4)
    psd2, freqs2  = matplotlib.mlab.psd(noise2, Fs = 1e5)
    plt.loglog(freqs0, psd0)
    plt.loglog(freqs1, psd1)
    plt.loglog(freqs2, psd2)
    plt.show()

def beta(R = 2.5e-6, T = 300., p = 2.6e-4, m0 = bsfuncs.mbead ):#2*2.3258671e-26):
    '''calculates torsional damping coefficient. Defaluts: sphere 
       radius is 2.5um, T 300K, gas pressure is 10^-5mbar (1mpascal) 
       Default gas mass is N2'''
    vt = np.sqrt(constants.k*T/m0)
    beta_geo = (vt/p)*np.pi*R**4*np.sqrt(32/(9*np.pi))
    return beta_geo

def Sf(beta, T):
    '''fluctuation dissipation theorem converts torsional damping to PSD'''
    return 4*constants.k*T*beta

def torqueNoise(nsamp, Fs, R = 2.5e-6, T = 300., p = 2.6e-4, \
        m0 = bsfuncs.mbead):#2*2.3258671e-26):
    '''generates torque noise realization for 1dof.'''
    b = beta(R = R, T = T, p = p, m0 = m0)
    S = Sf(b, T)
    return np.sqrt(S * 0.5 * Fs)*np.random.randn(nsamp) #0.5 *Fs is the max frequency in the S spectrum?
