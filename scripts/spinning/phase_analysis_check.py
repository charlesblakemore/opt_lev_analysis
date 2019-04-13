import numpy as np
import matplotlib.pyplot as plt

tf = 100.
ns = 100000
tarr = np.linspace(0, tf, ns)
fc = 10.
Amod = 0.01
fmod = 1.
phi_mod = Amod*np.sin(2.*np.pi*fmod*tarr)
sig = np.sin(2.*np.pi*fc*tarr + phi_mod)
sig = sig**2
fft = np.fft.rfft(sig)
freqs = np.fft.rfftfreq(ns, d = tf/ns)
plt.loglog(freqs, np.abs(fft))
plt.show()



