import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import os, sys

##### Simulation parameters

T = 50.
update_freq = 1000000.
Npoints = T * update_freq

sample_freq = 5000.
Nsamples = T * sample_freq


##### Harmonic Oscillator parameters

fx = 180.  # Hz
fy = 190.  # Hz
fz = 60.   # Hz

wx = 2. * np.pi * fx
wy = 2. * np.pi * fy
xz = 2. * np.pi * fz

rbead = 2.5e-6  # m
mbead = (4. / 3) * np.pi * (rbead)**3 * 2200  # kg

kx = mbead * wx**2
ky = mbead * wy**2
kz = mbead * wz**2

kmat = np.array([[kx, 0., 0.],
                 [0., ky, 0.],
                 [0., 0., kz]])

noiseRMS = 10 * 1e-15  # N

noise_drive = 
