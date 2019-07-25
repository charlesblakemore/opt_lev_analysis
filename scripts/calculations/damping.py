import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

def linearBeta(R = 2.4e-6, T = 300,p = 2.6e-4, m0 = 2*2.3258671e-26):
#'''Calculation of translational damping coefficient. Defaults: radiusis 2.4 um, T = 300K, p = 10 pascal (0.1 mbar), m0 = 2*~2.32e-26 (N2)'''
	vt = np.sqrt(constants.k*T/m0)
	beta = np.pi*R**2*(128/(9*np.pi))**(1/2)*(1+(np.pi/8))

	return beta*p/vt

def rotBeta(ms_mass = 85.e-15,rho = 1.550e3, T = 300., p = 2.6e-4, m0 = 2*2.324e-26):
	vt = np.sqrt(constants.k*T/m0)
	
	ms_radius = (0.75 * ms_mass/(np.pi*rho))**(1./3.)
	
	#print(ms_radius)

	beta = (p/vt) * np.pi * ms_radius**4. * (32./(9.*np.pi))**(1./2.)

	scaled_beta = beta/( p * np.sqrt(m0) )

	return scaled_beta

if __name__ == "__main__":
	print(rotBeta())
