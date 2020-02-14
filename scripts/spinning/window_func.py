import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab

def flattop(M):
	a0 = 0.21557895
	a1 = 0.41663158
	a2 = 0.277263158
	a3 = 0.083578947
	a4 = 0.006947368

	n = np.arange(0,M)
	return a0 - a1*np.cos(2*np.pi*n/(M-1)) + \
		a2*np.cos(4*np.pi*n/(M-1)) - \
		a3*np.cos(6*np.pi*n/(M-1)) + \
		a4*np.cos(8*np.pi*n/(M-1))
 

