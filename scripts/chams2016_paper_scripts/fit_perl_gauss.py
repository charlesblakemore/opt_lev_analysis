import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = np.loadtxt("perl_gaussian.txt", delimiter=",", skiprows=1)

def gauss_fun(x,A,mu,sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

bp, bcov = opt.curve_fit(gauss_fun, data[:,0], data[:,1], p0=[1e6, 0, 0.05])

print bp[1] 

plt.figure()
plt.plot(data[:,0], data[:,1], 'ko')
xx = np.linspace(-0.3, 0.3, 1e3)
plt.plot(xx, gauss_fun(xx, bp[0], bp[1], bp[2]), 'r')
plt.show()
