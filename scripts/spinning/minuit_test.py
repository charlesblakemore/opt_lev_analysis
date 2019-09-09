import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy import signal
import h5py, time
import sys
from iminuit import Minuit, describe
from pprint import pprint # we use this to pretty print some stuff later

np.random.seed(12345678)


f0 = 1e3 * np.pi

xdat = np.linspace(0,1000,1001)
ydat = f0 * np.exp(-1.0*xdat / 500.0) + 250.0 + 25.0*np.random.randn(len(xdat))
yerr = 5.0*np.random.randn(len(xdat)) + 25.0

npts = len(xdat)

# plt.plot(xdat, ydat)
# plt.show()



def fit_fun(x, a, b, c):
    first_term = f0 * np.exp( a * x) 
    second_term = (b / (-1.0*a)) * (1 - np.exp(a * x))
    return first_term + second_term + c


# definition of the cost function to minimize, examplary chisquare
def chisquare_1d(a, b, c):
    resid = ydat - fit_fun(xdat, a, b, c)
    return (1.0 / (npts - 1.0) ) * np.sum(resid**2 / yerr**2)

print chisquare_1d(-1.0 / 500, 250.0/500.0, 0)


m=Minuit(chisquare_1d, 
         a = -1.0 / 750.0, # set start parameter
         #limit_a= (limit_lower,limit_upper) # if you want to limit things
         #fix_a = "True", # you can also fix it
         b = 1000.0,
         c = 200.0,
         errordef = 1,
         print_level = 1)
m.migrad(ncall=500000)
#m.minos(), if you need fancy mapping




plt.plot(xdat, ydat)
plt.plot(xdat, fit_fun(xdat, m.values["a"], m.values["b"], m.values["c"]))
# plt.xlim(3500,4000)

plt.show()



m.draw_mnprofile('a', bound = 5, bins = 100)

plt.show()