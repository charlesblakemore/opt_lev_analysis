
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from build_yukfuncs import *
from scipy.optimize import curve_fit

z = np.zeros(1000)
y = np.zeros(1000)
x = np.linspace(5e-6, 5e-5, 1000)
pts = ptarr(x, y, z)

li = np.argmin(np.abs(lambdas - 25E-6))
yf = yukfuncs[0][48]
fx = yf(pts)

def ffun(x, a, b, c, d):
    return a*np.exp(-x/(d))/(b*x + c)

popt, pcov = curve_fit(ffun, x, fx, p0 = [10.**-24, 1., 0., 25E-6])


fx2 = ffun(x, *popt)

def ffun(x, a, b, c, d):
    return a*np.exp(-x/(d))/(b*x + c)


fx = yf(pts)
wi = max(fx)
popt, pcov = curve_fit(ffun, x, fx, p0 = [-10**-24, 1., 0., lambdas[0]])
plt.plot(x, fx, 'o')
plt.plot(x, ffun(x, *popt), 'o')
plt.show()


fgs_10um = []
popti = popt
dind = np.argmin(np.abs(10E-6-x))

for i, l in enumerate(lambdas):
    yf = yukfuncs[0][i]
    fx = yf(pts)
    wi = min(fx)
    popt, pcov = curve_fit(ffun, x, fx, p0 = [wi, 1., 0., lambdas[0]])
    popti = popt
    fg = np.gradient(ffun(x, *popt))/np.gradient(x)
    fgs_10um.append(fg[dind])
    #plt.plot(x, ffun(x, *popt)-fx, 'o')
    #plt.show()


def ffun2(x, a, b, c, d):
    return a + -b*np.exp(-(x/c)) + d*x 

p0 = [-50., 100., 15., 0.]
popt, pvoc = curve_fit(ffun2, np.log(lambdas), np.log(np.abs(fgs_10um)), p0 = p0)


plt.plot(np.log(lambdas), np.log(np.abs(fgs_10um)), 'o')
plt.plot(np.log(lambdas), ffun2(np.log(lambdas), *popt))
plt.show()


fgs_10um_2 = np.exp(ffun2(np.log(lambdas), *popt))


limfreq_10um = (1E-13*1E6*1E-7)/fgs_10um_2
plt.loglog(lambdas, limfreq)
plt.show()


limitdata_path = '/sensitivities/decca1_limits.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/sensitivities/decca2_limits.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'

alpha_plot_lims = (1, 10**13)
lambda_plot_lims = (10**(-7), 10**(-4))



fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)


ax.loglog(lambdas, limfreq, '--',           label=r"frequency measurement s=15$\mu$m $t_{int} = 10^{4}s$", linewidth=3, color='b')

ax.loglog(lambdas, limfreq_10um, '--',           label=r"frequency measurement s=10$\mu$m $t_{int} = 10^{4}s$", linewidth=3, color='c')

ax.loglog(limitdata[:,0], limitdata[:,1], '--',           label=limitlab, linewidth=3, color='r')
ax.loglog(limitdata2[:,0], limitdata2[:,1], '--',           label=limitlab2, linewidth=3, color='k')
ax.grid()


ax.set_xlabel('$\lambda$ [m]')
ax.set_ylabel('|$\\alpha$|')

ax.legend(numpoints=1, fontsize=9)


plt.tight_layout()


plt.show()

