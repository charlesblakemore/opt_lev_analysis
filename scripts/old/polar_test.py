import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc('font', family='serif') 

def scatter_logpolar(ax, theta, r_, bullseye=0.3, **kwargs):
    min10 = np.log10(np.min(r_))
    max10 = np.log10(np.max(r_)*2)
    r = np.log10(r_) - min10 + bullseye
    #ax.scatter(theta, r, **kwargs)
    plt.errorbar( theta, r, xerr=0.1, yerr=0.1, fmt='ko')
    l = np.arange(np.floor(min10), max10)
    ax.set_rticks(l - min10 + bullseye) 
    ax.set_yticklabels(["1e%d" % x for x in l])
    ax.set_rlim(0, max10 - min10 + bullseye)
    ax.set_title('log-polar manual')
    return ax

fig = plt.figure()
ax = plt.subplot(111, polar=True )

scatter_logpolar( ax, [0.2*np.pi, 0.4*np.pi, -0.2*np.pi], [0.01, 0.05, 1e3] )
#plt.ylim([1e-3, 1e3])
plt.show()
