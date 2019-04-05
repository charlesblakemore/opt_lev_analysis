import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
import scipy.constants


matt = 7.E-11
sphi = 1.e-5
Q = 10.**4
d = 20.e-6
w0 = 10.**3

def dphi(Q, matt, d, lam, w0):
    num = scipy.constants.G*Q*matt*np.exp(-d/lam)*(d**2 + 2*d*lam + 2*lam**2)
    denom = w0**2*d**3*lam**2
    return num/denom




alpha_plot_lims = (1, 10**13)
lambda_plot_lims = (10**(-7), 10**(-4))

lam_plot_phis = np.linspace(lambda_plot_lims[0], lambda_plot_lims[-1], num = 10000)
alpha_plot_phis = sphi/dphi(Q, matt, d, lam_plot_phis, w0)
limitdata_path = '/sensitivities/decca1_limits.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/sensitivities/decca2_limits.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'




fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)


ax.loglog(lam_plot_phis, alpha_plot_phis, '--', \
          label="frequency measurement", linewidth=3, color='b')

ax.loglog(limitdata[:,0], limitdata[:,1], '--', \
          label=limitlab, linewidth=3, color='r')
ax.loglog(limitdata2[:,0], limitdata2[:,1], '--', \
          label=limitlab2, linewidth=3, color='k')
ax.grid()

ax.set_xlim(lambda_plot_lims[0], lambda_plot_lims[1])
ax.set_ylim(alpha_plot_lims[0], alpha_plot_lims[1])

ax.set_xlabel('$\lambda$ [m]')
ax.set_ylabel('|$\\alpha$|')

ax.legend(numpoints=1, fontsize=9)

#ax.set_title(figtitle)

plt.tight_layout()


plt.show()
