import matplotlib.pyplot as plt
import numpy as np
import dill as pickle



sens_to_plot = ['20180314_grav_noshield_cant-0mV_allharm.npy', \
                '20180314_grav_shieldin-nofield_cant-0mV_allharm.npy', \
                '20180314_grav_shieldin-1V-1300Hz_cant-0mV_allharm.npy', \
                '20180314_grav_shieldin-2V-2200Hz_cant-0mV_allharm.npy']

labs = ['No Shield', 'Shield', 'Shield - 1300Hz', 'Shield - 2200Hz']


plot_just_current = False


sens_dat = []
for sens_file in sens_to_plot:
    lambdas, alphas, diagalphas = np.load('/sensitivities/' + sens_file)
    sens_dat.append(alphas)

alpha_plot_lims = (1000, 10**13)
lambda_plot_lims = (10**(-7), 10**(-4))

limitdata_path = '/sensitivities/decca1_limits.txt'
limitdata = np.loadtxt(limitdata_path, delimiter=',')
limitlab = 'No Decca 2'

limitdata_path2 = '/sensitivities/decca2_limits.txt'
limitdata2 = np.loadtxt(limitdata_path2, delimiter=',')
limitlab2 = 'With Decca 2'




fig, ax = plt.subplots(1,1,sharex='all',sharey='all',figsize=(5,5),dpi=150)

if not plot_just_current:
    for i, sens in enumerate(sens_dat):
        ax.loglog(lambdas, sens, linewidth=2, label=labs[i])

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
