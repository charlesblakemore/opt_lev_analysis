import os, math
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib

# matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.size': 16})





xlim = (5e-8, 1e15) # limit in meter
ylim = (1e-12, 1e12)


savefig = True
fig_path = '/home/cblakemore/plots/thesis/alpha_lambda_full_range.svg'


###########################################################################
###########################################################################
###########################################################################


fig, ax = plt.subplots(figsize=(8,6))

## plot sensitivity compared to previous measurements

## theory models










#prev meas
cmeas = np.loadtxt('prev_meas/master_all.txt',delimiter=",",skiprows=1)
#plt.fill_between(cmeas[:,0],cmeas[:,1],1e20,color=[135./256,205./256,250/256.])
ax.fill_between(cmeas[:,0],cmeas[:,1],1e20,color='k',alpha=0.5, zorder=2)
ax.loglog(cmeas[:,0],cmeas[:,1],color='k',linewidth=2, zorder=3)

# ax.text(5e3, 1e3, 'Excluded by\nexperiments', \
#         ha='center', va='center', ma='center')
ax.text(1e4, 1e3, 'EXCLUDED BY\nEXPERIMENTS', \
        ha='center', va='center', ma='center')

my_string = 'Adapted from:\n'
my_string += '$\\rm{\\it{Prog.\\,Part.\\,Nucl.\\,Phys.\\,}}$'
my_string += '$\\bf{62}$'
my_string += ' 102 (2009)'

ax.text(3e14, 2e11, my_string, ha='right', va='top', ma='right')

# r'$\text{Adapted from:\nProg. Part. Nucl. Phys. 62(1) 102-134 (2009)}$'



# ### Yale

# cmeas = np.loadtxt('prev_meas/sushkov_prl_107_171101_2011.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1, zorder=3)



# ### Stanford

# cmeas = np.loadtxt('prev_meas/geraci_prd_78_022002_2008.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1, zorder=3)




# ### IUPUI

# # cmeas = np.loadtxt('prev_meas/decca_prl_94_240401_2005.txt',delimiter=",",skiprows=1)
# # ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1, zorder=3)

# cmeas = np.loadtxt('prev_meas/decca_2014.txt',delimiter=",",skiprows=0)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1, zorder=3)



# ### HUST

# cmeas = np.loadtxt('prev_meas/yang_prl_108_081101_2012.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1, zorder=3)

# cmeas = np.loadtxt('prev_meas/tan_prl_124_051301_2020.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1, zorder=3)



# ### Washington

# cmeas = np.loadtxt('prev_meas/kapner_prl_98_021101_2007.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1, zorder=3)

# cmeas = np.loadtxt('prev_meas/lee_prl_124_101101_2020.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1, zorder=3)






# ### Long range




# cmeas = np.loadtxt('prev_meas/adelberger_review_2009_long_range.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k-',linewidth=1, zorder=3)

# cmeas = np.loadtxt('prev_meas/adelberger_review_2009_llr.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k-',linewidth=1, zorder=3)

# # cmeas = np.loadtxt('prev_meas/irvine_1980.txt',delimiter=",",skiprows=1)
# # ax.loglog(cmeas[:,0],cmeas[:,1],'k-',linewidth=1, zorder=3)

# cmeas = np.loadtxt('prev_meas/irvine_1985.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k-',linewidth=1, zorder=3)

# cmeas = np.loadtxt('prev_meas/maryland_1993.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0],cmeas[:,1],'k-',linewidth=1, zorder=3)






ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.plot(np.array(xlim),[1,1], 'k--', zorder=1, alpha=0.75)

# ax.yaxis.set_yticks(np.logspace(-2,12,8))
ax.set_xlabel('Length scale, $\\lambda$ [m]')
ax.set_ylabel('Strength parameter, $|\\alpha|$')



ax.text(6e12, 5e-11, 'Planetary', ha='center', va='top')
ax.annotate('', xycoords='data', textcoords='data',\
            xy=(5e11, 1.1e-8), xytext=(6e12,5e-11), \
            ha='center', va='bottom', \
            arrowprops=dict(width=1, headwidth=6, 
                            headlength=6, facecolor='black'))


ax.text(3.0e5, 1e-11, 'LLR', ha='right', va='center')
ax.annotate('', xycoords='data', textcoords='data',\
            xy=(6e7, 4.3e-10), xytext=(5.2e5,1e-11), \
            ha='center', va='bottom', \
            arrowprops=dict(width=1, headwidth=6, 
                            headlength=6, facecolor='black'))

ax.text(1.0e3, 6e-10, 'Satellites and\nGeophysics', ha='right', va='center')
ax.annotate('', xycoords='data', textcoords='data',\
            xy=(1.5e6, 1.0e-7), xytext=(2e3,6e-10), \
            ha='center', va='bottom', \
            arrowprops=dict(width=1, headwidth=6, 
                            headlength=6, facecolor='black'))
ax.annotate('', xycoords='data', textcoords='data',\
            xy=(1e4, 2.0e-4), xytext=(2e3,6e-10), \
            ha='center', va='bottom', \
            arrowprops=dict(width=1, headwidth=6, 
                            headlength=6, facecolor='black'))

ax.text(2.0e-5, 1e-6, 'Laboratory', ha='center', va='top')
ax.annotate('', xycoords='data', textcoords='data',\
            xy=(5.0e-3, 4.3e-4), xytext=(2.0e-5,1e-6), \
            ha='center', va='bottom', \
            arrowprops=dict(width=1, headwidth=6, 
                            headlength=6, facecolor='black'))
ax.annotate('', xycoords='data', textcoords='data',\
            xy=(7.0e-6, 1.0e3), xytext=(2.0e-5,1e-6), \
            ha='center', va='bottom', \
            arrowprops=dict(width=1, headwidth=6, 
                            headlength=6, facecolor='black'))

fig.tight_layout()

if savefig:
    fig.savefig(fig_path)

plt.show()
