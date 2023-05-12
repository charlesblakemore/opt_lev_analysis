import os, math
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib

import matplotlib.patheffects as patheffects
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.lines

# matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.size': 20})


class SymHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle,xdescent, ydescent, width, height, fontsize, trans):
        xx = 0.7*height
        return super(SymHandler, self).create_artists(legend, orig_handle,xdescent, \
                                                        xx, width, height, fontsize, trans)


annotate = True
ref = False
institute = False
paper2021_bib_entries = True

annotate_theory = False
plot_theory = False

plot_limit = True
signed_limit = True
limit_ax = 2
limit_path = '/data/new_trap_processed/limits/20230330_limit_binning-10000.p'

old_limit_path = '/data/new_trap_processed/limits/20200320_limit_binning-10000.p'

plot_projection = True
projection_paths = [ \
                    '/data/new_trap_processed/limits/20200320_noise-limit_discovery.p', \
                    # '/data/new_trap_processed/limits/20200320_noise-limit_discovery_closer.p', \
                    '/data/new_trap_processed/limits/20200320_noise-limit_discovery_closer_scaled.p' \
                   ]
projection_facs = [ \
                   np.sqrt(10000.0 / 100000.0), \
                   np.sqrt(10000.0 / (30.0 * 24.0 * 3600.0)), \
                  ]
# projection_labels = [ \
#                      '$\\Delta x = 14~\\mu$, $\\Delta z = -15~\\mu$m, $T=10^{5}$ s\n$\\sigma = 6 \\times 10^{-17} ~ \\rm{N/rt(Hz)}$', \
#                      '$\\Delta x = 7.5~\\mu$, $\\Delta z = -5~\\mu$m, $T=1$ month\n$\\sigma = 1 \\times 10^{-18} ~ \\rm{N/rt(Hz)}$' \
#                     ]
projection_labels = [ \
                     'Present noise limit', \
                     'Projected next run' \
                    ]
projection_linestyles = [ \
                         '-.', \
                         (0, (3, 1.5, 1, 1.5, 1, 1.5)) \
                        ]
projection_colors = [ \
                     (0.973, 0.586, 0.252, 1.0), \
                     (0.495, 0.012, 0.658, 1.0) \
                    ]


xlim = (1, 100) # limit in micron
ylim = (1e-1, 1e11)
ylim = (1e2, 1e12)

text_alpha = 0.7
line_alpha = 0.8


savefig = True
fig_base = '/home/cblakemore/plots/20230330/mod_grav/comparisons/'
fig_path = os.path.join(fig_base, '20230330_vs_20200320_limits.svg')


###########################################################################
###########################################################################
###########################################################################


fig, ax = plt.subplots(figsize=(8,6))

## plot sensitivity compared to previous measurements

## theory models



if plot_limit:
    limit_data = pickle.load( open(limit_path, 'rb') )
    old_limit_data = pickle.load( open(old_limit_path, 'rb') )
    if signed_limit:
        ax.loglog(limit_data['pos_limit_by_axis'][0]*1e6, \
                  limit_data['pos_limit_by_axis'][limit_ax+1], \
                  label='$\\hat{\\alpha} > 0$', color='r', lw=2, ls='--', zorder=7)
        ax.loglog(old_limit_data['pos_limit_by_axis'][0]*1e6, \
                  old_limit_data['pos_limit_by_axis'][limit_ax+1], \
                  color='r', lw=2, ls='--', zorder=7, alpha=0.5)
        ax.fill_between(limit_data['pos_limit_by_axis'][0]*1e6, \
                        limit_data['pos_limit_by_axis'][limit_ax+1], \
                        10.0*ylim[1]*np.ones(len(limit_data['pos_limit_by_axis'][0])), \
                        fc='w', alpha=1.0, zorder=4, ec='none' )
        ax.fill_between(limit_data['pos_limit_by_axis'][0]*1e6, \
                        limit_data['pos_limit_by_axis'][limit_ax+1], \
                        10.0*ylim[1]*np.ones(len(limit_data['pos_limit_by_axis'][0])), \
                        fc='r', alpha=0.3, zorder=6, ec='none' )

        ax.loglog(limit_data['neg_limit_by_axis'][0]*1e6, \
                  limit_data['neg_limit_by_axis'][limit_ax+1], \
                  label='$\\hat{\\alpha} < 0$', color='b', lw=3, ls=':', zorder=7)
        ax.loglog(old_limit_data['neg_limit_by_axis'][0]*1e6, \
                  old_limit_data['neg_limit_by_axis'][limit_ax+1], \
                  color='b', lw=3, ls=':', zorder=7, alpha=0.5)
        ax.fill_between(limit_data['neg_limit_by_axis'][0]*1e6, \
                        limit_data['neg_limit_by_axis'][limit_ax+1], \
                        10.0*ylim[1]*np.ones(len(limit_data['neg_limit_by_axis'][0])), \
                        fc='w', alpha=1.0, zorder=4, ec='none' )
        ax.fill_between(limit_data['neg_limit_by_axis'][0]*1e6, \
                        limit_data['neg_limit_by_axis'][limit_ax+1], \
                        10.0*ylim[1]*np.ones(len(limit_data['neg_limit_by_axis'][0])), \
                        fc='b', alpha=0.3, zorder=6, ec='none' )
    else:
        ax.loglog(limit_data['limit'][0]*1e6, \
                  limit_data['limit'][limit_ax+1], \
                  label='20230330', color='r', lw=2, zorder=7)
        ax.loglog(old_limit_data['limit'][0]*1e6, \
                  old_limit_data['limit'][limit_ax+1], \
                  label='20200320', color='r', lw=2, zorder=7, alpha=0.5)
        ax.fill_between(limit_data['limit'][0]*1e6, \
                        limit_data['limit'][limit_ax+1], \
                        10.0*ylim[1]*np.ones(len(limit_data['neg_limit_by_axis'][0])), \
                        fc='w', alpha=1.0, zorder=4, ec='none' )
        ax.fill_between(limit_data['limit'][0]*1e6, \
                        limit_data['limit'][limit_ax+1], \
                        10.0*ylim[1]*np.ones(len(limit_data['limit'][0])), \
                        color='r', alpha=0.5, zorder=6 )


## Projections

if plot_projection:

    for i, fac in enumerate(projection_facs):
        projection_data = pickle.load( open(projection_paths[i], 'rb') )
        label = projection_labels[i]
        ls = projection_linestyles[i]
        color = projection_colors[i]

        lambdas = projection_data['pos_limit'][0]*1e6
        np.testing.assert_array_equal(lambdas, projection_data['neg_limit'][0]*1e6, \
                                      err_msg='Lambdas for pos/neg limits differ')

        limit = fac * np.max( np.abs(np.stack((projection_data['pos_limit'][1], \
                                               projection_data['neg_limit'][1]))), axis=0)

        # ax.loglog(projection_data['pos_limit'][0]*1e6, \
        #           projection_data['pos_limit'][1]*fac, \
        #           label=label, color=color, lw=2, ls=ls, zorder=1)

        ax.loglog(lambdas, limit, label=label, color=color, lw=3, ls=ls, zorder=5)



if plot_limit or plot_projection:
    ax.legend(handler_map={matplotlib.lines.Line2D: SymHandler()},\
              loc='upper right', fontsize=14, framealpha=1, \
              handleheight=2.25, labelspacing=0.05, ncol=2).set_zorder(100)








# moduli

cdat_m1l = np.loadtxt("theory/andy_mod1_low.txt",delimiter=",",skiprows=1)
cdat_m1h = np.loadtxt("theory/andy_mod1_high.txt",delimiter=",",skiprows=1)
hh_m1h = 30408*np.ones_like(cdat_m1l[:,0])##np.interp(cdat_m1l[:,0], cdat_m1h[:,0], cdat_m1h[:,1])
#plt.fill_between(cdat_m1l[:,0]*1e6, cdat_m1l[:,1], hh_m1h, color=[1,0.92,0.92])

cdat_m2l = np.loadtxt("theory/andy_mod2_low.txt",delimiter=",",skiprows=1)
cdat_m2h = np.loadtxt("theory/andy_mod2_high.txt",delimiter=",",skiprows=1)
hh_m2h = np.interp(cdat_m2l[:,0], cdat_m2h[:,0], cdat_m2h[:,1])
#plt.fill_between(cdat_m2l[:,0]*1e6, cdat_m2l[:,1], hh_m2h, color=[1,0.92,0.92])


# Gauge bosons

cdat_g1l = np.loadtxt("theory/andy_gauge_low.txt",delimiter=",",skiprows=1)
cdat_g1h = np.loadtxt("theory/andy_gauge_high.txt",delimiter=",",skiprows=1)
hh_g1h = np.interp(cdat_g1l[:,0], cdat_g1h[:,0], cdat_g1h[:,1])
#plt.fill_between(cdat_g1l[:,0]*1e6, cdat_g1l[:,1], hh_g1h, color=[0.85,1,0.85])

cdat_g2l = np.loadtxt("theory/andy_gauge2_low.txt",delimiter=",",skiprows=1)
cdat_g2h = np.loadtxt("theory/andy_gauge2_high.txt",delimiter=",",skiprows=1)
hh_g2h = np.interp(cdat_g2l[:,0], cdat_g2h[:,0], cdat_g2h[:,1])
#plt.fill_between(cdat_g2l[:,0]*1e6, cdat_g2l[:,1], hh_g2h, color=[0.75,1,0.75])







#prev meas
cmeas = np.loadtxt('prev_meas/master_all.txt',delimiter=",",skiprows=1)
#plt.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color=[135./256,205./256,250/256.])
ax.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color='k',alpha=0.2, zorder=2)
# ax.text(45, 3.5e6, 'Excluded by\nexperiments', \
#         horizontalalignment='center', \
#         verticalalignment='center', \
#         multialignment='center')


# cmeas = np.loadtxt('prev_meas/decca_prl_94_240401_2005.txt',delimaiter=",",skiprows=1)
# ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1, zorder=3)
# if annotate:
#     if ref:
#         ax.text(2, 1e9, 'PRL 94, 240401 (2005)', rotation=-45, \
#                  horizontalalignment='center', \
#                  verticalalignment='center', fontsize=9)
#     if institute:
#         ax.text(2, 1e9, 'IUPUI (2005)', rotation=-45, \
#                  horizontalalignment='center', \
#                  verticalalignment='center', fontsize=12)

cmeas = np.loadtxt('prev_meas/sushkov_prl_107_171101_2011.txt',delimiter=",",skiprows=1)
ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k-',alpha=line_alpha,linewidth=1, zorder=3)
if annotate:
    if ref:
        ax.text(3.5, 1.5e8, 'PRL 107, 17110111 (2011)', rotation=-33, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)
    if institute:
        ax.text(2.0, 3.0e8, 'Yale (2011)', rotation=-32, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)
    if paper2021_bib_entries:
        ax.text(2.0, 6.0e8, '[19]', \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)


cmeas = np.loadtxt('prev_meas/geraci_prd_78_022002_2008.txt',delimiter=",",skiprows=1)
ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',alpha=line_alpha,linewidth=1, zorder=3)
if annotate:
    if ref:
        ax.text(15, 9e2, 'PRD 78, 022002 (2008)', rotation=-15, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)
    if institute:
        ax.text(30, 9e2, 'Stanford (2008)', rotation=-15, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)
    if paper2021_bib_entries:
        ax.text(20, 3.0e3, '[17]', \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)

# cmeas = np.loadtxt('prev_meas/kapner_prl_98_021101_2007.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1, zorder=3)
# if annotate:
#     if ref:
#         ax.text(140, 2.5e-1, 'PRL 98, 021101 (2007)', rotation=-33, \
#                  horizontalalignment='center', \
#                  verticalalignment='center', fontsize=9)
#     if institute:
#         ax.text(140, 2.0e-1, 'Washington (2007)', rotation=-34, \
#                  horizontalalignment='center', \
#                  verticalalignment='center', fontsize=9)

cmeas = np.loadtxt('prev_meas/lee_prl_124_101101_2020.txt',delimiter=",",skiprows=1)
ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',alpha=line_alpha,linewidth=1, zorder=3)
if annotate:
    if ref:
        ax.text(40, 5, 'PRL 124, 101101 (2020)', rotation=-33, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)
    if institute:
        # ax.text(40, 4, 'Washington (2020)', rotation=-38, \
        #          horizontalalignment='center', \
        #          verticalalignment='center', fontsize=12)
        ax.text(6.85, 2.0e4, 'U. Wash. (2020)', rotation=-58, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)
    if paper2021_bib_entries:
        ax.text(6.95, 1.0e4, '[15]', \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)


cmeas = np.loadtxt('prev_meas/chen_prl_116_221102_2016.txt',\
                   delimiter=",",skiprows=0)
ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',alpha=line_alpha,linewidth=1, zorder=3)
if annotate:
    if ref:
        ax.text(3, 1.2e6, 'PRL 116, 221102 (2016)', rotation=-20, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)
    if institute:
        ax.text(2.5, 1e6, 'IUPUI (2016)', rotation=-20, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)
    if paper2021_bib_entries:
        ax.text(2.5, 1.5e6, '$\\left[18\\right]$', \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=16, \
                 zorder=3, alpha=text_alpha)


# cmeas = np.loadtxt('prev_meas/yang_prl_108_081101_2012.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1, zorder=3)
# if annotate:
#     if ref:
#         ax.text(140, 2.5e-1, 'PRL 108, 081101 (2012)', rotation=-33, \
#                  horizontalalignment='center', \
#                  verticalalignment='center', fontsize=9)
#     if institute:
#         ax.text(140, 2.0e-1, 'HUST (2012)', rotation=-34, \
#                  horizontalalignment='center', \
#                  verticalalignment='center', fontsize=9)





if plot_theory:

    cdat = np.loadtxt("theory/andy_yuk.txt",delimiter=",",skiprows=1)
    ax.loglog(cdat[:,0]*1e6,cdat[:,1],'--',linewidth=1, #color=[0.25,0.25,0.25], \
               color='C0', zorder=1)
    if annotate_theory:
        ax.text(0.3, 1.5e5, 'Yukawa messengers', fontsize=12,\
                 horizontalalignment='center', \
                 verticalalignment='center', color='C0')

    cdat = np.loadtxt("theory/andy_dil.txt",delimiter=",",skiprows=1)
    ax.loglog(cdat[:,0]*1e6,cdat[:,1],'--',linewidth=1, #color=[0.25,0.25,0.25], \
               color='C1', zorder=1)
    if annotate_theory:
        ax.text(0.3, 1e4, 'Dilaton', fontsize=12, \
                 horizontalalignment='center', \
                 verticalalignment='center', color='C1')




    ax.loglog(cdat_m1l[:,0]*1e6, cdat_m1l[:,1],':',linewidth=1,# color=[0.25,0.25,0.25], \
               zorder=1, color='C2')
    ax.loglog(cdat_m1l[:,0]*1e6, hh_m1h,':',linewidth=1,# color=[0.25,0.25,0.25], \
               zorder=1, color='C2')
    if annotate_theory:
        ax.annotate('', xy=(2, 3.5e4), xycoords='data', xytext=(2, 0.3e1), textcoords='data', \
                     arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', color='C2'))
        ax.text(2, 0.2e3, 'gluon\nmodulus', fontsize=9, color='C2', \
                 horizontalalignment='center', verticalalignment='center', \
                 multialignment='center', bbox=dict(fc='w', ec='w'))




    ax.loglog(np.append(cdat_m2l[:,0][::-1], cdat_m2l[:,0][0]*10.0)*1e6, \
               np.append(cdat_m2l[:,1][::-1], cdat_m2l[:,1][0]), \
               ':',linewidth=1, #color=[0.25,0.25,0.25], \
               zorder=1, color='C3')

    ax.loglog(cdat_m2l[:,0]*1e6, hh_m2h,':',linewidth=1, #color=[0.25,0.25,0.25], \
               zorder=1, color='C3')
    if annotate_theory:
        ax.annotate('', xy=(0.15, 2.7e3), xycoords='data', xytext=(0.15, 0.15), textcoords='data', \
                     arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', color='C3'))
        ax.text(0.15, 0.2e2, 'heavy q\nmoduli', fontsize=9, color='C3', \
                 horizontalalignment='center', verticalalignment='center', multialignment='center', \
                 bbox=dict(fc='w', ec='w'))


    ## These two are gauaged baryons, Ithink

    #plt.loglog(cdat_g1l[:,0]*1e6, cdat_g1l[:,1],':',linewidth=1, #color=[0.25,0.25,0.25], \
    #           zorder=1, color='C4')
    #plt.loglog(cdat_g1l[:,0]*1e6, hh_g1h,':',linewidth=1, #color=[0.25,0.25,0.25], \
    #           zorder=1, color='C4')


    #plt.loglog(cdat_g2l[:,0]*1e6, cdat_g2l[:,1],':',linewidth=1, #color=[0.25,0.25,0.25], \
    #           zorder=1, color='C5')
    #plt.loglog(cdat_g2l[:,0]*1e6, hh_g2h,':',linewidth=1, #color=[0.25,0.25,0.25], \
    #           zorder=1, color='C5')












#cdat = np.loadtxt("prev_meas/andy_init_sens.txt",delimiter=",",skiprows=1)
##plt.loglog(cdat[:,0]*1e6,cdat[:,1],'r',linewidth=2.5,label=lab)
#cdat = np.loadtxt("prev_meas/andy_fut_sens.txt",delimiter=",",skiprows=1)
#plt.loglog(cdat[:,0]*1e6,cdat[:,1],'r--',linewidth=2.5,label=lab)
#cdat = np.loadtxt("prev_meas/andy_matfut_sens.txt",delimiter=",",skiprows=1)
#plt.loglog(cdat[:,0]*1e6,cdat[:,1],'r:',linewidth=2.5,label=lab)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.plot(np.array(xlim),[1,1],'k--', zorder=1)
ax.grid()
# ax.yaxis.set_yticks(np.logspace(-2,12,8))
ax.set_xlabel('Length scale, $\\lambda$ [$\\mu$m]')
ax.set_ylabel('Strength parameter, $\\alpha$')


fig.tight_layout()

if savefig:
    fig.savefig(fig_path)

plt.show()
