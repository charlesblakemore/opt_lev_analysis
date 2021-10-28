import os, math
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib

import matplotlib.patheffects as patheffects
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.lines

from bead_util import make_all_pardirs


class SymHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, \
                       width, height, fontsize, trans):
        xx = 0.7*height
        return super(SymHandler, self)\
                    .create_artists(legend, orig_handle, xdescent, \
                                    xx, width, height, fontsize, trans)


# matplotlib.rcParams.update({'font.size': 14})
# matplotlib.rcParams.update({'font.size': 18})
plt.rcParams['font.size'] = 18

annotate = False
annotate_bbox = {'boxstyle': 'round,pad=0.1', \
                 'fc': 'none', \
                 'ec': 'none', \
                 'alpha': 0.75}

plot_theory = False

microspheres_color = (0.495, 0.012, 0.658, 1.0)

plot_projection = True
projection_paths = [ \
                    '/data/new_trap_processed/limits/20200320_noise-limit_discovery.p', \
                    # '/data/new_trap_processed/limits/20200320_noise-limit_discovery_closer.p', \
                    '/data/new_trap_processed/limits/20200320_noise-limit_discovery_closer_scaled.p' \
                   ]
### Factors to turn limit/sensitivity from 10^4 second noise dataset 
### into a limit/sensitivity for a 10^5 second dataset, assuming 
### 1/sqrt(T) integration for the first projection, and then for a
### 30 day integration with a closer "fake position" of the attractor
projection_facs = [ \
                   np.sqrt(10000.0 / 100000.0), \
                   np.sqrt(10000.0 / (30.0 * 24.0 * 3600.0)), \
                  ]
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

projection_alphalim = [ \
                       (-np.inf, 1e35), \
                       (-np.inf, np.inf) \
                      ]


axis_scale = (1e6, '$\\mu$m')
# axis_scale = (1.0, 'm')

xlim = (1e-9, 1e-3) # limit in meter
ylim = (1e-3, 1.1e23)

plot_unity_alpha_line = True

text_alpha = 1.0
line_alpha = 1.0

theory_linewidth = 1
data_linewidth = 2


savefig = False
figbase = '/home/cblakemore/plots/for_alexf'
figname = 'alpha_lambda_exclusions_with_projections.svg'
fig_path = os.path.join(figbase, figname)






###########################################################################
###########################################################################
###########################################################################


fig, ax = plt.subplots(figsize=(8.5,6))
# fig, ax = plt.subplots(figsize=(10,6))


## Projections

if plot_projection:

    for i, fac in enumerate(projection_facs):
        projection_data = pickle.load( open(projection_paths[i], 'rb') )
        label = projection_labels[i]
        ls = projection_linestyles[i]
        color = projection_colors[i]

        lambdas = projection_data['pos_limit'][0]*1e6
        limit = np.max( np.abs(np.stack((projection_data['pos_limit'][1], \
                                         projection_data['neg_limit'][1]))), axis=0)
        limit *= fac

        label = projection_labels[i]
        ls = projection_linestyles[i]
        color = projection_colors[i]
        alphalim = projection_alphalim[i]

        if len(alphalim):
            inds = (limit > alphalim[0]) * (limit < alphalim[1])
        else:
            inds = limit > -np.inf

        ax.loglog(lambdas[inds], limit[inds], label=label, \
                  color=color, lw=4, ls=ls, zorder=5)



if plot_projection:
    ax.legend(handler_map={matplotlib.lines.Line2D: SymHandler()},\
              loc='upper right', fontsize=14, framealpha=1, \
              handleheight=2.25, labelspacing=0.05, ncol=1).set_zorder(100)




if plot_unity_alpha_line:
    ax.axhline(1.0, color='k', lw=2, ls='--', zorder=2)







#prev meas
cmeas = np.loadtxt('prev_meas/master_all.txt',delimiter=",",skiprows=1)
#plt.fill_between(cmeas[:,0]*axis_scale[0],cmeas[:,1],1e20,color=[135./256,205./256,250/256.])
ax.fill_between(cmeas[:,0]*axis_scale[0],cmeas[:,1],1e30,color='k',alpha=0.2, zorder=2)
# ax.text(45, 3.5e6, 'Excluded by\nexperiments', \
#         horizontalalignment='center', \
#         verticalalignment='center', \
#         multialignment='center')

ax.text(100, 1.0e16, 'EXCLUDED BY\nEXPERIMENTS', \
        ha='center', va='center', ma='center', fontsize=14)



cmeas = np.loadtxt('prev_meas/blakemore_prd_104_l061101_2021.txt',delimiter=",",skiprows=1)
# inds = cmeas[:,1] < 1e17
inds = cmeas[:,1] < 1e30
ax.loglog(cmeas[:,0][inds]*axis_scale[0], cmeas[:,1][inds], ls='-', color=microspheres_color, \
          alpha=line_alpha,linewidth=data_linewidth, zorder=4)
ax.fill_between(cmeas[:,0][inds]*axis_scale[0], cmeas[:,1][inds], 1e30,\
                color=microspheres_color, alpha=0.2, zorder=3)
if annotate:
    ax.annotate('', zorder=4,\
                xy=(2.5e-6, 3.0e10), xytext=(7.5e-6, 3e13), \
                arrowprops={'width': 1, 'headwidth': 3, 'headlength': 3, \
                            'shrink': 0.05, 'color': microspheres_color, \
                            'alpha': text_alpha*0.7})
    ax.annotate('Blakemore et al.\nPRD 104 L061101 (2021)', \
                xy=(2.5e-6, 3.0e10), xytext=(7.5e-6, 3e13), \
                ha='center', va='bottom', fontsize=10, fontweight='bold',\
                zorder=4, alpha=text_alpha, color=microspheres_color, \
                bbox=annotate_bbox)








# cmeas = np.loadtxt('prev_meas/sushkov_prl_107_171101_2011.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'k-',alpha=line_alpha,linewidth=data_linewidth, zorder=4)
# if annotate:
#     ax.annotate('', zorder=4,\
#                 xy=(1.8e-6, 1.9e8), xytext=(1.17e-5, 1e13), \
#                 arrowprops={'width': 1, 'headwidth': 3, 'headlength': 3, \
#                             'shrink': 0.05, 'color': 'k', \
#                             'alpha': text_alpha*0.7})
#     ax.annotate('Sushkov et al.\nPRL 107 171101 (2011)', \
#                 xy=(1.8e-6, 1.9e8), xytext=(1.17e-5, 1e13), \
#                 ha='center', va='bottom', fontsize=10, fontweight='bold',\
#                 zorder=4, alpha=text_alpha, color='k', \
#                 bbox=annotate_bbox)


# cmeas = np.loadtxt('prev_meas/geraci_prd_78_022002_2008.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'k',alpha=line_alpha,linewidth=data_linewidth, zorder=4)
# if annotate:
#     ax.annotate('', zorder=4,\
#                 xy=(3.0e-5, 300), xytext=(1.0e-6, 1560), \
#                 arrowprops={'width': 1, 'headwidth': 3, 'headlength': 3, \
#                             'shrink': 0.05, 'color': 'k', \
#                             'alpha': text_alpha*0.7})
#     ax.annotate('Geraci et al.\nPRD 78 022002 (2008)', \
#                 xy=(3.0e-5, 300), xytext=(1.5e-7, 1560), \
#                 ha='center', va='center', fontsize=10, fontweight='bold',\
#                 zorder=4, alpha=text_alpha, color='k', \
#                 bbox=annotate_bbox)


# color = 'C2'
color = 'k'
cmeas = np.loadtxt('prev_meas/lee_prl_124_101101_2020.txt',delimiter=",",skiprows=1)
ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],color,alpha=line_alpha,linewidth=data_linewidth, zorder=4)
if annotate:
    ax.annotate('', zorder=4,\
                xy=(8.6e-6, 11000), xytext=(1e-6, 1e3), \
                arrowprops={'width': 1, 'headwidth': 3, 'headlength': 3, \
                            'shrink': 0.05, 'color': color, \
                            'alpha': text_alpha*0.7})
    ax.annotate('Lee et al.\nPRL 124 101101 (2020)', \
                xy=(8.6e-6, 11000), xytext=(2.9e-7, 1e3), \
                ha='center', va='center', fontsize=10, fontweight='bold',\
                zorder=4, alpha=text_alpha, color=color, \
                bbox=annotate_bbox)


# color = 'C0'
color = 'k'
cmeas = np.loadtxt('prev_meas/chen_prl_116_221102_2016.txt',delimiter=",",skiprows=0)
ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],color,alpha=line_alpha,linewidth=data_linewidth, zorder=4)
if annotate:
    ax.annotate('', zorder=4,\
                xy=(2.1e-7, 1e9), xytext=(7.9e-8, 2e8), \
                arrowprops={'width': 1, 'headwidth': 3, 'headlength': 3, \
                            'shrink': 0.05, 'color': color, \
                            'alpha': text_alpha*0.7})
    ax.annotate('Chen et al.\nPRL 116 221102 (2016)', \
                xy=(2.1e-7, 1e9), xytext=(2.0e-8, 2e8), \
                ha='center', va='center', fontsize=10, fontweight='bold',\
                zorder=4, alpha=text_alpha, color=color, \
                bbox=annotate_bbox)


# cmeas = np.loadtxt('prev_meas/decca_prl_94_240401_2005.txt',delimiter=",",skiprows=1)
# ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'k',alpha=line_alpha,linewidth=data_linewidth, zorder=4)
# if annotate:
#     ax.annotate('', zorder=4,\
#                 xy=(9.9e-8, 6.9e12), xytext=(2.8e-8, 1e11), \
#                 arrowprops={'width': 1, 'headwidth': 3, 'headlength': 3, \
#                             'shrink': 0.05, 'color': 'k', \
#                             'alpha': text_alpha*0.7})
#     ax.annotate('Decca et al.\nPRL 94 240401 (2005)', \
#                 xy=(9.9e-8, 6.9e12), xytext=(4.0e-9, 1e11), \
#                 ha='center', va='center', fontsize=10, fontweight='bold',\
#                 zorder=4, alpha=text_alpha, color='k', \
#                 bbox=annotate_bbox)


# color = 'C0'
color = 'k'
cmeas = np.loadtxt('prev_meas/klimchitskaya_eurPhysJC_77_315_2017.txt',delimiter=",",skiprows=0)
ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],color,alpha=line_alpha,linewidth=data_linewidth, zorder=4)
if annotate:
    ax.annotate('', zorder=4,\
                xy=(2.1e-8, 2.5e16), xytext=(8e-9, 1e13), \
                arrowprops={'width': 1, 'headwidth': 3, 'headlength': 3, \
                            'shrink': 0.05, 'color': color, \
                            'alpha': text_alpha*0.7})
    ax.annotate('Klimchitskaya et al.\nEPJ C 77 315 (2017)', \
                xy=(2.1e-8, 2.5e16), xytext=(8e-9, 1e13), \
                ha='center', va='top', fontsize=10, fontweight='bold',\
                zorder=4, alpha=text_alpha, color=color, \
                bbox=annotate_bbox)


# color = 'C1'
color = 'k'
cmeas = np.loadtxt('prev_meas/leeb_prl_68_1472_1992b.txt',delimiter=",",skiprows=0)
inds = cmeas[:,0] < 2e-7
ax.loglog(cmeas[:,0][inds]*axis_scale[0],cmeas[:,1][inds],color,alpha=line_alpha,\
          linewidth=data_linewidth, zorder=4)
if annotate:
    ax.annotate('', zorder=4,\
                xy=(7.0e-8, 8e16), xytext=(6.0e-8, 1.0e20), \
                arrowprops={'width': 1, 'headwidth': 3, 'headlength': 3, \
                            'shrink': 0.05, 'color': color, \
                            'alpha': text_alpha*0.7})
    ax.annotate('Leeb et al.\nPRL 68 1472 (1992)', \
                xy=(7.0e-8, 8e16), xytext=(6.0e-8, 1.0e20), \
                ha='center', va='bottom', fontsize=10, fontweight='bold',\
                zorder=4, alpha=text_alpha, color=color, \
                bbox=annotate_bbox)


# color = 'C1'
color = 'k'
cmeas = np.loadtxt('prev_meas/nesvizhevsky_prd_77_034020_2008b.txt',delimiter=",",skiprows=0)
ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],color,alpha=line_alpha,linewidth=data_linewidth, zorder=4)
if annotate:
    ax.annotate('', zorder=4,\
                xy=(1.1e-9, 5e20), xytext=(4e-9, 2.5e17), \
                arrowprops={'width': 1, 'headwidth': 3, 'headlength': 3, \
                            'shrink': 0.05, 'color': color, \
                            'alpha': text_alpha*0.7})
    ax.annotate('Nesvizhevsky et al.\nPRD 77 034020 (2008)', \
                xy=(1.1e-9, 5e20), xytext=(4e-9, 2.5e17), \
                ha='center', va='top', fontsize=10, fontweight='bold',\
                zorder=4, alpha=text_alpha, color=color, \
                bbox=annotate_bbox)


# cmeas = np.loadtxt('prev_meas/kamiya_prl_114_161101_2015.txt',delimiter=",",skiprows=0)
# ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'k',alpha=line_alpha,linewidth=data_linewidth, zorder=4)
# if annotate:
#     ax.annotate('', zorder=4,\
#                 xy=(2.5e-9, 4.9e21), xytext=(3.3e-9, 2e24), \
#                 arrowprops={'width': 1, 'headwidth': 3, 'headlength': 3, \
#                             'shrink': 0.05, 'color': 'k', \
#                             'alpha': text_alpha*0.7})
#     ax.annotate('Kamiya et al.\nPRL 114 161101 (2015)', \
#                 xy=(2.5e-9, 4.9e21), xytext=(3.3e-9, 2e24), \
#                 ha='center', va='bottom', fontsize=10, fontweight='bold',\
#                 zorder=4, alpha=text_alpha, color='k', \
#                 bbox=annotate_bbox)


# cmeas = np.loadtxt('prev_meas/nesvizhevsky_prd_77_034020_2008a.txt',delimiter=",",skiprows=0)
# ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'k',alpha=line_alpha,linewidth=data_linewidth, zorder=4)


# cmeas = np.loadtxt('prev_meas/pokotilovski_physAtomNucl_69_924_2006.txt',delimiter=",",skiprows=0)
# ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'y',alpha=line_alpha,linewidth=data_linewidth, zorder=4)


# cmeas = np.loadtxt('prev_meas/haddock_prd_97_062002_2018.txt',delimiter=",",skiprows=0)
# ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'b',alpha=line_alpha,linewidth=data_linewidth, zorder=4)


# cmeas = np.loadtxt('prev_meas/voronin_jetpl_107_1_2018.txt',delimiter=",",skiprows=0)
# ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'k',alpha=line_alpha,linewidth=data_linewidth, zorder=4)


# cmeas = np.loadtxt('prev_meas/leeb_prl_68_1472_1992a.txt',delimiter=",",skiprows=0)
# ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'k',alpha=line_alpha,linewidth=data_linewidth, zorder=4)


cmeas = np.loadtxt('prev_meas/yang_prl_108_081101_2012.txt',delimiter=",",skiprows=1)
ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'k',alpha=line_alpha,linewidth=data_linewidth, zorder=4)
if annotate:
    if ref:
        ax.text(140, 2.5e-1, 'PRL 108, 081101 (2012)', rotation=-33, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)
    if institute:
        ax.text(140, 2.0e-1, 'HUST (2012)', rotation=-34, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)

cmeas = np.loadtxt('prev_meas/kapner_prl_98_021101_2007.txt',delimiter=",",skiprows=1)
ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'k',alpha=line_alpha,linewidth=data_linewidth, zorder=4)

cmeas = np.loadtxt('prev_meas/tan_prl_124_051301_2020.txt',delimiter=",",skiprows=1)
ax.loglog(cmeas[:,0]*axis_scale[0],cmeas[:,1],'k',alpha=line_alpha,linewidth=data_linewidth, zorder=4)


if plot_theory:

    cdat = np.loadtxt("theory/andy_yuk.txt",delimiter=",",skiprows=1)
    ax.loglog(cdat[:,0]*axis_scale[0],cdat[:,1],'--',linewidth=theory_linewidth, #color=[0.25,0.25,0.25], \
               color='C0', zorder=1)
    if annotate_theory:
        ax.text(0.3, 1.5e5, 'Yukawa messengers', fontsize=12,\
                 horizontalalignment='center', \
                 verticalalignment='center', color='C0')

    cdat = np.loadtxt("theory/andy_dil.txt",delimiter=",",skiprows=1)
    ax.loglog(cdat[:,0]*axis_scale[0],cdat[:,1],'--',linewidth=theory_linewidth, #color=[0.25,0.25,0.25], \
               color='C1', zorder=1)
    if annotate_theory:
        ax.text(0.3, 1e4, 'Dilaton', fontsize=12, \
                 horizontalalignment='center', \
                 verticalalignment='center', color='C1')


    # moduli

    cdat_m1l = np.loadtxt("theory/andy_mod1_low.txt",delimiter=",",skiprows=1)
    cdat_m1h = np.loadtxt("theory/andy_mod1_high.txt",delimiter=",",skiprows=1)
    hh_m1h = 30408*np.ones_like(cdat_m1l[:,0])##np.interp(cdat_m1l[:,0], cdat_m1h[:,0], cdat_m1h[:,1])
    #plt.fill_between(cdat_m1l[:,0]*axis_scale[0], cdat_m1l[:,1], hh_m1h, color=[1,0.92,0.92])

    cdat_m2l = np.loadtxt("theory/andy_mod2_low.txt",delimiter=",",skiprows=1)
    cdat_m2h = np.loadtxt("theory/andy_mod2_high.txt",delimiter=",",skiprows=1)
    hh_m2h = np.interp(cdat_m2l[:,0], cdat_m2h[:,0], cdat_m2h[:,1])
    #plt.fill_between(cdat_m2l[:,0]*axis_scale[0], cdat_m2l[:,1], hh_m2h, color=[1,0.92,0.92])


    ax.loglog(cdat_m1l[:,0]*axis_scale[0], cdat_m1l[:,1],':',linewidth=theory_linewidth,# color=[0.25,0.25,0.25], \
               zorder=1, color='C2')
    ax.loglog(cdat_m1l[:,0]*axis_scale[0], hh_m1h,':',linewidth=theory_linewidth,# color=[0.25,0.25,0.25], \
               zorder=1, color='C2')
    if annotate_theory:
        ax.annotate('', xy=(2, 3.5e4), xycoords='data', xytext=(2, 0.3e1), textcoords='data', \
                     arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', color='C2'))
        ax.text(2, 0.2e3, 'gluon\nmodulus', fontsize=9, color='C2', \
                 horizontalalignment='center', verticalalignment='center', \
                 multialignment='center', bbox=dict(fc='w', ec='w'))




    ax.loglog(np.append(cdat_m2l[:,0][::-1], cdat_m2l[:,0][0]*10.0)*axis_scale[0], \
               np.append(cdat_m2l[:,1][::-1], cdat_m2l[:,1][0]), \
               ':',linewidth=theory_linewidth, #color=[0.25,0.25,0.25], \
               zorder=1, color='C3')

    ax.loglog(cdat_m2l[:,0]*axis_scale[0], hh_m2h,':',linewidth=theory_linewidth, #color=[0.25,0.25,0.25], \
               zorder=1, color='C3')
    if annotate_theory:
        ax.annotate('', xy=(0.15, 2.7e3), xycoords='data', xytext=(0.15, 0.15), textcoords='data', \
                     arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', color='C3'))
        ax.text(0.15, 0.2e2, 'heavy q\nmoduli', fontsize=9, color='C3', \
                 horizontalalignment='center', verticalalignment='center', multialignment='center', \
                 bbox=dict(fc='w', ec='w'))



    # Gauge bosons

    cdat_g1l = np.loadtxt("theory/andy_gauge_low.txt",delimiter=",",skiprows=1)
    cdat_g1h = np.loadtxt("theory/andy_gauge_high.txt",delimiter=",",skiprows=1)
    hh_g1h = np.interp(cdat_g1l[:,0], cdat_g1h[:,0], cdat_g1h[:,1])
    #plt.fill_between(cdat_g1l[:,0]*axis_scale[0], cdat_g1l[:,1], hh_g1h, color=[0.85,1,0.85])

    cdat_g2l = np.loadtxt("theory/andy_gauge2_low.txt",delimiter=",",skiprows=1)
    cdat_g2h = np.loadtxt("theory/andy_gauge2_high.txt",delimiter=",",skiprows=1)
    hh_g2h = np.interp(cdat_g2l[:,0], cdat_g2h[:,0], cdat_g2h[:,1])
    #plt.fill_between(cdat_g2l[:,0]*axis_scale[0], cdat_g2l[:,1], hh_g2h, color=[0.75,1,0.75])

    ## These two are gauaged baryons, Ithink

    #plt.loglog(cdat_g1l[:,0]*axis_scale[0], cdat_g1l[:,1],':',linewidth=theory_linewidth, #color=[0.25,0.25,0.25], \
    #           zorder=1, color='C4')
    #plt.loglog(cdat_g1l[:,0]*axis_scale[0], hh_g1h,':',linewidth=theory_linewidth, #color=[0.25,0.25,0.25], \
    #           zorder=1, color='C4')


    #plt.loglog(cdat_g2l[:,0]*axis_scale[0], cdat_g2l[:,1],':',linewidth=theory_linewidth, #color=[0.25,0.25,0.25], \
    #           zorder=1, color='C5')
    #plt.loglog(cdat_g2l[:,0]*axis_scale[0], hh_g2h,':',linewidth=theory_linewidth, #color=[0.25,0.25,0.25], \
    #           zorder=1, color='C5')












ax.set_xlim(xlim[0]*axis_scale[0], xlim[1]*axis_scale[0])
ax.set_ylim(*ylim)
ax.plot(np.array(xlim),[1,1],'k--', zorder=1)
ax.grid()
# ax.yaxis.set_yticks(np.logspace(-2,12,8))
ax.set_xlabel('Length scale, $\\lambda$ [{:s}]'.format(axis_scale[1]))
ax.set_ylabel('Strength parameter, $\\alpha$')

# sax = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
# sax.set_xticks([1e-6, 1e-9])
# sax.set_xticks([], minor=True)
# sax.set_xticklabels(['$1~\\mu$m', '1 nm'])


fig.tight_layout()

# fig.text()
# fig.subplots_adjust(right=0.75)




if savefig:
    make_all_pardirs(fig_path)
    print('Saving figure to:')
    print('    {:s}'.format(fig_path))
    fig.savefig(fig_path)

plt.show()
