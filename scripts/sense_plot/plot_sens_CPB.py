import os, math
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib

# matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.size': 16})





annotate = True
ref = False
institute = True

annotate_theory = False

plot_theory = False

plot_projections = False
all_three = False

plot_limit = False
signed_limit = True
limit_ax = 2
limit_path = '/data/new_trap_processed/limits/20200320_limit_binning-10000.p'

xlim = (1, 100) # limit in micron
ylim = (1e-1, 1e11)
ylim = (1e3, 1e11)


savefig = False
fig_path = '/home/cblakemore/plots/20200320/mod_grav/20200320_limit_binning-10000.svg'


###########################################################################
###########################################################################
###########################################################################


fig, ax = plt.subplots(figsize=(8,6))

## plot sensitivity compared to previous measurements

## theory models



if plot_limit:
    limit_data = pickle.load( open(limit_path, 'rb') )
    if signed_limit:
        ax.loglog(limit_data['pos_limit'][0]*1e6, limit_data['pos_limit'][limit_ax+1], \
                  label='This work: $\\hat{\\alpha} > 0$', color='r', lw=2, ls='--', zorder=6)
        ax.fill_between(limit_data['pos_limit'][0]*1e6, limit_data['pos_limit'][limit_ax+1], \
                        10.0*ylim[1]*np.ones(len(limit_data['pos_limit'][0])), \
                        color='r', alpha=0.25, zorder=5 )

        ax.loglog(limit_data['neg_limit'][0]*1e6, limit_data['neg_limit'][limit_ax+1], \
                  label='                 $\\hat{\\alpha} < 0$', color='r', lw=3, ls=':', zorder=6)
        ax.fill_between(limit_data['neg_limit'][0]*1e6, limit_data['neg_limit'][limit_ax+1], \
                        10.0*ylim[1]*np.ones(len(limit_data['neg_limit'][0])), \
                        color='r', alpha=0.25, zorder=5 )
    else:
        ax.loglog(limit_data['limit'][0]*1e6, limit_data['limit'][limit_ax+1], \
                  label='This work', color='r', lw=2, zorder=6)
        ax.fill_between(limit_data['limit'][0]*1e6, limit_data['limit'][limit_ax+1], \
                        10.0*ylim[1]*np.ones(len(limit_data['limit'][0])), \
                        color='r', alpha=0.5, zorder=5 )





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
ax.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color='k',alpha=0.5, zorder=2)
ax.text(50, 1e6, 'Excluded by\nexperiments', \
        horizontalalignment='center', \
        verticalalignment='center', \
        multialignment='center')

# cmeas = np.loadtxt('prev_meas/decca_prl_94_240401_2005.txt',delimiter=",",skiprows=1)
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
ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k-',linewidth=1, zorder=3)
if annotate:
    if ref:
        ax.text(3.5, 1.5e8, 'PRL 107, 17110111 (2011)', rotation=-37, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=12)
    if institute:
        ax.text(2.5, 1.2e8, 'Yale (2011)', rotation=-37, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=12)

cmeas = np.loadtxt('prev_meas/geraci_prd_78_022002_2008.txt',delimiter=",",skiprows=1)
ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1, zorder=3)
if annotate:
    if ref:
        ax.text(15, 9e2, 'PRD 78, 022002 (2008)', rotation=-40, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=12)
    if institute:
        ax.text(14, 6e3, 'Stanford (2008)', rotation=-40, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=12)

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
ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1, zorder=3)
if annotate:
    if ref:
        ax.text(40, 5, 'PRL 124, 101101 (2020)', rotation=-33, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=12)
    if institute:
        # ax.text(40, 4, 'Washington (2020)', rotation=-38, \
        #          horizontalalignment='center', \
        #          verticalalignment='center', fontsize=12)
        ax.text(7.2, 2.0e4, 'Washington (2020)', rotation=-63, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=12)


cmeas = np.loadtxt('prev_meas/chen_prl_116_221102_2016.txt',\
                   delimiter=",",skiprows=0)
ax.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1, zorder=3)
if annotate:
    if ref:
        ax.text(3, 1.2e6, 'PRL 116, 221102 (2016)', rotation=-25, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=12)
    if institute:
        ax.text(2.5, 8.5e5, 'IUPUI (2016)', rotation=-25, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=12)


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









## Projections

if plot_projections:

    proj_mid = np.loadtxt('projections/attractorv2_rbead3_8um_sep7_5um_noise3e-18NrtHz_int2e+05s.txt', \
                          delimiter=',')
    label_mid = '$d_{\\rm MS} = 7.56~\\mu$m, $\\Delta x=7.5~\\mu$m,\n' \
                  + '$\\sigma = 3 \\times 10^{-18}~\\rm{N/\\sqrt{Hz}}$, ' \
                  + '$t = 2 \\times 10^{5}~$s'
    ax.loglog(proj_mid[:,0]*1e6, proj_mid[:,1], '--', color='r', lw=3, #alpha=0.6, \
               label=label_mid, zorder=20)


    proj_good = np.loadtxt('projections/attractorv2_rbead7_5um_sep3_0um_noise7e-20NrtHz_int3e+06s.txt', \
                           delimiter=',')
    label_good = '$d_{\\rm MS} = 15.0~\\mu$m, $\\Delta x=3.0~\\mu$m,\n' \
                  + '$\\sigma = 7 \\times 10^{-20}~\\rm{N/\\sqrt{Hz}}$, ' \
                  + '$t = 3 \\times 10^{6}~$s'
    ax.loglog(proj_good[:,0]*1e6, proj_good[:,1], '--', color='b', lw=3, #alpha=0.6, \
               label=label_good, zorder=20)


if plot_limit or plot_projections:
    ax.legend(loc='upper right', fontsize=14, framealpha=1, labelspacing=1.0).set_zorder(100)






#cdat = np.loadtxt("prev_meas/andy_init_sens.txt",delimiter=",",skiprows=1)
##plt.loglog(cdat[:,0]*1e6,cdat[:,1],'r',linewidth=2.5,label=lab)
#cdat = np.loadtxt("prev_meas/andy_fut_sens.txt",delimiter=",",skiprows=1)
#plt.loglog(cdat[:,0]*1e6,cdat[:,1],'r--',linewidth=2.5,label=lab)
#cdat = np.loadtxt("prev_meas/andy_matfut_sens.txt",delimiter=",",skiprows=1)
#plt.loglog(cdat[:,0]*1e6,cdat[:,1],'r:',linewidth=2.5,label=lab)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.plot(np.array(xlim),[1,1],'k--', zorder=1)
# ax.yaxis.set_yticks(np.logspace(-2,12,8))
ax.set_xlabel('Length scale, $\\lambda$ [$\\mu$m]')
ax.set_ylabel('Strength parameter, $\\alpha$')


fig.tight_layout()

if savefig:
    fig.savefig(fig_path)

plt.show()
