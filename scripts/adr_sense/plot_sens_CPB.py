import os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

gap_list = [6e-6,6e-6,6e-6]
lam_list = np.logspace(-1.0,3.0,40)*1e-6
print lam_list

sens_vals = np.zeros((len(lam_list),len(gap_list)))

force_vals = np.zeros((len(lam_list),len(gap_list)))
force_vals_old = np.zeros((len(lam_list),len(gap_list)))


annotate = False
ref = False
institute = True

annotate_theory = False

plot_theory = False

plot_projections = False
all_three = False

plot_shield_no_shield = False

plot_freq_sensitivity = True

fig=plt.figure(88, dpi=200)
## plot sensitivity compared to previous measurements

## theory models




# moduli

cdat_m1l = np.loadtxt("prev_meas/andy_mod1_low.txt",delimiter=",",skiprows=1)
cdat_m1h = np.loadtxt("prev_meas/andy_mod1_high.txt",delimiter=",",skiprows=1)
hh_m1h = 30408*np.ones_like(cdat_m1l[:,0])##np.interp(cdat_m1l[:,0], cdat_m1h[:,0], cdat_m1h[:,1])
#plt.fill_between(cdat_m1l[:,0]*1e6, cdat_m1l[:,1], hh_m1h, color=[1,0.92,0.92])

cdat_m2l = np.loadtxt("prev_meas/andy_mod2_low.txt",delimiter=",",skiprows=1)
cdat_m2h = np.loadtxt("prev_meas/andy_mod2_high.txt",delimiter=",",skiprows=1)
hh_m2h = np.interp(cdat_m2l[:,0], cdat_m2h[:,0], cdat_m2h[:,1])
#plt.fill_between(cdat_m2l[:,0]*1e6, cdat_m2l[:,1], hh_m2h, color=[1,0.92,0.92])


# Gauge bosons

cdat_g1l = np.loadtxt("prev_meas/andy_gauge_low.txt",delimiter=",",skiprows=1)
cdat_g1h = np.loadtxt("prev_meas/andy_gauge_high.txt",delimiter=",",skiprows=1)
hh_g1h = np.interp(cdat_g1l[:,0], cdat_g1h[:,0], cdat_g1h[:,1])
#plt.fill_between(cdat_g1l[:,0]*1e6, cdat_g1l[:,1], hh_g1h, color=[0.85,1,0.85])

cdat_g2l = np.loadtxt("prev_meas/andy_gauge2_low.txt",delimiter=",",skiprows=1)
cdat_g2h = np.loadtxt("prev_meas/andy_gauge2_high.txt",delimiter=",",skiprows=1)
hh_g2h = np.interp(cdat_g2l[:,0], cdat_g2h[:,0], cdat_g2h[:,1])
#plt.fill_between(cdat_g2l[:,0]*1e6, cdat_g2l[:,1], hh_g2h, color=[0.75,1,0.75])







#prev meas
cmeas = np.loadtxt('prev_meas/master_ext.txt',delimiter=",",skiprows=1)
#plt.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color=[135./256,205./256,250/256.])
plt.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color=[255./256,246./256,143/256.], zorder=2)
plt.text(2e2, 1e12, 'Excluded by\nexperiments', \
         horizontalalignment='center', \
         verticalalignment='center', \
         multialignment='center')

cmeas = np.loadtxt('prev_meas/decca_prl_94_240401_2005.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)
if annotate:
    if ref:
        plt.text(2, 0.6e11, 'PRL 94, 240401 (2005)', rotation=-12.5, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)
    if institute:
        plt.text(1, 0.8e11, 'IUPUI (2005)', rotation=-14.5, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)

cmeas = np.loadtxt('prev_meas/sushkov_prl_107_171101_2011.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k-',linewidth=1.5)
if annotate:
    if ref:
        plt.text(3.5, 1e8, 'PRL 107, 171101 (2011)', rotation=-37, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)
    if institute:
        plt.text(2.5, 2e8, 'Yale (2011)', rotation=-37, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)

cmeas = np.loadtxt('prev_meas/geraci_prd_78_022002_2008.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:-3,0]*1e6,cmeas[:-3,1],'k',linewidth=1.5)
if annotate:
    if ref:
        plt.text(55, 0.25e3, 'PRD 78, 022002 (2008)', rotation=-27, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)
    if institute:
        plt.text(45, 0.3e3, 'Stanford (2008)', rotation=-27, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)

#cmeas = np.loadtxt('prev_meas/kapner_prl_98_021101_2007.txt',delimiter=",",skiprows=1)
cmeas = np.loadtxt('prev_meas/eot-wash_limits_2006.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)
if annotate:
    if ref:
        plt.text(140, 2.5e-1, 'PRL 98, 021101 (2007)', rotation=-33, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)
    if institute:
        plt.text(140, 2.0e-1, 'Washington (2007)', rotation=-34, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)


cmeas = np.loadtxt('prev_meas/decca_2014.txt',delimiter=",",skiprows=0)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)
if annotate:
    if ref:
        plt.text(1, 0.25e8, 'PRL 116, 221102 (2016)', rotation=-33, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)
    if institute:
        plt.text(0.9, 0.25e8, 'IUPUI (2016)', rotation=-31, \
                 horizontalalignment='center', \
                 verticalalignment='center', fontsize=9)





if plot_theory:

    cdat = np.loadtxt("prev_meas/andy_yuk.txt",delimiter=",",skiprows=1)
    plt.loglog(cdat[:,0]*1e6,cdat[:,1],'--',linewidth=1, #color=[0.25,0.25,0.25], \
               color='C0', zorder=1)
    if annotate_theory:
        plt.text(0.3, 1.5e5, 'Yukawa messengers', fontsize=9,\
                 horizontalalignment='center', \
                 verticalalignment='center', color='C0')

    cdat = np.loadtxt("prev_meas/andy_dil.txt",delimiter=",",skiprows=1)
    plt.loglog(cdat[:,0]*1e6,cdat[:,1],'--',linewidth=1, #color=[0.25,0.25,0.25], \
               color='C1', zorder=1)
    if annotate_theory:
        plt.text(0.3, 1e4, 'Dilaton', fontsize=9, \
                 horizontalalignment='center', \
                 verticalalignment='center', color='C1')




    plt.loglog(cdat_m1l[:,0]*1e6, cdat_m1l[:,1],':',linewidth=1,# color=[0.25,0.25,0.25], \
               zorder=1, color='C2')
    plt.loglog(cdat_m1l[:,0]*1e6, hh_m1h,':',linewidth=1,# color=[0.25,0.25,0.25], \
               zorder=1, color='C2')
    if annotate_theory:
        plt.annotate('', xy=(2, 3.5e4), xycoords='data', xytext=(2, 0.3e1), textcoords='data', \
                     arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', color='C2'))
        plt.text(2, 0.2e3, 'gluon\nmodulus', fontsize=9, color='C2', \
                 horizontalalignment='center', verticalalignment='center', \
                 multialignment='center', bbox=dict(fc='w', ec='w'))




    plt.loglog(np.append(cdat_m2l[:,0][::-1], cdat_m2l[:,0][0]*10.0)*1e6, \
               np.append(cdat_m2l[:,1][::-1], cdat_m2l[:,1][0]), \
               ':',linewidth=1, #color=[0.25,0.25,0.25], \
               zorder=1, color='C3')

    plt.loglog(cdat_m2l[:,0]*1e6, hh_m2h,':',linewidth=1, #color=[0.25,0.25,0.25], \
               zorder=1, color='C3')
    if annotate_theory:
        plt.annotate('', xy=(0.15, 2.7e3), xycoords='data', xytext=(0.15, 0.15), textcoords='data', \
                     arrowprops=dict(arrowstyle='<->', connectionstyle='arc3', color='C3'))
        plt.text(0.15, 0.2e2, 'heavy q\nmoduli', fontsize=9, color='C3', \
                 horizontalalignment='center', verticalalignment='center', multialignment='center', \
                 bbox=dict(fc='w', ec='w'))



if plot_projections:

    if all_three:
        proj_bad = np.loadtxt('projections/attractorv2_sep15um_noise1e-17NrtHz_int1e4s.txt', \
                              delimiter=',')
        plt.loglog(proj_bad[:,0]*1e6, proj_bad[:,1], '--', color='r', lw=2, #alpha=0.6, \
                   label=r'$10^{-17}$ $\rm{N/\sqrt{Hz}}$, $s=12.5$ $\mu\rm{m}$, $t=10^4$ $\rm{s}$', \
                   zorder=20)

    proj_mid = np.loadtxt('projections/attractorv2_sep10um_noise1e-18NrtHz_int1e5s.txt', \
                          delimiter=',')
    plt.loglog(proj_mid[:,0]*1e6, proj_mid[:,1], '--', color='g', lw=2, #alpha=0.6, \
               label=r'$10^{-18}$ $\rm{N/\sqrt{Hz}}$, $s=7.5$ $\mu\rm{m}$, $t=10^5$ $\rm{s}$', \
               zorder=20)


    proj_good = np.loadtxt('projections/attractorv2_sep5um_noise1e-19NrtHz_int1e6s.txt', \
                           delimiter=',')
    plt.loglog(proj_good[:,0]*1e6, proj_good[:,1], '--', color='b', lw=2, #alpha=0.6, \
               label=r'$10^{-19}$ $\rm{N/\sqrt{Hz}}$, $s=2.5$ $\mu\rm{m}$, $t=10^6$ $\rm{s}$', \
               zorder=20)

    plt.legend(loc=1, fontsize=8)



if plot_shield_no_shield:


    no_shield = np.loadtxt('shield_no_shield/no_shield.csv', \
                          delimiter=',')
    plt.loglog((10**no_shield[:,0])*1e6, 10**no_shield[:,1], '--', color='g', lw=2, #alpha=0.6, \
               label=r'No shield', \
               zorder=20)


    shield = np.loadtxt('shield_no_shield/shield.csv', \
                          delimiter=',')
    plt.loglog(10**(shield[:,0])*1e6, 10**(shield[:,1]), '--', color='b', lw=2, #alpha=0.6, \
               label=r'Shield', \
               zorder=20)

    plt.legend()

if plot_freq_sensitivity:


    s_15um = np.loadtxt('freq_sensitivity/s_15um_1e4s.csv', \
                          delimiter=',')
    plt.loglog((10**s_15um[:,0])*1e6, 10**s_15um[:,1], '--', color='g', lw=2, #alpha=0.6, \
               label=r'15$\, \mu $m separation $10^4$s', \
               zorder=20)


    s_10um = np.loadtxt('freq_sensitivity/s_10um_1e4s.csv', \
                          delimiter=',')
    plt.loglog(10**(s_10um[:,0])*1e6, 10**(s_10um[:,1]), '--', color='b', lw=2, #alpha=0.6, \
               label=r'10$\, \mu $m separation $10^4$s', \
               zorder=20)

    plt.legend()
#cdat = np.loadtxt("prev_meas/andy_init_sens.txt",delimiter=",",skiprows=1)
##plt.loglog(cdat[:,0]*1e6,cdat[:,1],'r',linewidth=2.5,label=lab)
#cdat = np.loadtxt("prev_meas/andy_fut_sens.txt",delimiter=",",skiprows=1)
#plt.loglog(cdat[:,0]*1e6,cdat[:,1],'r--',linewidth=2.5,label=lab)
#cdat = np.loadtxt("prev_meas/andy_matfut_sens.txt",delimiter=",",skiprows=1)
#plt.loglog(cdat[:,0]*1e6,cdat[:,1],'r:',linewidth=2.5,label=lab)

plt.xlim([0.05, 1000])
plt.ylim([0.001,1e20])
plt.plot([0.01,1000],[1,1],'k--')
#plt.yticks(np.logspace(0,10,5))
plt.yticks(np.logspace(-2,20,7))
plt.xlabel('Length scale, $\lambda$ [$\mu$m]')
plt.ylabel(r'Strength parameter, $|\alpha|$')

fig.set_size_inches(6,4.5)
#plt.gcf().subplots_adjust(bottom=0.14,left=0.16,right=0.95,top=0.95)

plt.tight_layout()

plt.savefig('sens_plot_20um.pdf',format='pdf')

plt.show()
