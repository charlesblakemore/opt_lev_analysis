import os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

gap_list = np.array([0.7e-6, 2.3e-6, 3.1e-6, 5.6e-6])+2.5e-6
lam_list = np.logspace(-1.0,2.0,40)*1e-6
print(lam_list)

sens_vals = np.zeros((len(lam_list),len(gap_list)))
force_vals = np.zeros((len(lam_list),len(gap_list)))
force_vals_old = np.zeros((len(lam_list),len(gap_list)))

for i in range(len(gap_list)):
    for j in range(0,len(lam_list)):

        gap = gap_list[i]
        lam = lam_list[j]

        fname = 'data/lam_binnedint_depth_20.000_shield_0.000_gap_%.3f_lam_%.3f.npy' % (gap*1e6,lam*1e6)
        print(fname)
        if( not os.path.isfile(fname)): continue
        cval = np.load(fname)

        if( gap_list[i] < 2.8e-6+2.5e-6 ):
            sigf = 5e-17/np.sqrt(1e6)  ## force sensitivity with integration time
        else:
            sigf = 1.4e-20/np.sqrt(1e6)
        sens_vals[j,i] = sigf/cval ##[0]
        force_vals[j,i] = cval ##[0]


        ##### fname = 'data/lam_arr_cyl_%.3f_%.3f.npy' % (gap*1e6,lam*1e6)
        ##### cval = np.load(fname)

        ##### sigf2 = 1.9e-16/np.sqrt(10) ## force sensitivity with integration time
        ##### sens_vals_num[j,i] = sigf2/cval[0]

    #plt.loglog( lam_list, sens_vals[:,i] )



##gap_list = np.linspace(2.5e-6,17.5e-6,10)
##lam_list = np.logspace(-1.0,2.0,10)*1e-6

##gap_list2 = [5.0e-6, 7.5e-6, 10e-6]
##sens_vals_num_pot = np.zeros((len(lam_list),len(gap_list2)))

##for i in range(len(gap_list2)):
##    for j in range(2,len(lam_list)):

##        tot_vals = 2*len(gap_list)
##        curr_pot = np.zeros(2*len(gap_list))
##        for k in range(len(gap_list)):

##            gap = gap_list[k]
##            lam = lam_list[j]

##            fname = 'data/lam_arr_pot_cyl_%.3f_%.3f.npy' % (gap*1e6,lam*1e6)
##            cval = np.load(fname)
##            print cval[0]

##            curr_pot[k] += -cval[0]
##            ## now add the potential from the mass at neg d


##        ## now fit curr pot
##        fig = plt.figure()
##        plt.plot(gap_list,curr_pot)
##        plt.show()



f0 = 1e3
G = 6.67e-11
Ma = 20e-13
b = 1e-4
Q = 1e5

## now make analytical for point mass and DC measurement
lam_vals = np.logspace(-1.0,2.0,100)*1e-6
sens_vals2 = np.zeros((len(lam_vals),len(gap_list)))
for i in range(len(gap_list)):
    cgap = gap_list[i] + 2.5e-6

    sens_vals2[:,i] = (2*math.pi*f0)**2 * cgap**3/(4.0*G*Ma) * np.exp(cgap/lam_vals)/(1.0 + cgap/lam_vals + 0.5*(cgap/lam_vals)**2) * np.sqrt(b/(2*math.pi*f0*Q))


fig=plt.figure(88)
## plot sensitivity compared to previous measurements

## theory models

##gauge bosons
cmeas = np.loadtxt('prev_meas/master.txt',delimiter=",",skiprows=1)
x = cmeas[:,0]*1e6
gpts = x < 2.5
x = x[gpts]
y1 = np.exp(np.interp(np.log(cmeas[gpts,0]*1e6),np.log([1e-1, 2.0443]),np.log([1.6e9, 1.2954e8])))
y2 = cmeas[gpts,1]               
xtot = np.hstack((x,x[-1::-1]))
ytot = np.hstack((y1,y2[-1::-1]))
zippts = list(zip(xtot,ytot))
ax = plt.gca()
#ax.add_patch(Polygon(zippts,closed=True,fill=False,hatch='//'))


##gluon modulus
## both moduli
cmeas = np.loadtxt('prev_meas/master.txt',delimiter=",",skiprows=1)
x = cmeas[:,0]*1e6
gpts1 = x < 3.16
x1 = x[gpts1]
y1h = np.exp(np.interp(np.log(x1),np.log([1e-1, 3.57]),np.log([1.829e10, 1.5439e7])))
y1l = np.exp(np.interp(np.log(x1),np.log([1e-1, 3.15]),np.log([1055.008, 1.0394])))
gpts2 = x>3.16
x2 = x[gpts2]
y2l = np.exp(np.interp(np.log(x2),np.log([1e-1, 3.57]),np.log([1055.008, 1.0394])))
y2l[np.isnan(y2l)] = 0.1
y2h = cmeas[gpts2,1]
gpts3 = x<3.16
x3 = x[gpts3] ##[0.1, 3.16]
y3h = cmeas[gpts3,1]  ##np.exp(np.interp(np.log(x3),np.log([1e-1, 3.57]),np.log([1.829e10, 1.5439e7])))
y3l = np.exp(np.interp(np.log(x3),np.log([1e-1, 3.15]),np.log([1055.008, 1.0394])))
xtot = np.hstack((x1,x2,x2[-1::-1],x3[-1::-1]))
ytot = np.hstack((y1l,y2l,y2h[-1::-1],y3h[-1::-1]))
##raw_input('e')
##xtot = np.hstack((x2,x2[-1::-1]))
##ytot = np.hstack((y2l,y2h[-1::-1]))
zippts = list(zip(xtot,ytot))
ax.add_patch(Polygon(zippts,closed=True,fill=False,hatch='...',color=[0.5, 0.5, 0.5]))


#yukawa messengers
#ll = plt.plot([1e-1, 6.3e0], [74267.9, 74267.9],color=[0.25,0.25,0.25],ls='dotted',linewidth=2)
#seq = [5, 3]
#ll[0].set_dashes(seq)

#dilaton
#seq = [10, 4]
#ll=plt.plot([1e-1, 1.15e1], [6752.4, 6752.4],color=[0.25,0.25,0.25],ls='dotted',linewidth=2)
#ll[0].set_dashes(seq)

#prev meas
cmeas = np.loadtxt('prev_meas/master.txt',delimiter=",",skiprows=1)
#plt.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color=[135./256,205./256,250/256.])
plt.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color=[255./256,246./256,143/256.])

cmeas = np.loadtxt('prev_meas/decca_prl_94_240401_2005.txt',delimiter=",",skiprows=1)
#plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/sushkov_prl_107_171101_2011.txt',delimiter=",",skiprows=1)
#plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/geraci_prd_78_022002_2008.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[3:-3,0]*1e6,cmeas[3:-3,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/kapner_prl_98_021101_2007.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/decca_2014.txt',delimiter=",",skiprows=0)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)

col_list = ['b--', 'b', 'r--', 'r']
xmin_list = [0.09, 0.2, 0.15, 0.28]
xmax_list = [20, 20, 50, 50]
for i in range(len(gap_list)):
    gval = "%.1f" % (gap_list[i]*1e6)
    lab = r"Resonant excitation, $d = " + gval +  "\mu m$" 
    gpts = np.logical_and(lam_list*1e6 >= xmin_list[i],lam_list*1e6 <= xmax_list[i])
    plt.loglog(lam_list[gpts]*1e6,sens_vals[gpts,i],col_list[i],linewidth=2.5,label=lab)
    
    #lab = r"Constant excitation, $d = " + gval +  "\mu m$"
    #plt.loglog(lam_list*1e6,sens_vals_num[:,i],'b--',linewidth=2.5,label=lab)

    out_vals = np.hstack((lam_list, sens_vals[:,i]))
    np.save("data/sens_data_%.1f.npy"%(gap_list[i]*1e6), out_vals)

plt.xlim([0.1, 100])
plt.ylim([1,1e11])
#plt.yticks(np.logspace(0,10,5))
plt.yticks(np.logspace(0,10,6))
plt.xlabel('Length scale, $\lambda$ [$\mu$m]')
plt.ylabel(r'Strength parameter, $|\alpha|$')

fig.set_size_inches(5,4)
plt.gcf().subplots_adjust(bottom=0.14,left=0.16,right=0.95,top=0.95)
plt.savefig('sens_plot.pdf',format='pdf')



fig=plt.figure(188)
## plot sensitivity compared to previous measurements

## theory models

##gauge bosons
cmeas = np.loadtxt('prev_meas/master.txt',delimiter=",",skiprows=1)
x = cmeas[:,0]*1e6
gpts = x < 2.5
x = x[gpts]
y1 = np.exp(np.interp(np.log(cmeas[gpts,0]*1e6),np.log([1e-1, 2.0443]),np.log([1.6e9, 1.2954e8])))
y2 = cmeas[gpts,1]               
xtot = np.hstack((x,x[-1::-1]))
ytot = np.hstack((y1,y2[-1::-1]))
zippts = list(zip(xtot,ytot))
ax = plt.gca()
#ax.add_patch(Polygon(zippts,closed=True,fill=False,hatch='//'))


##gluon modulus
## both moduli
cmeas = np.loadtxt('prev_meas/master.txt',delimiter=",",skiprows=1)
x = cmeas[:,0]*1e6
gpts1 = x < 3.16
x1 = x[gpts1]
y1h = np.exp(np.interp(np.log(x1),np.log([1e-1, 3.57]),np.log([1.829e10, 1.5439e7])))
y1l = np.exp(np.interp(np.log(x1),np.log([1e-1, 3.15]),np.log([1055.008, 1.0394])))
gpts2 = x>3.16
x2 = x[gpts2]
y2l = np.exp(np.interp(np.log(x2),np.log([1e-1, 3.57]),np.log([1055.008, 1.0394])))
y2l[np.isnan(y2l)] = 0.1
y2h = cmeas[gpts2,1]
gpts3 = x<3.16
x3 = x[gpts3] ##[0.1, 3.16]
y3h = cmeas[gpts3,1]  ##np.exp(np.interp(np.log(x3),np.log([1e-1, 3.57]),np.log([1.829e10, 1.5439e7])))
y3l = np.exp(np.interp(np.log(x3),np.log([1e-1, 3.15]),np.log([1055.008, 1.0394])))
xtot = np.hstack((x1,x2,x2[-1::-1],x3[-1::-1]))
ytot = np.hstack((y1l,y2l,y2h[-1::-1],y3h[-1::-1]))
##raw_input('e')
##xtot = np.hstack((x2,x2[-1::-1]))
##ytot = np.hstack((y2l,y2h[-1::-1]))
zippts = list(zip(xtot,ytot))
#ax.add_patch(Polygon(zippts,closed=True,fill=False,hatch='...',color=[0.5, 0.5, 0.5]))


#yukawa messengers
#ll = plt.plot([1e-1, 6.3e0], [74267.9, 74267.9],color=[0.25,0.25,0.25],ls='dotted',linewidth=2)
#seq = [5, 3]
#ll[0].set_dashes(seq)

#dilaton
#seq = [10, 4]
#ll=plt.plot([1e-1, 1.15e1], [6752.4, 6752.4],color=[0.25,0.25,0.25],ls='dotted',linewidth=2)
#ll[0].set_dashes(seq)

#prev meas
cmeas = np.loadtxt('prev_meas/master.txt',delimiter=",",skiprows=1)
#plt.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color=[135./256,205./256,250/256.])
plt.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color=[255./256,246./256,143/256.])

cmeas = np.loadtxt('prev_meas/decca_prl_94_240401_2005.txt',delimiter=",",skiprows=1)
#plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/sushkov_prl_107_171101_2011.txt',delimiter=",",skiprows=1)
#plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/geraci_prd_78_022002_2008.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[3:-3,0]*1e6,cmeas[3:-3,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/kapner_prl_98_021101_2007.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/decca_2014.txt',delimiter=",",skiprows=0)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)

ourmeas = np.load("../cant_force/plots_yuk/final_yuk_limits_paper.npy")
plt.loglog( ourmeas[:,0]*1e6, ourmeas[:,1], 'b', linewidth=2.0 )

plt.xlim([0.1, 100])
plt.ylim([1,1e11])
plt.yticks(np.logspace(0,10,6))
plt.xlabel('Length scale, $\lambda$ [$\mu$m]')
plt.ylabel(r'Strength parameter, $|\alpha|$')

fig.set_size_inches(6,4.5)
plt.gcf().subplots_adjust(bottom=0.14,left=0.16,right=0.95,top=0.95)
plt.savefig('sens_plot_nolim.pdf',format='pdf')


fig_tot = plt.figure()
#prev meas
cmeas = np.loadtxt('prev_meas/master.txt',delimiter=",",skiprows=1)
#plt.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color=[135./256,205./256,250/256.])
plt.fill_between(cmeas[:,0],cmeas[:,1],1e20,color=[255./256,246./256,143/256.])

cmeas = np.loadtxt('prev_meas/decca_prl_94_240401_2005.txt',delimiter=",",skiprows=1)
#plt.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/sushkov_prl_107_171101_2011.txt',delimiter=",",skiprows=1)
#plt.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/geraci_prd_78_022002_2008.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[3:-3,0],cmeas[3:-3,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/kapner_prl_98_021101_2007.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/decca_2014.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/mid_range.txt',delimiter=",",skiprows=1)
plt.fill_between(cmeas[:,0],cmeas[:,1],1e20,color=[255./256,246./256,143/256.])

cmeas = np.loadtxt('prev_meas/mid_range.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/long_range_tot.txt',delimiter=",",skiprows=1)
plt.fill_between(cmeas[:,0],cmeas[:,1],1e20,color=[255./256,246./256,143/256.])

cmeas = np.loadtxt('prev_meas/long_range_lab.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1.5)
cmeas = np.loadtxt('prev_meas/long_range_geophys.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1.5)
cmeas = np.loadtxt('prev_meas/long_range_llr.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0],cmeas[:,1],'k',linewidth=1.5)

plt.plot([1e-7, 1e15], [1, 1], 'k--', linewidth=1.5)

plt.xlim([1e-7,  1e15])
plt.ylim([1e-12, 1e12])

plt.xlabel('Length scale, $\lambda$ [m]')
plt.ylabel(r'Strength parameter, $|\alpha|$')

fig_tot.set_size_inches(6,4.5)
plt.yticks(np.logspace(-12,12,9))
#plt.gca().tick_params(axis='y',which='minor',bottom='off')
plt.minorticks_off()
plt.gcf().subplots_adjust(bottom=0.14,left=0.16,right=0.95,top=0.95)
plt.savefig('lim_plot_full.pdf',format='pdf')

plt.show()


## now plot ratio between the two
fig=plt.figure(188)
#prev meas
cmeas = np.loadtxt('prev_meas/master.txt',delimiter=",",skiprows=1)
currsens = sens_vals[:,1]
ratio = np.interp(lam_list,cmeas[:,0],cmeas[:,1])/currsens

plt.loglog(lam_list*1e6,ratio)









plt.show()

fig=plt.figure(89)
for i in range(len(gap_list)):
    gval = "%.1f" % (gap_list[i]*1e6)
    lab = r"Resonant excitation, $d = " + gval +  "\mu m$" 
    plt.loglog(lam_list*1e6,sens_vals[:,i],linewidth=2,label=lab)
    ##plt.loglog(lam_vals,sens_vals2[:,i],'--')
    lab = r"Constant excitation, $d = " + gval +  "\mu m$"
    plt.loglog(lam_list*1e6,sens_vals_num[:,i],'--',linewidth=2,label=lab)
    plt.ylim([1e-2,1e20])
plt.legend(loc="upper right",prop={'size':10})
plt.xlabel('Yukawa length scale, $\lambda$ [$\mu$m]')
plt.ylabel(r'Yukawa coupling, $|\alpha|$')
fig.set_size_inches(6,4.5)
plt.savefig('dist_comp.pdf',format='pdf')


plt.figure(98)
for i in range(len(gap_list)):
    vals2 = np.interp(lam_list,lam_vals,sens_vals2[:,i])
    plt.loglog(lam_list,vals2/sens_vals[:,i])


fig = plt.figure(108)
for i in [1]: ##range(len(gap_list)):
    plt.loglog(lam_vals*1e6,sens_vals2[:,i],label="Potential calc, point masses")
    #plt.loglog(lam_list*1e6,sens_vals_num[:,i],'--',label="Force calc, distributed masses")
    plt.ylim([1,1e20])
    plt.xlabel('Yukawa length scale, $\lambda$ [$\mu$m]')
    plt.ylabel(r'Yukawa coupling, $|\alpha|$')
plt.legend(loc="lower left",prop={'size':10})
fig.set_size_inches(6,4.5)
plt.savefig('approx_comp.pdf',format='pdf')


