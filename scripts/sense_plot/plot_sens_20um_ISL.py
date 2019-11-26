import os, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib

matplotlib.rcParams.update({'font.size': 14})

gap_list = [6e-6,6e-6,6e-6]
lam_list = np.logspace(-1.0,2.0,40)*1e-6
print(lam_list)

sens_vals = np.zeros((len(lam_list),len(gap_list)))
force_vals = np.zeros((len(lam_list),len(gap_list)))
force_vals_old = np.zeros((len(lam_list),len(gap_list)))


#plt.figure()
for i in range(len(gap_list)):
    for j in range(0,len(lam_list)):

        gap = gap_list[i]
        lam = lam_list[j]

        fname = 'data_20um/lam_arr_20um_%.3f_%.3f.npy' % (gap*1e6,lam*1e6)
        print(fname)
        if( not os.path.isfile(fname)): continue
        cval = np.load(fname)
        print(cval) 

        if( i==0):
            sigf = 1e-19  ## force sensitivity with integration time
        elif(i==1):
            sigf = 5e-19/np.sqrt(1e5)  ## force sensitivity with integration time
        else:
            sigf = 1.6e-20/np.sqrt(1e5)  ## force sensitivity with integration time
        sens_vals[j,i] = sigf/cval ##[0]
        force_vals[j,i] = cval ##[0]


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
#ax.add_patch(Polygon(zippts,closed=True,fill=False,hatch='...',color=[0.5, 0.5, 0.5]))

#prev meas
cmeas = np.loadtxt('prev_meas/master.txt',delimiter=",",skiprows=1)
#plt.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color=[135./256,205./256,250/256.])
plt.fill_between(cmeas[:,0]*1e6,cmeas[:,1],1e20,color=[255./256,246./256,143/256.])

cmeas = np.loadtxt('prev_meas/decca_prl_94_240401_2005.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k:',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/sushkov_prl_107_171101_2011.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k-',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/geraci_prd_78_022002_2008.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:-3,0]*1e6,cmeas[:-3,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/kapner_prl_98_021101_2007.txt',delimiter=",",skiprows=1)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)

cmeas = np.loadtxt('prev_meas/decca_2014.txt',delimiter=",",skiprows=0)
plt.loglog(cmeas[:,0]*1e6,cmeas[:,1],'k',linewidth=1.5)

col_list = ['b', 'r', 'r--', 'r']
xmin_list = [1.1, 0.55, 0.4, 0.2]
xmax_list = [30, 80, 100, 100]
cdat = np.loadtxt("chas_curve.txt",delimiter=",",skiprows=1)
gpts = np.logical_and(cdat[:,0]>1.1e-6,cdat[:,0]<30e-6)
plt.loglog(cdat[gpts,0]*1e6,cdat[gpts,1],'b',linewidth=2.5)
for i in range(1,len(gap_list)):
    gval = "%.1f" % (gap_list[i]*1e6)
    lab = r"Resonant excitation, $d = " + gval +  "\mu m$" 
    gpts = np.logical_and(lam_list*1e6 >= xmin_list[i],lam_list*1e6 <= xmax_list[i])
    plt.loglog(lam_list[gpts]*1e6,sens_vals[gpts,i],col_list[i],linewidth=2.5,label=lab)
    
    #lab = r"Constant excitation, $d = " + gval +  "\mu m$"
    #plt.loglog(lam_list*1e6,sens_vals_num[:,i],'b--',linewidth=2.5,label=lab)

    out_vals = np.hstack((lam_list, sens_vals[:,i]))
    #np.save("data/sens_data_%.1f.npy"%(gap_list[i]*1e6), out_vals)

plt.xlim([0.5, 100])
plt.ylim([0.1,1e9])
plt.plot([0.5,100],[1,1],'k:')
#plt.yticks(np.logspace(0,10,5))
plt.yticks(np.logspace(0,10,6))
plt.xlabel('Length scale, $\lambda$ [$\mu$m]')
plt.ylabel(r'Strength parameter, $|\alpha|$')

fig.set_size_inches(5,4)
plt.gcf().subplots_adjust(bottom=0.14,left=0.16,right=0.95,top=0.95)
plt.savefig('sens_plot_20um.pdf',format='pdf')

plt.show()
