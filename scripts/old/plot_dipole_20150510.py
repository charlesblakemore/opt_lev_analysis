import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.optimize as sp

dlist = [5, 10, 20,] ##[1.125, 2.25, 3.5, 4.625]
aclist = np.ones_like(dlist)*1000
#vdc_list = [-5,-4,-3,-2,-1, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]
path = "/data/20150508/Bead2/dipsweep_%sum"
fdrive = 23 ## Hz

## make the list of possible dc values

plt.figure()
col_list = [[1,0.5,0],'b','r','g','c','m','y','k']

plt.plot([-35, 35],[0, 0], 'k--', linewidth=1.5)

def sort_fun( str ):
    vdc = re.findall("-?\d+mVDC", str)
    vdc = float( vdc[0][:-4])/1000.
    return vdc

slope_list = []
for col,d,acv in zip(col_list,dlist,aclist):

    dstr = str(d)
    dstr = dstr.replace(".", "_")


    ## get list of dc offsets
    #if( d==10 and col=='b'):
    #    flist = sorted(glob.glob( os.path.join(path%dstr +"_test", "*.h5") ), key=sort_fun)
    if(d == 10):
        flist = sorted(glob.glob( os.path.join(path%dstr +"_2", "*.h5") ), key=sort_fun)
    else:
        flist = sorted(glob.glob( os.path.join(path%dstr, "*.h5") ), key=sort_fun)

    corr_list = []
    corr_list2 = []
    for fname in flist:
        #print fname
        cfile = fname ##os.path.join(path%dstr, fname)
        vdc = re.findall("-?\d+mVDC", cfile)
        if( "_test" in cfile ):
            vdc = re.findall("-?\d+mV", cfile)            
            vdc = float( vdc[0][:-2])/1000.
            print vdc
        else:
            vdc = float( vdc[0][:-4])/1000.
        print cfile
        dat, attribs, cf = bu.getdata(cfile)
        if( len(dat) == 0 ): 
            print "skipping  " + fname
            continue
        fsamp = attribs["Fsamp"]

        corr_full = bu.corr_func(dat[:,bu.drive_column], dat[:,bu.data_columns[0]], fsamp, fdrive)
        corr_full2 = bu.corr_func(dat[:,bu.drive_column]**2, dat[:,bu.data_columns[0]], fsamp, fdrive)

        if(False):
            plt.figure()
            plt.plot(corr_full2)
            plt.show()

        corr_list.append([vdc, corr_full[0]])
        corr_list2.append([vdc, corr_full2[0]])

    ## scale into physical units 1e at 30 V/cm -> resp of 0.011
    ## i.e., 4.8 x 10^-16 N
    scale_fac = 4.8e-16/0.011 * 1/30.

    corr_list = np.array(corr_list)
    corr_list2 = np.array(corr_list2)

    corr_list[:,1] = corr_list[:,1]*scale_fac

    mid_pt = np.argmin( np.abs(corr_list[:,1]) )
    fit_pts = [np.max([0,mid_pt-1]),np.min([mid_pt+2,len(corr_list[:,1])-1])]
    #p = np.polyfit(corr_list[fit_pts[0]:fit_pts[1],0], corr_list[fit_pts[0]:fit_pts[1],1], 1)
    #p2 = np.polyfit(corr_list2[fit_pts[0]:fit_pts[1],0], corr_list2[fit_pts[0]:fit_pts[1],1], 1)
    #slope_list.append([p[0], -p[1]/p[0]])

    plt.plot(corr_list[:,0], corr_list[:,1], '-o', color=col, markerfacecolor=col, markeredgecolor=col, linewidth=1.5, label="d = %d $\mu$m"%(d))
    xx=np.linspace(corr_list[fit_pts[0],0],corr_list[fit_pts[1],0], 1e2)
    #plt.plot(xx, np.polyval(p, xx), col+":",  linewidth = 1.5) ##, label="slope = %.1e N/V"%p[0])

    yy = plt.ylim()
    #plt.plot( [-p[1]/p[0], -p[1]/p[0]], yy, 'k--')

    #plt.plot(corr_list2[:,0], corr_list2[:,1], 'ro', linewidth=1.5)
    #plt.plot(xx, np.polyval(p2, xx), 'r', linewidth = 1.5)


plt.xlim([-8, 8])
#plt.plot([29.88, 29.88],yy, 'k--', linewidth=1.5)
plt.ylim(yy)
#plt.ylim([-1e-16, 2e-16])
plt.xlabel("DC voltage [V]")
plt.ylabel("Force on microsphere [N]")
plt.legend(loc="upper right",prop={"size": 11})
#plt.gcf().set_size_inches(6,4.5)
# plt.savefig("plots/dipole_force_20150127.pdf")

# def qfun( d, p ):
#     return 1.0*p/d**2
# def cfun( d, p ):
#     return 1.0*p/d**3
# def qcfun(d, p1, p2):
#     return 1.0*p1/d**2 + 1.0*p2/d**3

# dvals = np.array(dlist)*5/0.125
# slope_list = np.array(slope_list)
# plt.figure()

# px2,_ = sp.curve_fit(qfun, dvals, slope_list[:,0], p0=-1e4)
# px3,_ = sp.curve_fit(cfun, dvals, slope_list[:,0], p0=-1e6)
# px4,_ = sp.curve_fit(qcfun, dvals, slope_list[:,0], p0=[px2,px3])

# xx = np.linspace(14, 200, 1e3)
# plt.plot(dvals, slope_list[:,0], 'ko')
# plt.plot(xx, qfun(xx, px2), "r", label="1/$r^2$", linewidth=1.5)
# plt.plot(xx, cfun(xx, px3), "b", label="1/$r^3$", linewidth=1.5)
# #plt.plot(xx, -qcfun(xx, px4[0],px4[1]), "g", label="1/$r^2$ + 1/$r^3$", linewidth=1.5)
# #plt.gca().set_yscale('log')
# plt.legend(loc="lower right")
# plt.xlabel("Distance from bead [$\mu$m]")
# plt.ylabel("Slope around Vdc = 0 [N/V]")
# plt.gcf().set_size_inches(6,4.5)
# plt.savefig("plots/response_vs_dist_20150127.pdf")

# plt.figure()
# plt.plot(dvals, slope_list[:,1], 'ko-', linewidth=1.5)
# plt.xlabel("Distance from bead [$\mu$m]")
# plt.ylabel("Minimum response voltage [V]")
# plt.ylim([-65, 35])
# plt.gcf().set_size_inches(6,4.5)
# plt.savefig("plots/min_response_volt_20150127.pdf")

plt.show()
