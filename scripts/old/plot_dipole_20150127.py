import glob, re, os
import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import scipy.optimize as sp

dlist = [0.375,0.75,0.875,1.25,2.125, 2.75, 3.5, 4.625,] ##[1.125, 2.25, 3.5, 4.625]
aclist = [50,100,100,100,250, 1000, 1000, 2000]
vdc_list = [-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-12.5,-10,-7,-6.5,-6,-5.7,-5.5,-5,0,5,7.5,10,12.5,15,17.5,20,22.5,25,27,27.5,28.99,29,29.2,29.5,29.7,29.8,29.85,29.86,29.9,29.90,30,30.01,30.2,30.5,31,32.5,35,40]

# cal_path = "/data/20150126/Bead1/chargelp"
# fcal_list = glob.glob( os.path.join(cal_path, "urmbar*.h5") )
# fcal_list = sorted(fcal_list, key = bu.find_str)
# #fcal_list.append( "/data/20150126/Bead1/cantidrive2/urmbar_nobead_50mV_41Hz.h5" )

# cal_list = []
# for fname in fcal_list:
#     print fname
#     dat, attribs, cf = bu.getdata(os.path.join(path, fname))
#     if( len(dat) == 0 ): continue
#     fsamp = attribs["Fsamp"]
#     corr_full = bu.corr_func(dat[:,bu.drive_column], dat[:,bu.data_columns[0]], fsamp, fdrive)
#     if(False):
#         plt.figure()
#         plt.plot(corr_full)
#         plt.show()
#     cal_list.append(corr_full[0])

# plt.figure()
# plt.plot(cal_list)
# plt.show()

path = "/data/20150127/Bead3/cantidrive"
fdrive = 41 ## Hz

plt.figure()
col_list = [[1,0.5,0],'b','r','g','c','m','y','k']

plt.plot([-35, 35],[0, 0], 'k--', linewidth=1.5)

slope_list = []
for col,d,acv in zip(col_list,dlist,aclist):

    dstr = str(d)
    dstr = dstr.replace(".", "_")
    if( dstr == "0_75" ): dstr += "0"

    flist = []
    for vdc in vdc_list:
        vstr = str(vdc)
        vstr = vstr.replace("-","m")
        vstr = vstr.replace(".","_")
        cname = "urmbar_xyzcool_%s_%sdc_%dmV_41Hz.h5"%(dstr, vstr, acv)
        flist.append( cname )


    corr_list = []
    corr_list2 = []
    for vdc, fname in zip(vdc_list, flist):
        print fname

        dat, attribs, cf = bu.getdata(os.path.join(path, fname))
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
    p = np.polyfit(corr_list[fit_pts[0]:fit_pts[1],0], corr_list[fit_pts[0]:fit_pts[1],1], 1)
    p2 = np.polyfit(corr_list2[fit_pts[0]:fit_pts[1],0], corr_list2[fit_pts[0]:fit_pts[1],1], 1)
    slope_list.append([p[0], -p[1]/p[0]])

    plt.plot(corr_list[:,0], corr_list[:,1], '-o', color=col, markerfacecolor=col, markeredgecolor=col, linewidth=1.5, label="d = %d $\mu$m"%(d*5/0.125))
    xx=np.linspace(corr_list[fit_pts[0],0],corr_list[fit_pts[1],0], 1e2)
    #plt.plot(xx, np.polyval(p, xx), col+":",  linewidth = 1.5) ##, label="slope = %.1e N/V"%p[0])

    yy = plt.ylim()
    #plt.plot( [-p[1]/p[0], -p[1]/p[0]], yy, 'k--')

    #plt.plot(corr_list2[:,0], corr_list2[:,1], 'ro', linewidth=1.5)
    #plt.plot(xx, np.polyval(p2, xx), 'r', linewidth = 1.5)


plt.xlim([-65, 35])
plt.plot([29.88, 29.88],yy, 'k--', linewidth=1.5)
plt.ylim(yy)
plt.ylim([-1e-16, 2e-16])
plt.xlabel("DC voltage [V]")
plt.ylabel("Force on microsphere [N]")
plt.legend(loc="upper left",prop={"size": 11})
plt.gcf().set_size_inches(6,4.5)
plt.savefig("plots/dipole_force_20150127.pdf")

def qfun( d, p ):
    return 1.0*p/d**2
def cfun( d, p ):
    return 1.0*p/d**3
def qcfun(d, p1, p2):
    return 1.0*p1/d**2 + 1.0*p2/d**3

dvals = np.array(dlist)*5/0.125
slope_list = np.array(slope_list)
plt.figure()

px2,_ = sp.curve_fit(qfun, dvals, slope_list[:,0], p0=-1e4)
px3,_ = sp.curve_fit(cfun, dvals, slope_list[:,0], p0=-1e6)
px4,_ = sp.curve_fit(qcfun, dvals, slope_list[:,0], p0=[px2,px3])

xx = np.linspace(14, 200, 1e3)
plt.plot(dvals, slope_list[:,0], 'ko')
plt.plot(xx, qfun(xx, px2), "r", label="1/$r^2$", linewidth=1.5)
plt.plot(xx, cfun(xx, px3), "b", label="1/$r^3$", linewidth=1.5)
#plt.plot(xx, -qcfun(xx, px4[0],px4[1]), "g", label="1/$r^2$ + 1/$r^3$", linewidth=1.5)
#plt.gca().set_yscale('log')
plt.legend(loc="lower right")
plt.xlabel("Distance from bead [$\mu$m]")
plt.ylabel("Slope around Vdc = 0 [N/V]")
plt.gcf().set_size_inches(6,4.5)
plt.savefig("plots/response_vs_dist_20150127.pdf")

plt.figure()
plt.plot(dvals, slope_list[:,1], 'ko-', linewidth=1.5)
plt.xlabel("Distance from bead [$\mu$m]")
plt.ylabel("Minimum response voltage [V]")
plt.ylim([-65, 35])
plt.gcf().set_size_inches(6,4.5)
plt.savefig("plots/min_response_volt_20150127.pdf")

plt.show()
