import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import bead_util as bu
from matplotlib.ticker import LogLocator
from matplotlib.path import Path
import matplotlib.patches as patches

from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as floating_axes

matplotlib.rc('font', family='serif') 
matplotlib.rc('font', serif='Times')
matplotlib.rc('text', usetex=True)

dir_list = ["/home/dcmoore/analysis/20140724/Bead3/no_charge_chirp",
            "/home/dcmoore/analysis/20140728/Bead5/no_charge_chirp",
            "/home/dcmoore/analysis/20140729/Bead4/no_charge_chirp",
            "/home/dcmoore/analysis/20140801/Bead6/no_charge",
            "/home/dcmoore/analysis/20140803/Bead8/no_charge",
            "/home/dcmoore/analysis/20140804/Bead1/no_charge",
            "/home/dcmoore/analysis/20140805/Bead1/no_charge",
            "/home/dcmoore/analysis/20140806/Bead1/no_charge",
            "/home/dcmoore/analysis/20140807/Bead1/no_charge",
            "/home/dcmoore/analysis/20140808/Bead7/no_charge"]

#dir_list = [ "/home/dcmoore/analysis/20140729/Bead4/no_charge_chirp",]
#dir_list = [ "/home/dcmoore/analysis/20140804/Bead1/no_charge", ]

do_morp = False
do_perl = False
if( do_morp ):
    out_name = "lim_bead_comb_morp_samesign.npy"
elif( do_perl ):
    out_name = "lim_bead_comb_perl_samesign.npy"
else:
    out_name = "lim_bead_comb_all_samesign.npy"

pos_frac = 1.0

fig = plt.figure()
ax = plt.subplot(111, polar=True, projection='polar' )

#grid_helper =floating_axes.GridHelperCurveLinear(PolarAxes.PolarTransform(),extremes=(0,np.pi, 0, np.log10(2e3)) )
#ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
#fig.add_subplot(ax)

#ax.set_ylim([1e-6, 10])            
#ax.set_yscale('log')

CL = 3.84 ## 99% CL

mlist = ['ks', 'ro', 'b^', 'g*']

yra = [0,2e3]

def plot_logpolar(ax, theta, r_, theta_err, r_err_, bullseye=0.3, do_ticks=False):
    #min10 = np.log10(np.min(r_))
    #max10 = np.log10(np.max(r_)*2)
    min10 = yra[0]
    max10 = np.log10(yra[1])
    r = np.log10(r_) - min10 + bullseye
    r_err_lo = np.log10(r_ - r_err_) - min10 + bullseye
    if(np.isnan(r_err_lo)):
        r_err_lo = 0.
    r_err_hi = np.log10(r_ + r_err_) - min10 + bullseye
    #ax.scatter(theta, r, **kwargs)
    #plt.errorbar( theta, r, xerr=0.1, yerr=0.1, fmt='ko')
    #print r_, r_err_
    #print r, r_err_lo, r_err_hi
    plot_error( theta, r, theta_err, [r_err_lo, r_err_hi] )
    l = np.arange(np.floor(min10), max10, 1)
    lm = np.log10(np.hstack((np.arange(1,10,1), np.arange(10,100,10), np.arange(100,1000,100))))

    ax_theta = 0 ##7.0*np.pi/8.0

    ax.set_rticks(l - min10 + bullseye) 
    ax.set_yticklabels([])

    

    ## this is terrible, have to draw my own tick marks
    if( do_ticks ):
        plt.plot([ax_theta, ax_theta],[np.log10(1), max10], 'k', linewidth=1)
        plt.plot([ax_theta+np.pi, ax_theta+np.pi],[np.log10(1), max10], 'k', linewidth=1)
        for l in lm:
            plt.plot([ax_theta, ax_theta+0.05/l], [l, l], 'k', linewidth=1)
            plt.plot([ax_theta+np.pi, ax_theta-0.05/l+np.pi], [l, l], 'k', linewidth=1)

        for rlab in [1, 10, 100, 1000]:
            ax.annotate(str(rlab), xy=(ax_theta, np.log10(rlab)), horizontalalignment='center', verticalalignment='top')

    ax.set_rlim(0, max10 - min10 + bullseye)
    return ax


def plot_error( xval, yval, xerr, yerr ):
    plt.plot( xval, yval, 'ko', markersize=5 )
    plt.plot( np.linspace(xval-xerr, xval, 1e2), yval*np.ones(1e2), 'k', linewidth=1.5 )
    plt.plot( np.linspace(xval+xerr, xval, 1e2), yval*np.ones(1e2), 'k', linewidth=1.5 )
    plt.plot( [xval, xval], [yerr[0], yval], 'k', linewidth=1.5 )
    plt.plot( [xval, xval], [yerr[1], yval], 'k', linewidth=1.5 )

    ## now do the hatches
    hatch_rad = 0.02 
    hatch_ang = 0.0075 * 2*np.pi
    plt.plot( [xval-xerr, xval-xerr], [yval-hatch_rad*yval, yval+hatch_rad*yval], 'k', linewidth=1.5 )    
    plt.plot( [xval+xerr, xval+xerr], [yval-hatch_rad*yval, yval+hatch_rad*yval], 'k', linewidth=1.5 )    
    plt.plot( [xval-hatch_ang/yval, xval+hatch_ang/yval], [yerr[0], yerr[0]], 'k', linewidth=1.5 )    
    plt.plot( [xval-hatch_ang/yval, xval+hatch_ang/yval], [yerr[1], yerr[1]], 'k', linewidth=1.5 )    

do_ticks = True
ang_list = []
for i,d in enumerate(dir_list):

    print d 

    c_zero = np.load( d + "/resid_data_0e.npy")
    c_ang = np.load( d + "/resid_data.npy")
    
    for dat in c_ang:
        
        if( dat[0] > 100 and dat[1]<0.1 ):
            ## this is the zero point, so fill it in
            #if( c_zero[0] < 0 ):
            #    dat[2] += np.pi
            #plt.errorbar( dat[2], np.abs(c_zero[0])*1e3, xerr=dat[3], yerr=c_zero[1]*1e3, fmt=mlist[i] )
            #plt.plot( dat[2], c_zero[0], 'k.')
            #plot_error( dat[2], np.abs(c_zero[0])*1e3, dat[3], c_zero[1]*1e3 )
            print "plotting: ", np.abs(c_zero[0])*1e6
            plot_logpolar(ax, dat[2], np.abs(c_zero[0])*1e6, dat[3], c_zero[1]*1e6, bullseye=0, do_ticks=do_ticks)
            if( do_ticks ): do_ticks = False


        ## now plot a wedge corresponding to the drive response
        if( dat[0] > 10 and np.abs( dat[1] ) < 5 and np.abs(dat[1]) > 0.9  ):
            #if( dat[1] < 0 ):
            #    dat[2] += np.pi
            ang_lo = dat[2] - dat[3]
            ang_hi = dat[2] + dat[3]
            rr = ax.get_ylim()
            #plt.plot( [ang_lo, ang_lo], yra, 'k', linewidth = 1.5 )
            #plt.plot( [ang_hi, ang_hi], yra, 'k', linewidth = 1.5 )
            ang_list.append( dat[2] - dat[3] )
            ang_list.append( dat[2] + dat[3] )

## now find the min and max angle on each side and draw a wedge
ang_list = np.array(ang_list)
##first look at only angles near 0
gidx = np.logical_or(ang_list < np.pi/2, ang_list > 3*np.pi/2)
min_ang = np.min( ang_list[ gidx ] + np.pi )
max_ang = np.max( ang_list[ gidx ] + np.pi )
min_ang -= np.pi
max_ang -= np.pi

verts = [[min_ang, 0],
         [min_ang, np.log10(yra[1])],
         [max_ang, np.log10(yra[1])],
         [max_ang, 0]]

path = Path(verts)
patch = patches.PathPatch(path, ec='none', fc=[0.7, 0.7, 0.7])
ax.add_patch(patch)

##now angles near 180
gidx = np.logical_and(ang_list > np.pi/2, ang_list < 3*np.pi/2)
min_ang = np.min( ang_list[ gidx ] )
max_ang = np.max( ang_list[ gidx ] )
#min_ang = -(2*np.pi + min_ang)
#max_ang = -(2*np.pi + max_ang)

verts2 = [[min_ang, 0],
          [min_ang, np.log10(yra[1])],
          [max_ang, np.log10(yra[1])],
          [max_ang, 0]]

path2 = Path(verts2)
patch2 = patches.PathPatch(path2, ec='none', fc=[0.7,0.7,0.7])
ax.add_patch(patch2)


plt.figtext(0.002, 0.69, "X component of residual response\n[$10^{-6}\ e$]")
#plt.rlabel("Response amplitude [$10^{-3}\ 3$]")
plt.xlabel("Angle of total response relative to field")

yval = 0.0485*4-0.01
plt.plot( [np.pi/2+0.01, yval], [0.4, 2.2], 'k', lw=2 )
twid = 0.025
tlen = 0.2
tvert = [ [ yval+twid, 2.2],
          [ yval-twid, 2.19],
          [ yval-twid/2., 2.2+tlen],
          [ yval+twid, 2.2] ]
path3 = Path(tvert)
patch3 = patches.PathPatch(path3, ec='none', fc=[0,0,0])
ax.add_patch(patch3)
plt.figtext(0.77, 0.585, r"$\vec{E}$", weight='bold', size='x-large')

#plt.yscale('log')
#plt.ylim(yra)

#ax.tick_params(axis='x', pad=100)

fig.set_size_inches(5,5)
plt.subplots_adjust(top=0.96, right=0.93, bottom=0.09, left=0.11)
plt.savefig("resid_plot.pdf")

plt.show()
raw_input('e')

##### now construct limit plot
np.random.seed(282649734)
#def nll( x, data, epsilon, n, nnuc, pfrac ):
#
#    L = n*nnuc*epsilon + x/2.*np.log(



def get_charge( nMC, epsilon, n, nnuc, pfrac ):

    mu1 = n*nnuc*pfrac
    mu2 = n*nnuc*(1-pfrac)

    if( mu1 < 1e4 ):
        num_p = np.random.poisson( mu1, size=nMC )
    else:
        num_p = np.random.normal( loc=mu1, scale=np.sqrt(mu1), size=nMC )

    if( mu2 < 1e4 ):
        num_e = np.random.poisson( mu2, size=nMC )
    else:
        num_e = np.random.normal( loc=mu2, scale=np.sqrt(mu2), size=nMC )

    raw_millicharge = epsilon*(num_p - num_e)
    ## now find only mod to nearest integer

    #raw_millicharge -= np.round(raw_millicharge)

    ## make sure number of bins is odd, so that we have a bin centered
    ## at zero
    #num_bins = (int(np.round(0.5/epsilon))/2)*2 + 1
    num_bins = 1e5 + 1

    hh, be = np.histogram(raw_millicharge, bins=num_bins, range=[-0.5, 0.5]) 
    hh = 1.0*hh/np.sum(hh)
    bc = be[:-1] + np.diff(be)/2.
        
    good_points = np.logical_and(hh > 0, bc > 0)
    bc = bc[good_points]
    hh = hh[good_points]

    if(pfrac == 0.5):
        hh[1:]*=2.0
    
    return bc, hh


def get_charge_analytic( epsilon, n, nnuc, pfrac):
    mu1 = n*pfrac*nnuc
    mu2 = n*(1-pfrac)*nnuc

    print mu1, mu2, mu1*epsilon, mu2*epsilon

    max_n = int( np.min([ 1e-3/epsilon, mu1*10]) )
    xvals = range(0, max_n+1)
    qpdf = sp.skellam.pmf( xvals, mu1, mu2 )

    xvals = np.array(xvals)
    ## fold back negative values assuming equal mu1, mu2 and
    ## symmetric around 0
    qpdf[1:] *= 2.0

    return xvals*epsilon, qpdf

## now make the limit plot
ep_vec = np.logspace(-8, 0, 1e2)
n_vec = np.logspace(-17, 0, 1e2)
if(do_morp or do_perl):
    n_vec = np.logspace(-23, 0, 1e2)

nMC = 1e4
nNuc = bu.num_nucleons
if( do_morp ):
    nNuc *= 1e6 * 0.3
if( do_perl ):
    nNuc_oil = 4.2e15
print nNuc
xx, yy = np.meshgrid(ep_vec, n_vec)

## make the list of x data
x_data = []
for i,d in enumerate(dir_list):
    c_zero = np.load( d + "/resid_data_0e.npy")
    x_data.append(c_zero)

x_data = np.array(x_data)
## save our data
#np.save( "our_resids.npy", x_data )


if(do_morp):
    x_data = np.load("morp_data.npy")
if( do_perl ):
    x_data = np.array([[0.25, 0.05],])

def calc_L( mu, sig, x, xpdf ):
    err_term = np.exp(-(x - np.abs(mu))**2/(2*sig**2))
    err_term[x < np.abs(mu)] = 1.    

    #xpdf[ xpdf < 1e-50 ] = 0.
    #xpdf[ np.isinf( xpdf ) ] = 0.
    nll = 2.0*( -np.log(np.sum( err_term*xpdf )) )

    # if (len(x)>1):
    #     plt.figure()
    #     plt.plot(x, xpdf)
    #     plt.plot(x, err_term)
    #     plt.show()

    return nll

plt.close('all')
plt.figure()

#now for each q, n calculate the distribution of charges on the
#bead
charge_mat = np.zeros_like(xx)
limit_curve = np.zeros_like(ep_vec)
for x_idx in range(len(ep_vec)):

    ## for this value of epsilon, cycle through all values of n and find
    ## the value of n at which the data are excluded at 1-alpha CL

    curr_L = np.zeros( len(n_vec) )
    
    for y_idx in range(len(n_vec)):

        #curr_charge = get_charge( nMC, ep_vec[x_idx], n_vec[y_idx], nNuc, pos_frac )
        #hh, be = np.histogram(curr_charge, bins=nMC/10+1, range=[-0.5, 0.5]) 
        #hh = 1.0*hh/np.sum(hh)
        #bc = be[:-1] + np.diff(be)/2.
        # if( True ):
        #     plt.figure()
        #     plt.plot(bc, hh)
        #     plt.show()
        
        #good_points = hh > 0
        #bc = bc[good_points]
        #hh = hh[good_points]


        bc, hh = get_charge(nMC, ep_vec[x_idx], n_vec[y_idx], nNuc, pos_frac )
        #bc, hh = get_charge_analytic( ep_vec[x_idx], n_vec[y_idx], nNuc, pos_frac )

        ## now evaluate likelihood at each charge pdf point
        L = 0.
        for j in range( len(x_data[:,0]) ):
            ## first evaluate for this data point at each possible charge
            ##print x_data[j,0], x_data[j,1]
            L += calc_L( x_data[j,0], x_data[j,1], bc, hh )


        if( do_perl ):
            L *= 42537104 ## num drops tested

        ### plt.figure()
        ### plt.plot( bc, L )
        ### plt.show()
        curr_L[y_idx] = L


    fpts = curr_L < curr_L[0]+5
    bad_diffs = np.argwhere(np.logical_not(fpts))

    for b in bad_diffs:
        if( b < np.argmin(curr_L) ):
            fpts[:b] = False
        else:
            fpts[b:] = False
    
    ## find 95% CL
    dnll = curr_L - curr_L[0]
    limit_curve[x_idx] = np.interp( CL, curr_L-curr_L[0], n_vec)

    plt.clf()
    #p = np.polyfit( n_vec[fpts], curr_L[fpts], 2)
    plt.semilogx( n_vec, curr_L, 'bo')
    plt.semilogx( n_vec[fpts], curr_L[fpts], 'ro')
    xx = np.linspace( n_vec[fpts][0], n_vec[fpts][-1], 1e3)
    #plt.semilogx( xx, np.polyval(p,xx), 'r')
    plt.ylim([curr_L[0],curr_L[0]+10])
    plt.xlim([np.min(n_vec), np.max(n_vec)])
    xx = plt.xlim()
    plt.plot(xx, [curr_L[0]+CL, curr_L[0]+CL], 'k--')
    yy = plt.ylim()
    plt.plot([limit_curve[x_idx], limit_curve[x_idx]], yy, 'g--')
    plt.title(str( ep_vec[x_idx] ) )
    plt.draw()
    plt.pause(0.5)
        

# plt.figure()
# #plt.pcolormesh( xx, yy, charge_mat, vmin=0, vmax = 1e-3)
# plt.contour(xx, yy, np.log10(charge_mat))
# plt.colorbar()
# plt.gca().set_xscale('log')
# plt.gca().set_yscale('log')

out_data = np.transpose(np.hstack((ep_vec, limit_curve)))
np.save(out_name, out_data)

plt.figure()
plt.loglog(ep_vec, limit_curve, '.-')

plt.show()
    




