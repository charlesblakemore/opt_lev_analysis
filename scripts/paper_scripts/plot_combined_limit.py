import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import scipy.stats as sp
import bead_util as bu
from matplotlib.path import Path
import matplotlib.patches as patches
import cPickle as pickle

matplotlib.rc('font', family='serif') 
#matplotlib.rc('hatch', linewidth=2) 

name_list = ["lim_bead_comb_all_samesign.npy",
             "lim_bead_comb_all_opsign.npy",
             "lim_bead_comb_morp_samesign.npy",
             "lim_bead_comb_morp_opsign.npy"]

## output dictionary containing limits, for alex
output_dict = {}

# plt.figure()
# for f in name_list:

#     lims = np.load(f)
#     mid_val = len(lims)/2
#     plt.loglog( lims[:mid_val], lims[mid_val:] )



### comparison plot
fig=plt.figure()
ax = plt.gca()

dtype=(2,2)

##### Us first ####
min_us  = 40e-6


## now solid
#good_vals = np.logical_and(lims[:mid_val] > min_us, lims[:mid_val] < 1)
lims = np.load("lim_bead_comb_all_samesign.npy")
mid_val = len(lims)/2
lims = np.load("lim_bead_comb_all_opsign.npy")
mid_val = len(lims)/2

verts = []
for x,y in zip(lims[:mid_val-1], lims[mid_val:-1]):
    if( x > min_us and x < 0.2 ):
        verts.append([x,y])
verts.append( [1-min_us, verts[-1][1]] )
verts.append( [1-min_us, 1.5] )
verts.append( [verts[0][0], 1.5] )
verts.append( [verts[0][0], verts[0][1]] )

path = Path(verts)
patch = patches.PathPatch(path, ec='k', fc=[0.7, 0.7, 0.7], lw=2)
ax.add_patch(patch)

#plt.loglog( lims[:mid_val][good_vals], lims[mid_val:][good_vals], 'k' )
#yy = plt.ylim()
## make patch for our limit
#plt.loglog( [lims[:mid_val][good_vals][0],lims[:mid_val][good_vals][0]],
#            [lims[mid_val:][good_vals][0], 1], 'k' )



###### Now Perl et al #######
moil = 7.0e-9 ## g
nNuc_oil = 4.2e15
resid_oil = 2.5e-4 ## e, negligible error
max_dev_oil = 0.25 ## e
ep_vec = np.logspace(-8, np.log10(0.25), 2e2)


ax.add_patch(Rectangle((0.25, 1.9e-23), 0.5, 1.5, ec='b', fc=[0.85,0.85,1], lw=2, hatch="xx"))
limit_vec = np.zeros_like(ep_vec)
limit_vec_neut = np.zeros_like(ep_vec)
ndrops = 42537104
## find largest n for which we don't expect any drops at 95% CL
## in n drops
n_vec = np.logspace(-24, 0, 1e3)
mu_perl = 0.2
sig_perl = 0.05
for i,e in enumerate(ep_vec):
    print "Working on epsilon = ", e
    n25 = 0.25/e ##int(np.ceil(0.25/e))
    for n in n_vec:
        tot_prob = 1-(sp.poisson.cdf(n25, n*nNuc_oil)**ndrops) ##(1-sp.poisson.cdf( n25, n*nNuc_oil ))
        ## need to integrate against the n distribution before calculating the
        ## cdf
        if( n*nNuc_oil < 1 ):
            cstd = 5
        else:
            cstd = np.sqrt( n*nNuc_oil)
        ngrid = np.unique( np.round(np.linspace(0, n*nNuc_oil + 10*cstd,1e3)) )
        if( len(ngrid) < 2 ):
            ngrid = np.array([0., 1., 2., 3., 4., 5.])
        ncdf = sp.poisson.cdf( ngrid, n*nNuc_oil )
        nprob = np.diff(ncdf)
        nprob = np.hstack([ncdf[0], nprob])

        if np.abs( np.sum( nprob ) - 1.) > 0.05:
            print "Warning sum is low for: ", e, n, np.sum(nprob)
            
        prob_vec = np.exp(-( ngrid*e - mu_perl )**2/(2*sig_perl**2))
        prob_vec[ ngrid*e < mu_perl ] = 1.0
        marg_like = -2*ndrops*np.log(np.sum(prob_vec*nprob))

        #if( marg_like > 3.84 ):
        if(tot_prob > 0.05):
            limit_vec[i] = n
            print "limit at ", n
            break


for i,e in enumerate(ep_vec):
    print "Working on epsilon = ", e
    n25 = 0.25/e ##int(np.floor(0.25/e))
    for n in n_vec:
        mu = 0.5*n*nNuc_oil
        if(mu < 1e8 ):
            tot_prob = (1-sp.skellam.cdf( n25, mu, mu)**ndrops)
        else:
            tot_prob = (1-sp.norm.cdf( n25, 0, np.sqrt(2)*mu)**ndrops)
        if( tot_prob > 0.05 ):
            limit_vec_neut[i] = n
            break

## skellam disrtibution doesn't work for very large mu, so fix these points
fit_points = bu.inrange( ep_vec, 5e-5, 1e-4)
p = np.polyfit( np.log10(ep_vec[fit_points]), np.log10(limit_vec_neut[fit_points]),1)
first_fit_point = np.argwhere(fit_points)[0]
lvn_old = limit_vec_neut*1.0
limit_vec_neut[:first_fit_point] = 10**(np.polyval(p,np.log10(ep_vec[:first_fit_point])))

plt.loglog( ep_vec, limit_vec, 'b--', dashes=dtype, lw=2)
plt.loglog( 1.-ep_vec, limit_vec, 'b--', dashes=dtype, lw=2)

output_dict["perl_lims_op_sign"] = np.vstack((ep_vec, limit_vec))

plt.loglog( ep_vec, limit_vec_neut, 'b--', lw=2)
plt.loglog( 1.-ep_vec, limit_vec_neut, 'b--', lw=2)

output_dict["perl_lims_same_sign"] = np.vstack((ep_vec, limit_vec_neut))

#plt.loglog( ep_vec, lvn_old, 'b*')

# lims = np.load("lim_bead_comb_perl_samesign.npy")
# mid_val = len(lims)/2
# gpts = lims[:mid_val] < 0.5
# plt.loglog( lims[:mid_val][gpts], lims[mid_val:][gpts], 'r--', dashes=dtype, lw=2 )
# lims = np.load("lim_bead_comb_perl_opsign.npy")
# mid_val = len(lims)/2
# ep_vec = lims[:mid_val]*1.0
# limit_vec_neut = lims[mid_val:]*1.0
# ##gpts = np.logical_and(gpts, lims[mid_val:] < 0.9)  ## skip point with failed fit
# plt.loglog( lims[:mid_val][gpts], lims[mid_val:][gpts], 'r--', lw=2 )


##### Marinelli and Morpurgo ####
min_morp = 0.3
ax.add_patch(Rectangle((min_morp, 1.36e-21), 1.-2*min_morp, 1.5, ec='r', fc=[1, 0.85, 0.85], lw=2, hatch="//"))
lims = np.load("lim_bead_comb_morp_samesign.npy")
mid_val = len(lims)/2
gpts = lims[:mid_val] < 0.5
plt.loglog( lims[:mid_val][gpts], lims[mid_val:][gpts], 'r--', dashes=dtype, lw=2 )
plt.loglog( 1.-lims[:mid_val][gpts], lims[mid_val:][gpts], 'r--', dashes=dtype, lw=2 )

output_dict["morp_lims_same_sign"] = np.vstack((lims[:mid_val][gpts], lims[mid_val:][gpts]))

lims = np.load("lim_bead_comb_morp_opsign.npy")
mid_val = len(lims)/2

ep_vec = lims[:mid_val]*1.0
limit_vec_neut = lims[mid_val:]*1.0
gpts = np.logical_and(gpts, lims[mid_val:] < 0.9)  ## skip point with failed fit
plt.loglog( lims[:mid_val][gpts], lims[mid_val:][gpts], 'r--', lw=2 )
plt.loglog( 1.-lims[:mid_val][gpts], lims[mid_val:][gpts], 'r--', lw=2 )

output_dict["morp_lims_op_sign"] = np.vstack((lims[:mid_val][gpts], lims[mid_val:][gpts]))


#### Us again ###
#### Plot our lines last to make sure it's on top ####
lims = np.load("lim_bead_comb_all_samesign.npy")
mid_val = len(lims)/2
plt.loglog( lims[:mid_val-10], lims[mid_val:-10], 'k--', dashes=dtype, lw=2 )

output_dict["our_lims_same_sign"] = np.vstack((lims[:mid_val-10], lims[mid_val:-10]))

lims = np.load("lim_bead_comb_all_opsign.npy")
mid_val = len(lims)/2
dtype2=[6,5]
plt.loglog( lims[:mid_val-10], lims[mid_val:-10], 'k--', dashes=dtype2, lw=2 )

output_dict["our_lims_op_sign"] = np.vstack((lims[:mid_val-10], lims[mid_val:-10]))

ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_yticks([1, 1e-4, 1e-8, 1e-12, 1e-16, 1e-20, 1e-24])
plt.xlim([1e-6,1])
plt.ylim([1e-24, 1])
plt.xlabel("Fractional charge, $\epsilon$")
plt.ylabel("Abundance per nucleon, $n_\chi$")

fig.set_size_inches(5,3.333)
plt.subplots_adjust(top=0.95, right=0.95, bottom=0.165, left=0.175)
plt.savefig("limit_plot.eps")

of = open("limits.pkl", "wb")
pickle.dump( output_dict, of )
of.close()


plt.show()

