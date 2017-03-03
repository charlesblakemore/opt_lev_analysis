import math, sys, random, mcint
from scipy import integrate
import numpy as np

gap = float(sys.argv[1])
lam = float(sys.argv[2])

print gap, lam

## calculate the yukawa force over a distributed test mass assumed to be cube

D = 5 # diameter of bead (um)
rhob = 2e3 # density bead (kg/m^3)
rhoa = 19.3e3 # density attractor
rhosi = 2.3e3 # density attractor
a = 10 # length of attractor cube side (um)
a_depth = 200 # depth of attractor cube side (um)
au_thick = 0.2 # shield layer thickness (um)

def dV(phi,theta,r):
    return r**2 * math.sin(theta)

alpha = 1.0
G = 6.67398e-11 

#def Fg(phi, theta, r, currx, curry, currz):
def integrand(xin):
    r = xin[0]
    theta = xin[1]
    phi = xin[2]
    currx = xin[3]
    curry = xin[4]
    currz = xin[5]
    ## distance between r,theta,phi point and currx,curry,currz measured relative to center of cube at (gap + a/2, 0, 0) 
    dx = r*math.sin(theta)*math.cos(phi) - (gap + a_depth/2.0 + currx)    
    dy = r*math.sin(theta)*math.sin(phi) - curry
    dz = r*math.cos(theta) - currz
    dist = math.sqrt( dx**2 + dy**2 + dz**2 )        

    ##only want component in x direction (all others cancel)
    rhat_dot_xhat = abs(dx)/dist

    return (alpha*G*(rhoa-rhosi)/dist**2)*math.exp(-dist/lam)*(1.0 + dist/lam)*rhob*dV(phi,theta,r)*rhat_dot_xhat


def sampler():
    while True:
        r     = random.uniform(0.,D/2.)
        theta = random.uniform(0.,2.*math.pi)
        phi   = random.uniform(0.,math.pi)
        x = random.uniform(-a_depth/2.0+au_thick,a_depth/2.0+au_thick)
        y  = random.uniform(-a/2., a/2.)
        z = random.uniform(-a/2., a/2.)

        yield (r, theta, phi, x, y, z)

nmc = 100000000
domainsize = D * math.pi**2 * a_depth * a**2
random.seed(1)
result, error = mcint.integrate(integrand, sampler(), measure=domainsize, n=nmc)



print "integral is: ", result, error

#fname = 'data/lam_arr_%.3f_%.3f.npy' % (gap*1e6,lam*1e6)
#np.save(fname,intval)

         
                        


