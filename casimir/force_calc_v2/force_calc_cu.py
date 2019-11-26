import math, sys
from scipy import integrate
import numpy as np

gap = float(sys.argv[1])
lam = float(sys.argv[2])
xoff = float(sys.argv[3])

print(gap, lam)

## calculate the yukawa force over a distributed test mass assumed to be cube

D = 5e-6 # diameter of bead (m)
rhob = 2e3 # density bead (kg/m^3)
rhoa = 19.3e3 # density attractor
#rhosi = 2.3e3 # density attractor
rhosi = 8.96e3 # density of copper
a = 10e-6 # length of attractor cube side (m)
a_depth = 10e-6 # depth of attractor cube side (m)
au_thick = 0.2e-6 # shield layer thickness (m)
##gap = 7.5e-6 # gap between cube face and bead center

def dV(phi,theta,r):
    return r**2 * math.sin(theta)

alpha = 1.0
G = 6.67398e-11 

def Fg(phi, theta, r, currx, curry, currz):
    ## distance between r,theta,phi point and currx,curry,currz measured relative to center of cube at (gap + a/2, 0, 0) 
    dx = r*math.sin(theta)*math.cos(phi) - (gap + curr_thick/2.0 + currx)    
    dy = r*math.sin(theta)*math.sin(phi) - curry
    dz = r*math.cos(theta) - currz
    dist = math.sqrt( dx**2 + dy**2 + dz**2 )        

    ##only want component in x direction (all others cancel)
    rhat_dot_xhat = abs(dx)/dist

    return (alpha*G*(rhoa-rhosi)/dist**2)*math.exp(-dist/lam)*(1.0 + dist/lam)*rhob*dV(phi,theta,r)*rhat_dot_xhat

def Fg_tot(z,y,x):
    def Fg_curr(phi, theta,r):
        return Fg(phi,theta,r,x,y,z)
    f1 = integrate.tplquad(Fg_curr, 0.0, D/2.0, lambda y: 0.0, lambda y: math.pi, lambda y,z: 0.0, lambda y,z: 2.0*math.pi, epsabs=1e-2, epsrel=1e-2)
    return f1[0]

curr_thick = a_depth
intval = integrate.tplquad(Fg_tot, -a_depth/2.0, a_depth/2.0, lambda y: -a/4.0, lambda y: a/4.0, lambda y,z: -a/2.0, lambda y,z: a/2.0, epsabs=1e-2, epsrel=1e-2)

#curr_thick = au_thick
#intval_shield = integrate.tplquad(Fg_tot, -au_thick/2.0, au_thick/2.0, lambda y: -a/2.0, lambda y: a/2.0, lambda y,z: -a/2.0, lambda y,z: a/2.0, epsabs=1e-4, epsrel=1e-4)

print("integral is: ", intval)

fname = 'data/lam_arr_cu_%.3f_%.3f.npy' % (gap*1e6,lam*1e6)
np.save(fname,intval)

         
                        


