import math, sys
from scipy import integrate
import numpy as np

gap = float(sys.argv[1])
lam = float(sys.argv[2])

print(gap, lam)

## calculate potential over cylindrical mass as function of position

D = 5e-6 # diameter of bead (m)
rhob = 2e3 # density bead (kg/m^3)
rhoa = 20e3 # density attractor
a = 20e-6 # length of attractor cube side (m)
##gap = 7.5e-6 # gap between cube face and bead center

def dV(phi,theta,r):
    return r**2 * math.sin(theta)

alpha = 1.0
G = 6.67398e-11 

def Fg(phi, theta, r, currx, curry, currz):
    ## distance between r,theta,phi point and currx,curry,currz measured relative to center of cube at (gap + a/2, 0, 0) 
    dx = r*math.sin(theta)*math.cos(phi) - (gap + a/2.0 + currx)    
    dy = r*math.sin(theta)*math.sin(phi) - curry
    dz = r*math.cos(theta) - currz
    dist = math.sqrt( dx**2 + dy**2 + dz**2 )        

    return (alpha*G*rhoa/dist)*math.exp(-dist/lam)*rhob*dV(phi,theta,r)

def Fg_tot(z,y,x):
    def Fg_curr(phi, theta,r):
        return Fg(phi,theta,r,x,y,z)
    f1 = integrate.tplquad(Fg_curr, 0.0, D/2.0, lambda y: 0.0, lambda y: math.pi, lambda y,z: 0.0, lambda y,z: 2.0*math.pi)
    return f1[0]



intval = integrate.tplquad(Fg_tot, -a/2.0, a/2.0, lambda x: -a/2.0, lambda x: a/2.0, lambda x,y: -math.sqrt(a**2 - x**2), lambda x,y: math.sqrt(a**2 - x**2), epsabs=1e-4, epsrel=1e-4)
    
fname = 'lam_arr_pot_cyl_%.3f_%.3f.npy' % (gap*1e6,lam*1e6)
np.save(fname,intval)

         
                        


