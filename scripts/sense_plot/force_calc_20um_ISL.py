import math, sys
from scipy import integrate
import numpy as np

gap = float(sys.argv[1])
lam = float(sys.argv[2])
#zoff = float(sys.argv[3])


print(gap, lam)

## calculate the yukawa force over a distributed test mass assumed to be cube

D = 20e-6 # diameter of bead (m)
rhob = 2e3 # density bead (kg/m^3)
rhoa = 19.3e3 # density attractor
rhosi = 2.3e3 # density attractor
a = 20e-6 # length of attractor cube side (m)
a_depth = 2000e-6 # depth of attractor cube side (m)
au_thick = 1e-6 # shield layer thickness (m)
##gap = 7.5e-6 # gap between cube face and bead center

zoff_list = np.arange(6)*a

rb = D/2.0

alpha = 0. #1e15
G = 6.67398e-11 

curr_thick = a_depth

def vol(r):
    return 4./3*np.pi*r**3
def fv(r):
    return np.exp(-2*r)*(1+r) + r - 1.

A1 = vol(rb)
A2 = 1.5*alpha*vol(lam)*fv(rb/lam)

def Fz_tot(currz,curry,currx):
    x = currx + gap + rb
    y = curry
    z = currz+zoff
    d = np.sqrt( x**2 + y**2 + z**2 )
    Fzout = x/d**3 * (A1 + A2*np.exp(-(d-rb)/lam)*(1.+d/lam))
    return Fzout

force_list = []
for zoff in zoff_list:
    intval = integrate.tplquad(Fz_tot, 0, a_depth, lambda y: -10e-6, lambda y: 10e-6, lambda y,z: -a/2.0, lambda y,z: a/2.0, epsrel=1e-2 )

    #print intval

    integ = intval[0] * G*rhob #/alpha
    integ_err = intval[1] * G*rhob #/alpha

    force_list.append( [integ, integ_err] )

    #print "integral is: ", zoff, integ, integ_err

#curr_thick = au_thick
#intval_shield = integrate.tplquad(Fg_tot, -au_thick/2.0, au_thick/2.0, lambda y: -a/2.0, lambda y: a/2.0, lambda y,z: -a/2.0, lambda y,z: a/2.0, epsabs=1e-4, epsrel=1e-4)

force_list = np.array(force_list)

## combine with neighboring
fcent = (force_list[0,0]+2*force_list[2,0]+2*force_list[4,0])*rhoa - (2*force_list[1,0]+2*force_list[3,0]+2*force_list[5,0])*rhosi 

print(force_list)
print(fcent)
fname = 'data_20um/lam_arr_20um_ISL_%.3f_%.3f.npy' % (gap*1e6,lam*1e6)
np.save(fname,fcent)

         
                        


