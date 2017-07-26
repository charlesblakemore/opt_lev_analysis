import math, sys
from scipy import integrate
import numpy as np

gap = float(sys.argv[1])
lam = float(sys.argv[2])
#zoff = float(sys.argv[3])

zoff_list = np.linspace(-100,100,501)*1e-6

print gap, lam

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

rb = D/2.0

def dV(phi,theta,r):
    return r**2 * math.sin(theta)

alpha = 1e15
G = 6.67398e-11 

def Vg_tot(currx,curry,currz):
    d = np.sqrt( currx**2 + curry**2 + currz**2 )
    Vout = alpha*lam/d*( np.exp(-(d + rb)/lam)*( lam**2 + lam*rb + rb**2) 
                                               - np.exp(-(d - rb)/lam)*( lam**2 - lam*rb + rb**2) )
    return Vout

fix_term = alpha*np.exp(-rb/lam)*( (lam**2 + rb**2) * (np.exp(2*rb/lam) -1) - lam*rb*(np.exp(2*rb/lam)) )
print fix_term

def Fz_tot(currx,curry,currz):
    x = currx + gap + rb + a_depth/2.0
    y = curry
    z = currz+zoff
    d = np.sqrt( x**2 + y**2 + z**2 )
    Fzout = fix_term*x*(lam+d)/d**3 * np.exp(-d/lam)
    return Fzout



curr_thick = a_depth
#intval = integrate.tplquad(Vg_tot, -a_depth/2.0+gap+rb, a_depth/2.0+gap+rb, lambda y: -a/4.0, lambda y: a/4.0, lambda y,z: -a/2.0+zoff, lambda y,z: a/2.0+zoff, epsrel=1e-2 )

force_list = []
for zoff in zoff_list:
    intval = integrate.tplquad(Fz_tot, -a_depth/2.0, a_depth/2.0, lambda y: -a/4.0, lambda y: a/4.0, lambda y,z: -a/2.0, lambda y,z: a/2.0, epsrel=1e-2 )

    #print intval

    integ = intval[0] * -2.*np.pi*G*rhob*rhoa/alpha
    integ_err = intval[1] * -2.*np.pi*G*rhob*rhoa/alpha

    force_list.append( [integ, integ_err] )

    #print "integral is: ", zoff, integ, integ_err

#curr_thick = au_thick
#intval_shield = integrate.tplquad(Fg_tot, -au_thick/2.0, au_thick/2.0, lambda y: -a/2.0, lambda y: a/2.0, lambda y,z: -a/2.0, lambda y,z: a/2.0, epsabs=1e-4, epsrel=1e-4)



fname = 'data/lam_arr_cu_%.3f_%.3f.npy' % (gap*1e6,lam*1e6)
np.save(fname,force_list)

         
                        


