import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import tplquad
import scipy.interpolate

def vol(rad):
    #returns volume of sphere of radius rad
    return 4./3.*np.pi*rad**3

def f(r):
    #Function thats part of yukawa potential between a sphere and point mass.
    return np.exp(-2.*r)*(1+r) + r - 1.

def V(rcc, rb, lam):
    #Potential for interaction between sphere and point mass. Need to multiply by alpha*G*m*pho to get into physical units.
    return -3./2.*np.exp(-(rcc-rb)/lam)*vol(lam)*f(rb/lam)/(rcc)

def f_z(x, y, z, rb, lam):
    #Force in the z direction between a sphere centered at the origin and point mass at x, y, z.
    rcc = np.sqrt(x**2 + y**2 + z**2)
    return (z/rcc)*((-1.*V(rcc, rb, lam)/lam) -V(rcc, rb, lam)/rcc)

def f_z_cant(cantobj, dz, rb, lam):
    xl = lambda x: cantobj.xmin
    xh = lambda x: cantobj.xmax
    yl = lambda x, y: cantobj.ymin
    yh = lambda x, y: cantobj.ymax
    fzint = lambda y, x, z: f_z(x, y, z-cantobj.zmax - dz, rb, lam)
    return tplquad(fzint, cantobj.zmin, cantobj.zmax, xl, xh, yl, yh, epsrel = 1e-2, epsabs = 1e-3)

class cant_geometry:
    #A class representing a cantilever's geometry. Methods compute force between the cantilever 

    def __init__(self, xr, yr, zr, rb, lam):
        self.xmin = xr[0]
        self.xmax = xr[1]
        self.ymin = yr[0]
        self.ymax = yr[1]
        self.zmin = zr[0]
        self.zmax = zr[1]
        self.rb = rb
        self.lam = lam
        self.zs = "Zrange for force curve not computed"
        self.fzs = "fzs not calculated"
        self.fzinterp = "z force interpolating spline not calculated"


    def fz_curve(self, zmax = 250, nz = 10, rtol = 0.1):
        #Calculates the force on a sphere in the z-direction as a function of z spacing from the cantilever.
        zmin = self.rb
        self.zs = np.arange(zmin, zmax, self.lam/3.)
        self.fzs = np.zeros(len(self.zs))
        fzer = lambda dz: f_z_cant(self, dz, self.rb, self.lam)[0]
        self.fzs[0] = fzer(self.zs[0])
        n = 0
        while self.fzs[n]/self.fzs[0]>rtol and n<len(self.zs):
            n += 1
            self.fzs[n] = fzer(self.zs[n])
        self.fzinterp = scipy.interpolate.interp1d(self.zs, self.fzs, kind = 'cubic')
        
