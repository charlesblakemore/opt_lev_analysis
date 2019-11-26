import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy.interpolate as interpolate
import scipy.optimize as optimize
from scipy.integrate import tplquad
import scipy, sys, time



def dist(p1, p2):
	return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)



def dist_p_arrp1(p1, xs, ys, zs):
	return np.sqrt((xs - p1[0])**2 + (ys - p1[1])**2 + (zs - p1[2])**2)

def dist_p_arrp2(p1, xs, ys, zs):
	xnew = (xs - p1[0])**2
	ynew = (ys - p1[1])**2
	znew = (zs - p1[2])**2
	return np.sqrt(np.add.outer(np.add.outer(xnew, ynew), znew))

xx = np.linspace(-250e-6, 250e-6, 499)
yy = np.linspace(-1000e-6, 0e-6, 999)
zz = np.linspace(-5e-6, 5e-6, 11)

xpoints, ypoints, zpoints = np.meshgrid(xx, yy, zz, indexing='ij')
xpoints = np.array(xpoints)
ypoints = np.array(ypoints)
zpoints = np.array(zpoints)

point = [0,0,0]

start1 = time.time()
dists1 = dist_p_arrp1(point, xpoints, ypoints, zpoints)
stop1 = time.time()

print('Old Method: ', stop1 - start1)


start2 = time.time()
dists2 = dist_p_arrp2(point, xx, yy, zz)
stop2 = time.time()

print('New Method: ', stop2 - start2)

