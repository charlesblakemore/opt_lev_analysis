import time, sys, os
import dill as pickle

import numpy as np
import scipy.constants as constants
import scipy.interpolate as interp
import scipy.optimize as opti

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import grav_util_3 as gu
import bead_util as bu
import configuration as config

import warnings
warnings.filterwarnings("ignore")


posvec = np.linspace(-250.0, 250.0, 501)
xval = 20.0
zval = 28.6

ones = np.ones_like(posvec)

#pts = np.stack((xval*ones, posvec, zval*ones), axis=-1)
pts = [xval*ones, posvec, zval*ones]
p0 = [xval, 0.0, zval]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[0], pts[1], pts[2])




rot_angles = [1.0, 0.0, 0.0]
rot_angles = (np.pi / 180.0) * np.array(rot_angles)

rx = np.array([[1.0, 0.0, 0.0], \
               [0.0, np.cos(rot_angles[0]), -1.0*np.sin(rot_angles[0])], \
               [0.0, np.sin(rot_angles[0]), np.cos(rot_angles[0])]])

ry = np.array([[np.cos(rot_angles[1]), 0.0, np.sin(rot_angles[1])], \
               [0.0, 1.0, 0.0], \
               [-1.0*np.sin(rot_angles[1]), 0.0, np.cos(rot_angles[1])]])

rz = np.array([[np.cos(rot_angles[2]), -1.0*np.sin(rot_angles[2]), 0.0], \
               [np.sin(rot_angles[2]), np.cos(rot_angles[2]), 0.0], \
               [0.0, 0.0, 1.0]])


rxy = np.matmul(ry, rx)
rxyz = np.matmul(rz, rxy)

mesh_list = np.meshgrid(pts[0] - xval, posvec, pts[2] - zval, \
                                  indexing='ij')

print(rxyz)

new_pts = []
for resp in [0,1,2]:
    new_pts_vec = np.zeros_like(pts[resp])
    for resp2 in [0,1,2]:
        new_pts_vec += rxyz[resp,resp2] * (pts[resp2] - p0[resp2])
    new_pts.append(new_pts_vec)




ax.scatter(new_pts[0]+p0[0], new_pts[1]+p0[1], new_pts[2]+p0[2])
ax.set_xlim(0,50)
ax.set_ylim(-250,250)
ax.set_zlim(0,60)






plt.show()
