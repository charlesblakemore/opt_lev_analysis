import numpy as np
import cPickle as pickle
import scipy, sys, time

import matplotlib.pyplot as plt

import bead_util as bu


plot = False

#path = '/data/comsol_outputs/patch_potentials/patch_pot_200nm_1Vrms_1_DATA.txt'

path = '/data/comsol_outputs/patch_potentials/patch_pot_2um_1Vrms_50um-deep-patches.txt'

parts = path.split('/')
name = parts[-1][:-4]

if name[-4:] == 'DATA':
    name = name[:-5]

xx_path = '/processed_data/comsol_data/patch_potentials/' + name + '.xx'
yy_path = '/processed_data/comsol_data/patch_potentials/' + name + '.yy'
zz_path = '/processed_data/comsol_data/patch_potentials/' + name + '.zz'
field_path = '/processed_data/comsol_data/patch_potentials/' + name + '.field'
pot_path = '/processed_data/comsol_data/patch_potentials/' + name + '.potential'

bu.make_all_pardirs(pot_path)

# Load a regularly-gridded dataset
fil = open(path, 'r')
lines = fil.readlines()
fil.close()


### CONVERT STUPID COMSOL OUTPUT TO SENSIBLE FORM
# Load grid points
linenum = 0
for line in lines:
    linenum += 1

    if line[0] == '%':
        continue

    if 'xx' not in locals():
        xx = [float(x) for x in line.split()]
        continue
    if 'yy' not in locals():
        yy = [float(x) for x in line.split()]
        continue
    if 'zz' not in locals():
        zz = [float(x) for x in line.split()]
        continue

    if 'xx' in locals() and 'yy' in locals() and 'zz' in locals():
        break


'''
print
print xx
raw_input()
print
print yy
raw_input()
print
print zz
raw_input()
'''


# Extract data from the remainder of the file
potential = np.zeros((len(xx), len(yy), len(zz)), dtype='float')
Ex = np.zeros((len(xx), len(yy), len(zz)), dtype='float')
Ey = np.zeros((len(xx), len(yy), len(zz)), dtype='float')
Ez = np.zeros((len(xx), len(yy), len(zz)), dtype='float')


pot_ind = linenum - 1
n_gridlines = len(yy) * len(zz)

Ex_ind = pot_ind + n_gridlines + 2
Ey_ind = Ex_ind + n_gridlines + 2
Ez_ind = Ey_ind + n_gridlines + 2


for j, y in enumerate(yy):
    for k, z in enumerate(zz):
        potline = lines[j + k*len(yy) + pot_ind]
        #print 'POT: ', potline
        #raw_input()
        new_potline = np.array([float(x) for x in potline.split()])
        badpts = np.isnan(new_potline)
        new_potline[badpts] = 1.0

        potential[:,j,k] = new_potline
        

        Exline = lines[j + k*len(yy) + Ex_ind]
        #print 'EXXX: ', Exline
        #raw_input()
        new_Exline = np.array([float(x) for x in Exline.split()])
        badpts = np.isnan(new_Exline)
        new_Exline[badpts] = 1.0

        Ex[:,j,k] = new_Exline
        

        Eyline = lines[j + k*len(yy) + Ey_ind]
        #print 'EYYY', Eyline
        #raw_input()
        new_Eyline = np.array([float(x) for x in Eyline.split()])
        badpts = np.isnan(new_Eyline)
        new_Eyline[badpts] = 1.0

        Ey[:,j,k] = new_Eyline
        

        Ezline = lines[j + k*len(yy) + Ez_ind]
        #print 'EZZZ', Ezline
        #raw_input()
        new_Ezline = np.array([float(x) for x in Ezline.split()])
        badpts = np.isnan(new_Ezline)
        new_Ezline[badpts] = 1.0

        Ez[:,j,k] = new_Ezline


print 'Done!'



## Transform COMSOL coordinate system to data/cantilever coordinate system
xx_tmp = np.copy(xx)
yy_tmp = np.copy(yy)

xx = yy_tmp
yy = xx_tmp

print Ex.shape, Ey.shape, Ez.shape

Ex = np.swapaxes(np.copy(Ex), 0, 1)
Ey = np.swapaxes(np.copy(Ey), 0, 1)
Ez = np.swapaxes(np.copy(Ez), 0, 1)
potential = np.swapaxes(np.copy(potential), 0, 1)

print Ex.shape, Ey.shape, Ez.shape

xx = -1.0 * xx[::-1]
Ex = np.flip(np.copy(Ex), 0)
Ey = np.flip(np.copy(Ey), 0)
Ez = np.flip(np.copy(Ez), 0)
potential = np.flip(np.copy(potential), 0)

#print xx

tempEx = np.copy(Ex)
Ex = -1.0 * np.copy(Ey)
Ey = np.copy(tempEx)

del tempEx

field = np.stack((Ex, Ey, Ez), axis=0)

np.save(open(xx_path, 'wb'), xx)
np.save(open(yy_path, 'wb'), yy)
np.save(open(zz_path, 'wb'), zz)
np.save(open(field_path, 'wb'), field)
np.save(open(pot_path, 'wb'), potential)

print field_path


if plot:
    yz_plane = np.meshgrid(yy, zz, indexing='ij')

    levels = np.linspace(np.min(potential), np.max(potential), 100)# * 0.1

    for i in range(10):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cont = ax.contourf(yz_plane[0]*1e6, yz_plane[1]*1e6, potential[25+i,:,:], levels)
        cbar = plt.colorbar(cont)
        ax.quiver(yz_plane[0][::25,::5]*1e6, yz_plane[1][::25,::5]*1e6, \
                  Ey[25+i,::25,::5], Ez[50+i,::25,::5], color='k')
        ax.set_xlabel('Displaccment Along Cantilever [um]')
        ax.set_ylabel('Height [um]')
        plt.tight_layout()
    plt.show()

