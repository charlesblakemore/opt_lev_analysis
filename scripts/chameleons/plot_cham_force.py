import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu

cham_force = np.loadtxt("cham_vs_x.txt",skiprows=8)

#scale into N
xx = np.linspace(1,1000,1e3)
cdrive = bu.get_chameleon_force( xx*1e-6 )

sfac = np.interp(-10.,cham_force[:,0],cham_force[:,1])/np.interp(10.,xx,cdrive.flatten())

#cham_force[:,0] = -1.0*cham_force[::-1,0]
#cham_force[:,1] = cham_force[::-1,1]/sfac
cham_force = np.zeros( (len(xx),2) )
cham_force[:,0] = xx
cham_force[:,1] = cdrive

plt.figure()
plt.plot(cham_force[:,0], cham_force[:,1])
plt.plot( xx, cdrive )
plt.show()

closest_list = np.linspace(20., 1000., 50)
deltaf = np.zeros_like(closest_list)

delta = 300.
for i,c in enumerate(closest_list):
    deltaf[i] = np.interp( c, cham_force[:,0], cham_force[:,1] ) - np.interp( c+delta, cham_force[:,0], cham_force[:,1] )
    print c, deltaf[i]
print cham_force[:,0]

plt.figure()
plt.semilogy( closest_list, deltaf, 'ko-' )
plt.semilogy( closest_list, 0.9e-20*(1/closest_list-1/(closest_list+delta)), 'ro-' )

plt.show()
