#!/usr/bin/python

import numpy as np
import pickle as pickle
import scipy.interpolate as interpolate
import scipy, sys, time

rbead = float(sys.argv[1])
sep = float(sys.argv[2])
height = float(sys.argv[3])


rhopath = '/farmshare/user_data/cblakemo/gravity_sim/test_masses/attractor_v2/rho_arr.p'
rho, xx, yy, zz = pickle.load(open(rhopath, 'rb'))
print("Density Loaded.")
sys.stdout.flush()

xx = np.array(xx)
yy = np.array(yy)
zz = np.array(zz)

xzeros = np.zeros(len(xx))
zzeros = np.zeros(len(zz))

dx = np.abs(xx[1] - xx[0])
dy = np.abs(yy[1] - yy[0])
dz = np.abs(zz[1] - zz[0])

cell_volume = dx * dy * dz
m = rho * cell_volume

G = 6.67e-11     # m^3 / (kg s^2)
rhobead = 2200.

travel = 500.0e-6
cent = 0.0e-6
Npoints = 1001.
beadposvec = np.linspace(cent - 0.5*travel, cent + 0.5*travel, Npoints)

lambdas = np.logspace(-6.3, -3, 100)
lambdas = lambdas[::-1]


respath = '/farmshare/user_data/cblakemo/sim_results/'
respath = respath + 'rbead_' + str(rbead)
respath = respath + '_sep_' + str(sep)
respath = respath + '_height_' + str(height)
respath = respath + '.p'
results_dic = {}
results_dic['order'] = 'Rbead, Sep, Height, Yuklambda'

pickle.dump([], open(respath + '.test0', 'wb'))


def dist(p1, p2):
	return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def dist_p_arrp(p1, xs, ys, zs):
	xnew = (xs - p1[0])**2
	ynew = (ys - p1[1])**2
	znew = (zs - p1[2])**2
	return np.sqrt(np.add.outer(np.add.outer(xnew, ynew), znew))


results_dic[rbead] = {}
results_dic[rbead][sep] = {}
results_dic[rbead][sep][height] = {}


Gterm = 2. * rbead**3


#allseps = []
#for ind, xpos in enumerate(beadposvec):
#	beadpos = [xpos, sep+rbead, height]
#
#	s = dist_p_arrp(beadpos, xx, yy, zz) - rbead
#	ysep = dist_p_arrp([0, sep+rbead, 0], np.zeros(len(xx)), yy, np.zeros(len(zz)))
#	zsep = dist_p_arrp([0, 0, height], np.zeros(len(xx)), np.zeros(len(yy)), zz)
#	yprojection = ysep / (s + rbead)
#	zprojection = zsep / (s + rbead)
#
#	allseps.append((s, yprojection, zprojection))
#
#print 'Computed seps'
#sys.stdout.flush()

GforcecurveX = []
GforcecurveZ = []
for ind, xpos in enumerate(beadposvec):
	beadpos = [xpos, sep+rbead, height]

	s = dist_p_arrp(beadpos, xx, yy, zz) - rbead
	ysep = dist_p_arrp([0, sep+rbead, 0], np.zeros(len(xx)), yy, np.zeros(len(zz)))
	zsep = dist_p_arrp([0, 0, height], np.zeros(len(xx)), np.zeros(len(yy)), zz)
	yprojection = ysep / (s + rbead)
	zprojection = zsep / (s + rbead)

	#s = allseps[ind][0]
	#yprojection = allseps[ind][1]
	#zprojection = allseps[ind][2]

	prefac = ((2. * G * m * rhobead * np.pi) / (3. * (rbead + s)**2))

	ytotforce = np.sum(prefac * Gterm * yprojection)
	ztotforce = np.sum(prefac * Gterm * zprojection)

	GforcecurveX.append(ytotforce)
	GforcecurveZ.append(ztotforce)

GforcecurveX = np.array(GforcecurveX)
GforcecurveZ = np.array(GforcecurveZ)

print('Computed normal grav')
sys.stdout.flush()



pickle.dump([], open(respath + '.test1', 'wb'))


for yukind, yuklambda in enumerate(lambdas):
	per = int(100. * float(yukind) / float(len(lambdas)))
	if not per % 1:
		print(str(per) + ',', end=' ')
	sys.stdout.flush()

	func = np.exp(-2. * rbead / yuklambda) * (1. + rbead / yuklambda) + rbead / yuklambda - 1.

	yukforcecurveX = []
	yukforcecurveZ = []
	for ind, xpos in enumerate(beadposvec):

		beadpos = [xpos, sep+rbead, height]

		s = dist_p_arrp(beadpos, xx, yy, zz) - rbead
		ysep = dist_p_arrp([0, sep+rbead, 0], np.zeros(len(xx)), yy, np.zeros(len(zz)))
		zsep = dist_p_arrp([0, 0, height], np.zeros(len(xx)), np.zeros(len(yy)), zz)
		yprojection = ysep / (s + rbead)
		zprojection = zsep / (s + rbead)

		#s = allseps[ind][0]
		#yprojection = allseps[ind][1]
		#zprojection = allseps[ind][2]

		prefac = ((2. * G * m * rhobead * np.pi) / (3. * (rbead + s)**2))

		yukterm = 3 * yuklambda**2 * (rbead + s + yuklambda) * func * np.exp( - s / yuklambda )

		ytotforce = np.sum(prefac * yukterm * yprojection)
		ztotforce = np.sum(prefac * yukterm * zprojection)

		yukforcecurveX.append(ytotforce)
		yukforcecurveZ.append(ztotforce)

	yukforcecurveX = np.array(yukforcecurveX)
	yukforcecurveZ = np.array(yukforcecurveZ)

	results_dic[rbead][sep][height][yuklambda] = \
		(GforcecurveX, GforcecurveZ, yukforcecurveX, yukforcecurveZ)

results_dic['posvec'] = beadposvec

pickle.dump(results_dic, open(respath, 'wb') )
