#!/usr/local/bin/python
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy.interpolate as interpolate
from scipy.integrate import tplquad
import scipy, sys, time, math

def round_sig(x, sig=2):
	# round a number to certain number of sig figs
	if x == 0:
		return 0
	else:
		return round(x, sig-int(math.floor(math.log10(x)))-1)

def get_color_map(n):
	jet = plt.get_cmap('jet')
	cNorm = colors.Normalize(vmin=0, vmax=n)
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
	outmap = []
	for i in range(n):
		outmap.append( scalarMap.to_rgba(i) )
	return outmap

def dist(p1, p2):
	return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def dist_p_arrp(p1, xs, ys, zs):
	xnew = (xs - p1[0])**2
	ynew = (ys - p1[1])**2
	znew = (zs - p1[2])**2
	return np.sqrt(np.add.outer(np.add.outer(xnew, ynew), znew))

rhopath = '/Users/charlesblakemore/Stanford/beads/' + \
			'gravity/test_masses/attractor_v2/rho_arr.p'
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

alpha = 1.0e6
yuklambda = 100.0e-6    # m, Length scale of interaction
G = 6.67e-11          # m^3 / (kg s^2)

rhobead = 2200.       # kg / m^3

travel = 80.0e-6      # m
cent = 0.0e-6
Npoints = 50
beadposvec = np.linspace(cent - 0.5*travel, cent + 0.5*travel, Npoints)


BACKGROUND = 1.0e-18    # Blocked light background in N
SCALE = 1.0 / BACKGROUND




sepsweep = False
bead_seps = np.linspace(5.0e-6, 20.0e-6, 10)
rbead = 10e-6         # m

radiisweep = True
bead_rbeads = np.linspace(2.5e-6, 15e-6, 10)
sep = 10e-6           # m





if sepsweep:
	iterlist = bead_seps
if radiisweep:
	iterlist = bead_rbeads




detforce = np.zeros(len(iterlist))

Nparams = len(iterlist)

colors_yeay = get_color_map(Nparams)

forcediffs = []
for iterind, iterval in enumerate(iterlist):
	col = colors_yeay[iterind]
	if sepsweep:
		sep = iterval
		labval = round_sig(sep, 3)
		lab = 'Sep: ' + str(labval*1e6) + ' um'
	if radiisweep:
		rbead = iterval
		labval = round_sig(rbead, 3)
		lab = 'Rbead: ' + str(labval*1e6) + ' um'

	print('Params %i / %i :' % (iterind+1, Nparams), end=' ')

	############## INTEGRATE THINGS ###################

	func = np.exp(-2 * rbead / yuklambda) * (1 + rbead / yuklambda) + rbead / yuklambda - 1

	forcevpos = []
	for ind, xpos in enumerate(beadposvec):
		per = int(100. * float(ind) / float(Npoints))
		if not per % 10:
			print('.', end=' ') #str(per) + ',',
		sys.stdout.flush()
		beadpos = [xpos, sep+rbead, 0]

		s = dist_p_arrp(beadpos, xx, yy, zz) - rbead
		ysep = dist_p_arrp([0, sep+rbead, 0], xzeros, yy, zzeros)
		projection = ysep / (s + rbead)

		prefac = ((2. * G * m * rhobead * np.pi) / (3. * (rbead + s)**2)) 
		
		Gterm = 2. * rbead**3
		yukterm = 3 * yuklambda**2 * (rbead + s + yuklambda) * func * np.exp( - s / yuklambda )

		forces = prefac * projection * (Gterm + alpha * yukterm) 
		totforce = np.sum(forces)

		forcevpos.append(totforce)

	forcevpos = np.array(forcevpos)
	forcediffs.append(np.max(forcevpos) - np.min(forcevpos))
	plt.plot(beadposvec*1e6, forcevpos*1e15, marker='o', \
				linestyle='solid', color=col, label=lab, mew=0)
	print()

forcediffs = np.array(forcediffs)

if sepsweep:
	if yuklambda*1e6 > 1.0:
		title = 'Alpha: %g, Lambda: %g um, Rbead: %i um' % \
					(alpha, int(yuklambda*1e6), int(rbead*1e6))
	else:
		title = 'Alpha: %g, Lambda: %g um, Rbead: %i um' % \
					(alpha, yuklambda*1e6, int(rbead*1e6))
if radiisweep:
	if yuklambda*1e6 > 1.0:
		title = 'Alpha: %g, Lambda: %g um, Sep: %i um' % \
					(alpha, int(yuklambda*1e6), int(sep*1e6))
	else:
		title = 'Alpha: %g, Lambda: %g um, Sep: %i um' % \
					(alpha, yuklambda*1e6, int(sep*1e6))
plt.title(title, fontsize=18)
plt.xlabel('Cantilever Displacement from Center [um]', fontsize=14)
plt.ylabel('Force [fN]', fontsize=14)
plt.legend(loc=0, numpoints=1, ncol=2, fontsize=9)

plt.figure()
plt.plot(iterlist*1e6, forcediffs*1e15, lw=2)
if sepsweep:
	plt.xlabel('Cantilever Microsphere Separation [um]', fontsize=14)
if radiisweep:
	plt.xlabel('Bead Radius [um]', fontsize=14)

plt.title(title, fontsize=18)
plt.ylabel('Differential Force [fN]', fontsize=14)


respath = '/Users/charlesblakemore/Stanford/beads/gravity/data/'

if sepsweep:
	results_dir = pickle.load( open(respath + 'sepsweeps.p', 'rb') )
if radiisweep:
	results_dir = pickle.load( open(respath + 'radiisweeps.p', 'rb') )

if alpha not in results_dir:
	results_dir[alpha] = {}
if yuklambda not in results_dir[alpha]:
	results_dir[alpha][yuklambda] = {}

if sepsweep:
	results_dir[alpha][yuklambda][rbead] = (iterlist, forcediffs)
	pickle.dump(results_dir, open(respath + 'sepsweeps.p', 'wb'))
if radiisweep:
	results_dir[alpha][yuklambda][sep] = (iterlist, forcediffs)
	pickle.dump(results_dir, open(respath + 'radiisweeps.p', 'wb'))


plt.show()

input()






