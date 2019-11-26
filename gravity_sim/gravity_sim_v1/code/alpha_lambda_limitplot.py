import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy.interpolate as interpolate
import scipy.optimize as optimize
from scipy.integrate import tplquad
import scipy, sys, time

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


BACKGROUND = 1.0e-19    # Blocked light background in N
SCALE = 1.0 / BACKGROUND

rbead = 10.0e-6          # m
sep = 7.0e-6           # m

lambdas = np.logspace(-6.3, -3, 50)
lambdas = lambdas[::-1]
alphas = np.zeros(len(lambdas))

G = 6.67e-11            # m^3 / (kg s^2)
rhobead = 2200.

travel = 80.0e-6
cent = 0.0e-6
Npoints = 80.
beadposvec = np.linspace(cent - 0.5*travel, cent + 0.5*travel, Npoints)

Gterm = 2. * rbead**3



Gforcecurve = []
for ind, xpos in enumerate(beadposvec):
	beadpos = [xpos, sep+rbead, 0]

	s = dist_p_arrp(beadpos, xx, yy, zz) - rbead
	ysep = dist_p_arrp([0, sep+rbead, 0], np.zeros(len(xx)), yy, np.zeros(len(zz)))
	projection = ysep / (s + rbead)

	prefac = ((2. * G * m * rhobead * np.pi) / (3. * (rbead + s)**2))

	totforce = np.sum(prefac * Gterm * projection)
	Gforcecurve.append(totforce)

Gdiff = np.max(Gforcecurve) - np.min(Gforcecurve)
print("Computed gravitational contribution.")
sys.stdout.flush()


#def calculate_yukdiff(yuklambda, beadposvec=beadposvec, rbead=rbead, sep=sep, \
#						m=m, coords=(xx, yy, zz))
#	xx = coords[0]
#	yy = coords[1]
#	zz = coords[2]


for yukind, yuklambda in enumerate(lambdas):
	per = int(100. * float(yukind) / float(len(lambdas)))
	if not per % 1:
		print(str(per) + ',', end=' ')
	sys.stdout.flush()

	func = np.exp(-2. * rbead / yuklambda) * (1. + rbead / yuklambda) + rbead / yuklambda - 1.

	yukforcecurve = []
	for ind, xpos in enumerate(beadposvec):
		beadpos = [xpos, sep+rbead, 0]

		#s = dist_p_arrp(beadpos, xpoints, ypoints, zpoints) - rbead
		s = dist_p_arrp(beadpos, xx, yy, zz) - rbead
		#ysep = dist_p_arrp([0, sep+rbead, 0], xyseppoints, yyseppoints, zyseppoints)
		ysep = dist_p_arrp([0, sep+rbead, 0], xzeros, yy, zzeros)
		projection = ysep / (s + rbead)

		prefac = ((2. * G * m * rhobead * np.pi) / (3. * (rbead + s)**2))

		yukterm = 3 * yuklambda**2 * (rbead + s + yuklambda) * func * np.exp( - s / yuklambda )

		totforce = np.sum(prefac * yukterm * projection)
		yukforcecurve.append(totforce)

	yukdiff = np.max(yukforcecurve) - np.min(yukforcecurve)


	fitfun = lambda alphapow: np.abs( (Gdiff + 1.0 * 10**alphapow * yukdiff)*SCALE \
										- BACKGROUND*SCALE )

	guess = np.log10((1.0 - Gdiff*SCALE) / (yukdiff * SCALE))

	#fit = optimize.minimize(fitfun, guess, bounds=[(guess*0.5,guess*2.0)])
	#if verbose:
	#	print fit
	#	if pause:
	#		raw_input()
	#newpow = fit.x[0]

	#if verbose:
	#	print "Guess: ", guess
	#	print "New power: ", newpow

	alphas[yukind] = 10.0**guess


respath = '/Users/charlesblakemore/Stanford/beads/gravity/data/'
results_dir = pickle.load( open(respath + 'alpha_lambda_estimates_grfp.p', 'rb') )

if rbead not in results_dir:
	results_dir[rbead] = {}
if sep not in results_dir[rbead]:
	results_dir[rbead][sep] = {}

results_dir[rbead][sep][BACKGROUND] = (lambdas, alphas)

pickle.dump( open(respath + 'alpha_lambda_estimates_grfp.p', 'wb') )


plt.loglog(lambdas, alphas)


plt.show()


input()



























# Clean up holes in density profile
























