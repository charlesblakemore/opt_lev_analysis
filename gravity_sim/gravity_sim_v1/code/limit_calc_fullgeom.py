import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import scipy.interpolate as interpolate
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

BACKGROUND = 1.0e-18    # Blocked light background in N
SCALE = 1.0 / BACKGROUND


bead_seps = np.linspace(5.0e-6, 20.0e-6, 6)
bead_rbeads = np.linspace(2.5e-6, 15e-6, 6)

detforce = np.zeros((len(bead_seps), len(bead_rbeads)))

Ncombos = len(bead_seps) * len(bead_rbeads)
counter = 0

colors_yeay = get_color_map(len(bead_seps))

for sepind, sep in enumerate(bead_seps):
	col = colors_yeay[-1 * sepind]

	for beadind, rbead in enumerate(bead_rbeads):

		counter += 1
		print('Params %i / %i :' % (counter, Ncombos), end=' ')

		#if sep < 2.0 * rbead:
		#	detforce[sepind, beadind] = 0
		#	continue

		############## INTEGRATE THINGS ###################
		lab = 'Sep: ' + str(sep) + ', Rbead: ' + str(rbead)

		alpha = 1.0e6         # Strength of Yukawa interaction relative to gravity
		yuklambda = 1.0e-4    # m, Length scale of interaction
		G = 6.67e-11          # m^3 / (kg s^2)

		#rbead = 2.5e-6        # m
		rhobead = 2200.       # kg / m^3

		#sep = 10.0e-6         # m

		travel = 80.0e-6      # m
		cent = 0.0e-6
		Npoints = 50
		beadposvec = np.linspace(cent - 0.5*travel, cent + 0.5*travel, Npoints)

		func = np.exp(-2 * rbead / yuklambda) * (1 + rbead / yuklambda) + rbead / yuklambda - 1

		force1 = []
		for ind, xpos in enumerate(beadposvec):
			per = int(100. * float(ind) / float(Npoints))
			if not per % 10:
				print('.', end=' ') #str(per) + ',',
			sys.stdout.flush()
			beadpos = [xpos, sep+rbead, 0]

			s = dist_p_arrp(beadpos, xx, yy, zz) - rbead
			ysep = dist_p_arrp([0, sep+rbead, 0], xzeros, yy, zzeros)

			prefac = ((2. * G * m * rhobead * np.pi) / (3. * (rbead + s)**2)) 
			sumterms = 2. * rbead**3 + 3 * alpha * yuklambda * (rbead + s + yuklambda) * \
				( (rbead - yuklambda) * np.exp(-s / yuklambda) + (rbead + yuklambda) * np.exp(-(2.*rbead + s) / yuklambda) )

			forces = prefac * sumterms * (ysep / (s+rbead))
			totforce1 = np.sum(forces)

			sys.stdout.flush()

			force1.append(totforce1)

		deriv = np.gradient(force1)
		zeros = np.where(np.diff(np.signbit(deriv)))[0]

		print()

		if len(zeros) == 1:
			detforce[sepind, beadind] = 0.
			continue

		cdetforce = np.max(force1) - np.min(force1)
		detforce[sepind, beadind] = cdetforce

		plt.plot(beadposvec*1e6, np.array(force1)*1e15, color=col, lw=beadind+1)#, label=lab)

plt.xlabel('Displacement from Center [um]')
plt.ylabel('Gravity + Yukawa Force [fN]')
plt.legend(loc=0, numpoints=1)

plt.figure()
plt.contourf(np.array(bead_rbeads)*1e6, np.array(bead_seps)*1e6, detforce*1e15, 50)
plt.xlabel("Bead Radius [um]")
plt.ylabel("Cantilever Bead Separation [um]")
plt.colorbar()

plt.show()





















# Clean up holes in density profile
























