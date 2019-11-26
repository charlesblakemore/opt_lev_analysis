#!/usr/bin/python

import numpy as np
import pickle as pickle
import scipy.interpolate as interp
import scipy.signal as signal
import scipy.optimize as opti
import scipy, sys, time

import matplotlib.pyplot as plt

#rbead = float(sys.argv[1])
#sep = float(sys.argv[2])
#height = float(sys.argv[3])


rbead = 2.40e-6
sep = 5e-6
height = 2.0e-6

rhopath = '/home/charles/gravity/test_masses/attractor_v2/rho_arr.p'
rho, xx, yy, zz = pickle.load(open(rhopath, 'rb'))
print("Density Loaded.")
sys.stdout.flush()

xx = np.array(xx)
yy = np.array(yy)
zz = np.array(zz)

xinds = np.abs(xx) <= 25.0e-6
yinds = np.abs(yy) <= 100.0e-6
zinds = np.abs(zz) <= 5.0e-6

xx2 = xx[xinds]
yy2 = yy[yinds]
zz2 = zz[zinds]
rho2 = rho[xinds,:,:][:,yinds,:]

xzeros = np.zeros(len(xx))
yzeros = np.zeros(len(yy))
zzeros = np.zeros(len(zz))

xzeros2 = np.zeros(len(xx2))
yzeros2 = np.zeros(len(yy2))
zzeros2 = np.zeros(len(zz2))

dx = np.abs(xx[1] - xx[0])
dy = np.abs(yy[1] - yy[0])
dz = np.abs(zz[1] - zz[0])

cell_volume = dx * dy * dz
m = rho * cell_volume
m2 = rho2 * cell_volume

G = 6.67e-11     # m^3 / (kg s^2)
rhobead = 2200.

travel = 500.0e-6
cent = 0.0e-6
Npoints = 1001.
beadposvec = np.linspace(cent - 0.5*travel, cent + 0.5*travel, Npoints)

beadposvec2 = np.linspace(-500e-6, 500e-6, 2001)


lambdas = np.logspace(-6.3, -3, 100)
lambdas = lambdas[::-1]

lambdas = np.array(lambdas[4:6])


respath = '/farmshare/user_data/cblakemo/sim_results/'
respath = respath + 'rbead_' + str(rbead)
respath = respath + '_sep_' + str(sep)
respath = respath + '_height_' + str(height)
respath = respath + '.p'
results_dic = {}
results_dic2 = {}
results_dic['order'] = 'Rbead, Sep, Height, Yuklambda'

#pickle.dump([], open(respath + '.test0', 'wb'))


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def dist_p_arrp(p1, xs, ys, zs):
    xnew = (xs - p1[0])**2
    ynew = (ys - p1[1])**2
    znew = (zs - p1[2])**2
    val = np.sqrt(np.add.outer(np.add.outer(xnew, ynew), znew))
    return val


results_dic[rbead] = {}
results_dic[rbead][sep] = {}
results_dic[rbead][sep][height] = {}


results_dic2[rbead] = {}
results_dic2[rbead][sep] = {}
results_dic2[rbead][sep][height] = {}

Gterm = 2. * rbead**3


'''
start = time.time()

GforcecurveX = []
GforcecurveY = []
GforcecurveZ = []
for ind, xpos in enumerate(beadposvec):
    beadpos = [xpos, sep+rbead, height]

    s = dist_p_arrp(beadpos, xx, yy, zz) - rbead

    # These are used to compute projections and thus need to maintain sign
    xsep = dist_p_arrp([xpos, 0, 0], xx, yzeros, zzeros)
    xind = np.argmin(np.abs(xx - xpos))
    xsep[:xind,:,:] *= -1.0

    # We a priori know that all yseps should be negative
    ysep = dist_p_arrp([0, sep+rbead, 0], xzeros, yy, zzeros)
    ysep *= -1.0

    zsep = dist_p_arrp([0, 0, height], xzeros, yzeros, zz)
    zind = np.argmin(np.abs(zz - height))
    zsep[:,:,:zind] *= -1.0

    xprojection = xsep / (s + rbead)
    yprojection = ysep / (s + rbead)
    zprojection = zsep / (s + rbead)

    #s = allseps[ind][0]
    #yprojection = allseps[ind][1]
    #zprojection = allseps[ind][2]

    prefac = ((2. * G * m * rhobead * np.pi) / (3. * (rbead + s)**2))

    xtotforce = np.sum(prefac * Gterm * xprojection)
    ytotforce = np.sum(prefac * Gterm * yprojection)
    ztotforce = np.sum(prefac * Gterm * zprojection)

    # SWAP X AND Y AXES TO MATCH DATA AXES
    GforcecurveX.append(ytotforce)
    GforcecurveY.append(xtotforce)
    GforcecurveZ.append(ztotforce)

GforcecurveX = np.array(GforcecurveX)
GforcecurveY = np.array(GforcecurveY)
GforcecurveZ = np.array(GforcecurveZ)

print 'Computed normal grav'
sys.stdout.flush()



for yukind, yuklambda in enumerate(lambdas):
    per = int(100. * float(yukind) / float(len(lambdas)))
    if not per % 1:
        print str(per) + ',',
    sys.stdout.flush()

    func = np.exp(-2. * rbead / yuklambda) * (1. + rbead / yuklambda) + rbead / yuklambda - 1.

    yukforcecurveX = []
    yukforcecurveY = []
    yukforcecurveZ = []
    for ind, xpos in enumerate(beadposvec):

        beadpos = [xpos, sep+rbead, height]

        # Always a positive number by definition
        s = dist_p_arrp(beadpos, xx, yy, zz) - rbead

        # These are used to compute projections and thus need to maintain sign
        xsep = dist_p_arrp([xpos, 0, 0], xx, yzeros, zzeros)
        xind = np.argmin(np.abs(xx - xpos))
        xsep[:xind,:,:] *= -1.0

        # We a priori know that all yseps should be negative
        ysep = dist_p_arrp([0, sep+rbead, 0], xzeros, yy, zzeros)
        ysep *= -1.0

        zsep = dist_p_arrp([0, 0, height], xzeros, yzeros, zz)
        zind = np.argmin(np.abs(zz - height))
        zsep[:,:,:zind] *= -1.0

        xprojection = xsep / (s + rbead)
        yprojection = ysep / (s + rbead)
        zprojection = zsep / (s + rbead)

        #s = allseps[ind][0]
        #yprojection = allseps[ind][1]
        #zprojection = allseps[ind][2]

        prefac = ((2. * G * m * rhobead * np.pi) / (3. * (rbead + s)**2))

        yukterm = 3 * yuklambda**2 * (rbead + s + yuklambda) * func * np.exp( - s / yuklambda )

        xtotforce = np.sum(prefac * yukterm * xprojection)
        ytotforce = np.sum(prefac * yukterm * yprojection)
        ztotforce = np.sum(prefac * yukterm * zprojection)

        # SWAP X AND Y AXES TO MATCH DATA AXES
        yukforcecurveX.append(ytotforce) 
        yukforcecurveY.append(xtotforce)
        yukforcecurveZ.append(ztotforce)

    yukforcecurveX = np.array(yukforcecurveX)
    yukforcecurveY = np.array(yukforcecurveY)
    yukforcecurveZ = np.array(yukforcecurveZ)

    results_dic[rbead][sep][height][yuklambda] = \
                (GforcecurveX, GforcecurveY, GforcecurveZ, yukforcecurveX, yukforcecurveY, yukforcecurveZ)

stop = time.time()
'''


start2 = time.time()

GforcecurveX2 = []
GforcecurveY2 = []
GforcecurveZ2 = []
for ind, xpos in enumerate(beadposvec2):
    beadpos = [xpos, sep+rbead, height]

    s = dist_p_arrp(beadpos, xx2, yy2, zz2) - rbead

    # These are used to compute projections and thus need to maintain sign
    xsep = dist_p_arrp([xpos, 0, 0], xx2, yzeros2, zzeros2)
    xind = np.argmin(np.abs(xx2 - xpos))
    xsep[:xind,:,:] *= -1.0

    # We a priori know that all yseps should be negative
    ysep = dist_p_arrp([0, sep+rbead, 0], xzeros2, yy2, zzeros2)
    ysep *= -1.0

    zsep = dist_p_arrp([0, 0, height], xzeros2, yzeros2, zz2)
    zind = np.argmin(np.abs(zz2 - height))
    zsep[:,:,:zind] *= -1.0

    xprojection = xsep / (s + rbead)
    yprojection = ysep / (s + rbead)
    zprojection = zsep / (s + rbead)

    #s = allseps[ind][0]
    #yprojection = allseps[ind][1]
    #zprojection = allseps[ind][2]

    prefac = ((2. * G * m2 * rhobead * np.pi) / (3. * (rbead + s)**2))

    xtotforce = np.sum(prefac * Gterm * xprojection)
    ytotforce = np.sum(prefac * Gterm * yprojection)
    ztotforce = np.sum(prefac * Gterm * zprojection)

    # SWAP X AND Y AXES TO MATCH DATA AXES
    GforcecurveX2.append(ytotforce)
    GforcecurveY2.append(xtotforce)
    GforcecurveZ2.append(ztotforce)

GforcecurveX2 = np.array(GforcecurveX2)
GforcecurveY2 = np.array(GforcecurveY2)
GforcecurveZ2 = np.array(GforcecurveZ2)

print('Computed normal grav')
sys.stdout.flush()



for yukind, yuklambda in enumerate(lambdas):
    per = int(100. * float(yukind) / float(len(lambdas)))
    if not per % 1:
        print(str(per) + ',', end=' ')
    sys.stdout.flush()

    func = np.exp(-2. * rbead / yuklambda) * (1. + rbead / yuklambda) + rbead / yuklambda - 1.

    yukforcecurveX2 = []
    yukforcecurveY2 = []
    yukforcecurveZ2 = []
    for ind, xpos in enumerate(beadposvec2):

        beadpos = [xpos, sep+rbead, height]

        s = dist_p_arrp(beadpos, xx2, yy2, zz2) - rbead

        # These are used to compute projections and thus need to maintain sign
        xsep = dist_p_arrp([xpos, 0, 0], xx2, yzeros2, zzeros2)
        xind = np.argmin(np.abs(xx2 - xpos))
        xsep[:xind,:,:] *= -1.0

        # We a priori know that all yseps should be negative
        ysep = dist_p_arrp([0, sep+rbead, 0], xzeros2, yy2, zzeros2)
        ysep *= -1.0

        zsep = dist_p_arrp([0, 0, height], xzeros2, yzeros2, zz2)
        zind = np.argmin(np.abs(zz2 - height))
        zsep[:,:,:zind] *= -1.0

        xprojection = xsep / (s + rbead)
        yprojection = ysep / (s + rbead)
        zprojection = zsep / (s + rbead)


        #s = allseps[ind][0]
        #yprojection = allseps[ind][1]
        #zprojection = allseps[ind][2]

        prefac = ((2. * G * m2 * rhobead * np.pi) / (3. * (rbead + s)**2))

        yukterm = 3 * yuklambda**2 * (rbead + s + yuklambda) * func * np.exp( - s / yuklambda )

        xtotforce = np.sum(prefac * yukterm * xprojection)
        ytotforce = np.sum(prefac * yukterm * yprojection)
        ztotforce = np.sum(prefac * yukterm * zprojection)

        # SWAP X AND Y AXES TO MATCH DATA AXES
        yukforcecurveX2.append(ytotforce) 
        yukforcecurveY2.append(xtotforce)
        yukforcecurveZ2.append(ztotforce)

    yukforcecurveX2 = np.array(yukforcecurveX2)
    yukforcecurveY2 = np.array(yukforcecurveY2)
    yukforcecurveZ2 = np.array(yukforcecurveZ2)

    results_dic2[rbead][sep][height][yuklambda] = \
                (GforcecurveX2, GforcecurveY2, GforcecurveZ2, \
                 yukforcecurveX2, yukforcecurveY2, yukforcecurveZ2)

partial_sim = results_dic2[rbead][sep][height][lambdas[0]]

GX2 = interp.interp1d(beadposvec2, partial_sim[0], kind='cubic')
GY2 = interp.interp1d(beadposvec2, partial_sim[1], kind='cubic')
GZ2 = interp.interp1d(beadposvec2, partial_sim[2], kind='cubic')
yukX2 = interp.interp1d(beadposvec2, partial_sim[3], kind='cubic')
yukY2 = interp.interp1d(beadposvec2, partial_sim[4], kind='cubic')
yukZ2 = interp.interp1d(beadposvec2, partial_sim[5], kind='cubic')


newGX = np.zeros(len(beadposvec))
newGY = np.zeros(len(beadposvec))
newGZ = np.zeros(len(beadposvec))

newyukX = np.zeros(len(beadposvec))
newyukY = np.zeros(len(beadposvec))
newyukZ = np.zeros(len(beadposvec))

finger_inds = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])

def find_ind(pos):
    if np.abs(pos) <= 25e-6:
        ind = 0
    elif np.abs(pos) > 25e-6:
        if np.abs(pos) <= 75e-6:
            if pos > 0:
                ind = 1.0
            elif pos < 0:
                ind = -1.0
        elif np.abs(pos) > 75e-6:
            if np.abs(pos) <= 125e-6:
                if pos > 0:
                    ind = 2.0
                if pos < 0:
                    ind = -2.0
            elif np.abs(pos) > 125e-6:
                if np.abs(pos) <= 175e-6:
                    if pos > 0:
                        ind = 3.0
                    if pos < 0:
                        ind = -3.0
                elif np.abs(pos) > 175e-6:
                    if pos > 0:
                        ind = 4.0
                    if pos < 0:
                        ind = -4.0

    newpos = pos - ind * 50e-6

    return ind, newpos

for ind, pos in enumerate(beadposvec):
    if np.abs(pos) - 225e-6 > 0:
        newGX[ind] = 0.0
        newGY[ind] = 0.0
        newGZ[ind] = 0.0
        newyukX[ind] = 0.0
        newyukY[ind] = 0.0
        newyukZ[ind] = 0.0
        continue

    finger_ind, newpos = find_ind(pos)
    #testpos = pos + 250e-6
    #finger_ind, mod = divmod(testpos - 25e-6, 50e-6)
    #newpos = mod - 25e-6
    width = 50.0e-6
    newGX[ind] = np.sum(GX2(newpos + (finger_inds+finger_ind) * width))
    newGY[ind] = np.sum(GY2(newpos + (finger_inds+finger_ind) * width))
    newGZ[ind] = np.sum(GZ2(newpos + (finger_inds+finger_ind) * width))    
    newyukX[ind] = np.sum(yukX2(newpos + (finger_inds+finger_ind) * width))
    newyukY[ind] = np.sum(yukY2(newpos + (finger_inds+finger_ind) * width))
    newyukZ[ind] = np.sum(yukZ2(newpos + (finger_inds+finger_ind) * width))

stop2 = time.time()

partial_sim = (newGX, newGY, newGZ, newyukX, newyukY, newyukZ)


#full_sim = results_dic[rbead][sep][height][lambdas[0]]
#pickle.dump(full_sim, open('/home/charles/gravity/code/full_sim.p', 'wb') )

full_sim = pickle.load( open('/home/charles/gravity/code/full_sim.p', 'rb') )

#print "Full sim time: ", stop - start
print("Periodic sim time: ", stop2 - start2)

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

for ind in [0,1,2,3,4,5]:
    
    partial = partial_sim[ind]
    full = full_sim[ind]

    partial = signal.detrend(partial)
    full = signal.detrend(full)

    partial_popt, _ = opti.curve_fit(quadratic, beadposvec, partial)
    full_popt, _ = opti.curve_fit(quadratic, beadposvec, full)

    #partial = partial - quadratic(beadposvec, *partial_popt)
    #full = full - quadratic(beadposvec, *full_popt)

    plt.figure()
    plt.plot(beadposvec, partial)
    plt.plot(beadposvec, full)

plt.show()



#results_dic['posvec'] = beadposvec

#pickle.dump(results_dic, open(respath, 'wb') )
