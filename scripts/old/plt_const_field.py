import bead_util as bu
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

path11e = "/data/20150512/Bead1/11ecal_2"
path100epos = "/data/20150512/Bead1/100ecal_positive"
path100eneg = "/data/20150512/Bead1/100ecal"


amps11e, dcs11e = bu.get_DC_force(path11e, ind = -5)
amps100e, dcs100e = bu.get_DC_force(path100epos, ind = -5)
amps100en, dcs100en = bu.get_DC_force(path100eneg, ind = -5)

amps100et = np.concatenate((amps100e, amps100en))
dcs100et = np.concatenate((dcs100e ,-1.*dcs100en))

spars = [1e2, -6.]

bf11e, bc11e = opt.curve_fit(bu.lin, dcs11e, amps11e)
bf100e, bc100e = opt.curve_fit(bu.lin, dcs100et, amps100et)

dc11e = -1.*bf11e[1]/bf11e[0]
dc100e = -1.*bf100e[1]/bf100e[0]
crat = bf100e[0]/bf11e[0]


print 'dc11e = ', dc11e
print 'dc100e = ', dc100e
print 'crat = ', crat


plt.plot(dcs100et, amps100et, 'o')
plt.show()
