import bead_util as bu
import matplotlib.pyplot as plt

cal_path = '/data/20150512/Bead1/100ecal_positive'
neutral = '/data/20150512/Bead1/neutral'
charge = '/data/20150512/Bead1/100e'

e_charge = 1.602e-19

cal = bu.calibrate_dc(cal_path, 11*10.9, make_plt = True)

ampsn, distsn =  bu.get_DC_force(neutral, -5)
ampsc, distsc =  bu.get_DC_force(charge, -5)
fn = ampsn*cal
fc = ampsc*cal

plt.plot(distsn, fn, 'x', label = 'neutral')
plt.plot(distsc, fc, 'x',label = '11 charges')
plt.legend()
plt.xlabel('distance [um]')
plt.ylabel('force[N]')
plt.show()

E = (fn-fc)/(11*10.9*e_charge)

plt.plot(distsn, E, 'x')
plt.xlabel('distance')
plt.ylabel('inferred electric field [V/m]')
plt.show()
