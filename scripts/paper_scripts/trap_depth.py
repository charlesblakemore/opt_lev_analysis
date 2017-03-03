import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu

path = "/data/20140729/Bead4/linearity/"
fname = "urmbar_xyzcool_167mV_200Hz_0.h5"

rc, fp, _ = bu.get_calibration(path+"../2mbar_zcool_50mV_40Hz.h5", [10,300], make_plot=True)
print rc
rc2, _, _ = bu.get_calibration(path+fname, [10,300], make_plot=True)
#J1x = bu.get_avg_noise([path + fname,],0,[0,0,0])

print "trap_depth: ", 0.5*bu.bead_mass * (2*np.pi*fp[1])**2 * (3e-6)**2 * 6.2e18
print "trap_depth 2: ", 0.167*200/1e-3 * 3e-6


plt.show()

