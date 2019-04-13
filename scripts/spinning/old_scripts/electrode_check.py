import numpy as np
import matplotlib.pyplot as plt
from hs_digitizer import *
import os

path = "/data/20181204/bead1/high_speed_digitizer/electrode_test"
fname = "5v_5v_5v_25k_0.h5"

obj = hsDat(os.path.join(path, fname))
ns = 1
nf = 100000
plt.plot(obj.dat[:nf:ns, 1], obj.dat[:nf:ns, 2], 'o')
plt.xlabel("Vx")
plt.ylabel("Vy")

plt.show()

