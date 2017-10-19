import numpy as np
import matplotlib.pyplot as plt

import bead_util as bu
import configuration as config


#xfile = '/data/20171018/chopper_profiling/xprof.h5'
xfile = '/data/20171018/chopper_profiling/xprof_bright_fast.h5'

#yfile = '/data/20171018/chopper_profiling/yprof.h5'
yfile = '/data/20171018/chopper_profiling/yprof_bright_fast.h5'



xfilobj = bu.DataFile()
xfilobj.load(xfile)
xfilobj.load_other_data()

yfilobj = bu.DataFile()
yfilobj.load(yfile)
yfilobj.load_other_data()



rawx = xfilobj.other_data[-1]
rawy = yfilobj.other_data[-1]

profx = xfilobj.other_data[-2]
profy = yfilobj.other_data[-2]

gradx = np.gradient(rawx)
grady = np.gradient(rawy)




plt.plot(rawx)
plt.plot(rawy)

plt.figure()
plt.plot(profx*2)
plt.plot(gradx)

plt.figure()
plt.plot(profy*2)
plt.plot(grady)

plt.show()

