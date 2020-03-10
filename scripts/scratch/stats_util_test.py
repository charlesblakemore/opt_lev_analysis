import numpy as np
import matplotlib.pyplot as plt
import time

import bead_util as bu

neval = 100

samples = np.random.randn(5000)

test = bu.ECDF(samples)
interp_func = test.build_interpolator(npts=1000)
testpdf = test.PDF(npts=1000, smoothing=0.01, limfacs=(3.0,3.0))
# test2 = bu.ECDF2(samples)

xarr = np.linspace(-5, 5, 100)

# start1 = time.time()
# for i in range(neval):
#     test(xarr)
# stop1 = time.time()

# start2 = time.time()
# for i in range(neval):
#     test2(xarr)
# stop2 = time.time()

# print( '      Loop  :  {:0.2f} ms per eval'.format((stop1 - start1)*1000.0/neval) )
# print( 'Vectorized  :  {:0.2f} ms per eval'.format((stop2 - start2)*1000.0/neval) )


plt.plot(xarr, test(xarr))
plt.plot(xarr, interp_func(xarr), ls='--')

plt.figure()
plt.hist(samples, density=True, bins=20)
plt.plot(xarr, np.abs(testpdf(xarr)))
plt.show()