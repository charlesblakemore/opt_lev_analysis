import numpy as np
import argparse
import cv2

import matplotlib.pyplot as plt

path_base = '/home/charles/opt_lev_analysis/scripts/image_analysis/bead_radii/'
im = '20190108_bead-radii-test-1.TIF'
im = '20190108_bead-radii-test-1_8000x.TIF'

verbose = True
plot = True



####################################################################


image = cv2.imread(path_base + im)

output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

## Crop various parts of the image
sem = gray[:484,:]
bar = gray[484:,:]

scale = bar[20:,375:625]

## Determine the pixel -> micron ratio from the scale bar
scale_1d = scale[9,:]
for pixel_ind, pixel in enumerate(scale_1d):
    if pixel > 126:
        start_ind = pixel_ind
        break
for pixel_ind, pixel in enumerate(scale_1d[::-1]):
    if pixel > 126:
        stop_ind = len(scale_1d) - pixel_ind - 1
        break
len = stop_ind - start_ind + 1
if verbose:
    print 'resolution [um]: ', 5.0 / float(len)

if plot:
    plt.imshow(sem, cmap='gray')
    plt.figure()
    plt.imshow(bar, cmap='gray')
    plt.figure()
    plt.imshow(scale, cmap='gray')
    plt.plot([start_ind, stop_ind], [9, 9])
    plt.show()
