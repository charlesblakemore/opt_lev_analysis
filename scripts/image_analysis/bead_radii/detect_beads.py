import numpy as np
import argparse
import cv2

import matplotlib.pyplot as plt

path_base = '/home/charles/opt_lev_analysis/scripts/image_analysis/bead_radii/'
#im = '20190108_bead-radii-test-1.TIF'
#im = '20190108_bead-radii-test-1_8000x.TIF'
#im = '20190108_bead-radii-test-1_5000x.TIF'
im = '20190108_bead-radii-test-2_10000x.TIF'
#im = '20190108_bead-radii-test-3_5000x.TIF'

verbose = True
plot_scale = False



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
resolution = 5.0 / float(len)
if verbose:
    print('length of scale: ', len)
    print('resolution [um]: ', resolution)

if plot_scale:
    #plt.imshow(sem, cmap='gray')
    #plt.figure()
    plt.imshow(bar, cmap='gray')
    plt.figure()
    plt.imshow(scale, cmap='gray')
    plt.plot([start_ind, stop_ind], [9, 9])
    plt.show()


##### Find the beads


circles = cv2.HoughCircles(sem, cv2.HOUGH_GRADIENT, dp=1.3, \
                           minDist=200, param1=75, param2=100)#, \
                           #minRadius=20, maxRadius=100)
print(circles)

fig = plt.figure()
ax = plt.subplot(111)
ax.imshow(sem, cmap='gray')
plot_circles = []
for circle in circles[0]:
    print('RADIUS DETECTED: ', circle[2] * resolution) 
    plot_circles.append( plt.Circle((circle[0], circle[1]), circle[2], \
                                    color='r', fill=False) )
    ax.add_artist(plot_circles[-1])
plt.show()
