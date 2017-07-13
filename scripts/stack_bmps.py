import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Image
import cv2
import os
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
import glob
import re
from scipy import stats

path = "/data/20170712/images_to_stack"

files = glob.glob(path + "/*.bmp")


for f in files:
    if f==files[0]:
        img = mpimg.imread(f, 0).astype('float64')
    else:
        img += mpimg.imread(f, 0).astype('float64')

plt.imshow(img)
plt.show()
