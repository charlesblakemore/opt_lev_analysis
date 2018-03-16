####Script for testing image_util.py
import numpy as np
import matplotlib.pyplot as plt
from image_util import *


path = '/data/20171129/imgrid_pico_step'
paths = getPaths(path)
igs = map(ImageGrid, paths)

#make plots for post
#images = [igs[5].images[31].imarr, igs[5].images[28].imarr, \
#          igs[5].images[3].imarr, igs[5].images[67].imarr]
#plotImages(images)

