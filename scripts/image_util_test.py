####Script for testing image_util.py
import numpy as np
import matplotlib.pyplot as plt
from image_util import *
from beam_profile import *

path1 = '/data/20180529/imaging_tests/p0/image_grid_good'
path2 = '/data/20180529/imaging_tests/p1/image_grid'

prof_path1 = '/data/20180529/imaging_tests/p0/xprofile'
prof_path2 = '/data/20180529/imaging_tests/p1/profile'


ig1 = ImageGrid(path1)
ig2 = ImageGrid(path2)

#cents1, es1 = find_beam_crossing(prof_path1)
#cents2, es2 = find_beam_crossing(prof_path2)
#dimgs = ig1.measureGrid(ig2, make_plot = True)
#make plots for post
#images = [igs[5].images[31].imarr, igs[5].images[28].imarr, \
#          igs[5].images[3].imarr, igs[5].images[67].imarr]
#plotImages(images)

