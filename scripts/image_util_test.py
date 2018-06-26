####Script for testing image_util.py
import numpy as np
import matplotlib.pyplot as plt
from image_util import *
from beam_profile import *

path_to_measure = '/data/20180622/image_grids/image_grid_beam_crossing_foreward1_good'
path_with_profile = '/data/20180622/image_grids/image_grid_beam_crossing'


profile_path = '/data/20180622/image_grids/image_grid_beam_crossing_profile_good'



s = measure_separation(path_to_measure, path_with_profile, profile_path)
#cents1, es1 = find_beam_crossing(prof_path1)
#cents2, es2 = find_beam_crossing(prof_path2)
#dimgs = ig1.measureGrid(ig2, make_plot = True)
#make plots for post
#images = [igs[5].images[31].imarr, igs[5].images[28].imarr, \
#          igs[5].images[3].imarr, igs[5].images[67].imarr]
#plotImages(images)

