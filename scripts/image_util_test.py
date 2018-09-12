####Script for testing image_util.py
import numpy as np
import matplotlib.pyplot as plt
from image_util import *
from beam_profile import *

###

#path_to_measure = '/data/20180625/bead1/imgrids/image_grid_final_position'
path_to_measure = '/data/20180704/bead1/imgrids/imgrid_forward_72'
#path_to_measure = '/data/20180704/bead1/imgrids/imgrids_after/initial_position'
###


### Calibration files
path_with_profile = '/data/20180625/bead1/imgrids/image_grid_final_forward'
profile_path = '/data/20180625/bead1/imgrids/image_grid_final_forward_profile'

#profile_path = '/data/20180704/bead1/imgrids/imgrids_after/over_trap_position_profile_forward_more'
#path_with_profile = '/data/20180704/bead1/imgrids/imgrids_after/over_trap_position_grid_forward_more'


ig_data = ImageGrid(path_to_measure)
ig_cal = ImageGrid(path_with_profile)

plt.imshow(ig_data.images[0].imarr)
plt.figure()
plt.imshow(ig_cal.images[0].imarr)
plt.show()


s = measure_separation(path_to_measure, path_with_profile, profile_path)
print s



#cents1, es1 = find_beam_crossing(prof_path1)
#cents2, es2 = find_beam_crossing(prof_path2)
#dimgs = ig1.measureGrid(ig2, make_plot = True)
#make plots for post
#images = [igs[5].images[31].imarr, igs[5].images[28].imarr, \
#          igs[5].images[3].imarr, igs[5].images[67].imarr]
#plotImages(images)

