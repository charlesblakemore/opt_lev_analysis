####Script for testing image_util.py
import numpy as np
import matplotlib.pyplot as plt
from image_util import *
from beam_profile import *

###
### Calibration files
paths_with_profile =\
        ['/data/20180625/bead1/imgrids/image_grid_final_forward',\
        '/data/20180625/bead1/imgrids/image_grid_final_forward2',\
        '/data/20180625/bead1/imgrids/image_grid_final_forward3']

profile_paths =\
        ['/data/20180625/bead1/imgrids/image_grid_final_forward_profile',\
        '/data/20180625/bead1/imgrids/image_grid_final_forward2_profile',\
        '/data/20180625/bead1/imgrids/image_grid_final_forward3_profile']

def cer(directory):
    cmean, error = find_beam_crossing(directory)
    return [cmean, error]


igs = map(ImageGrid, paths_with_profile)
mrel1 = lambda ig: igs[0].measureGrid(ig)

cents = np.array(map(cer, profile_paths))
dig = np.array(map(mrel1, igs))

diffs_igs = dig[:, 0, 0] - dig[0, 0, 0]
diffs_profs = cents[0, 0] - cents[:, 0]

es = dig[:, 0, 1]

plt.plot(diffs_profs, diffs_igs, 'o')
plt.show()

plt.errorbar(diffs_profs, diffs_igs - diffs_profs, es, fmt = 'o')
plt.xlabel("Position Difference from profiles [um]")
plt.ylabel("Image analysis - profiles [um]")
plt.show()

