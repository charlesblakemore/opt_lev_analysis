import numpy as np
import time

import temp_bead_util as bu


filname = '/data/20180528/trig_test/beam_on_4.h5'

raw_dat, quad_dat, pos_dat = bu.getdata(filname)

## accept data within a week
diff_thresh = 1.0 * (1.0 / (7.0 * 24.0 * 3600.0))
ctime = time.time()

timestamp = False
writing_data = False

mask1 = np.uint64(0x000000000000FFFF)
mask2 = np.uint64(0x00000000FFFF0000)
mask3 = np.uint64(0x0000FFFF00000000)
mask4 = np.uint64(0x000000000000FFFF)

for ind, dat in enumerate(raw_dat):
    if (ctime - float(dat) * 10**(-9)) < diff:
        timestamp = True
    if timestamp:
        if ind == 0:
            assert 
        
