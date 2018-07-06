import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import grav_util as gu

reload_dat = False

if reload_data:
    path = "/data/20180625/bead1/grav_data/no_shield/X60-80um_Z15-25um_17Hz"
    files = bu.find_all_fnames(path)
    dict = gu.get_data_at_harms(files[10:])


