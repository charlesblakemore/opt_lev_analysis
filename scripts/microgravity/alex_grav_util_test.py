import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import grav_util as gu

dat_dir = "/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz"

files = bu.find_all_fnames(dat_dir)

badict = gu.get_data_at_harms(files)
