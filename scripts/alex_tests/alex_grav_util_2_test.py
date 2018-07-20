import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import grav_util_2 as gu2

path = "/data/20180704/bead1/grav_data/shield_1s_1h"
files = bu.find_all_fnames(path)

AD = gu2.AggregateData(files)
