import os, time
import numpy as np

gap_list = np.array([0.5, 2., 5., 10., 20.])*1e-6
lam_list = np.logspace(-1.0,2.0,40)*1e-6
#print lam_list

for i in range(len(gap_list)):
    for j in range(len(lam_list)):
        cc = "bsub -q medium python force_calc_v3.py %e %e" % (gap_list[i], lam_list[j])
    
        print cc
        os.system(cc)
        time.sleep(2)
