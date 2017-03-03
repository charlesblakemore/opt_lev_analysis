import os
import numpy as np

gap_list = [5.0e-6, 7.5e-6, 10e-6]
lam_list = np.logspace(-1.0,2.0,20)*1e-6
print lam_list

for i in range(len(gap_list)):
    for j in range(len(lam_list)):
        cc = "bsub -q xlong batchPython_cyl.sh %e %e" % (gap_list[i], lam_list[j])
    
        print cc
        os.system(cc)

