import numpy as np
import matplotlib.pyplot as plt
import bead_util as bu
import grav_util as gu

reprocess = False
save_name = "badict.p"

dat_dir = "/data/20180625/bead1/grav_data/shield/X50-75um_Z15-25um_17Hz"
theory_data_dir = '/data/grav_sim_data/2um_spacing_data/'



if reprocess:

    files = bu.find_all_fnames(dat_dir)
    badict = gu.get_data_at_harms(files)
    gu.save_fildat(save_name, badict)

else:

    badict = gu.load_fildat(save_name)


dadict = gu.filedat_tup_to_dict(badict)
gf, yf, lams, lims = gu.build_mod_grav_funcs(theory_data_dir)
yf = np.array(yf)
lm25umind = np.argmin((lams - 25E-6)**2)
gu.apply_operation_to_file(dadict, gu.generate_template_fft, yf[:, lm25umind])

